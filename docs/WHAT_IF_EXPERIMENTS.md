# What-if experiments: investigate, do not merely run

Every experiment follows the same scientific protocol:

1. State a mechanistic hypothesis before running code.
2. Change exactly one independent variable.
3. Keep seed, data order, token budget, evaluation set, and logging fixed.
4. Run at least three seeds when comparing noisy training outcomes.
5. Report mean, spread, learning curves, parameter count, tokens, and wall time.
6. Name an observation that would falsify your explanation.
7. Inspect individual predictions; aggregate loss can hide failure modes.

Do not ask only “did loss improve?” Ask *why*, *where*, *at what cost*, and *under
which distribution*.

## 1. Tokenization and data representation

### What if the vocabulary is larger?

- Change: byte vocabulary vs BPE vocabularies of 512, 2K, and 8K.
- Hold fixed: raw training text and total optimizer steps.
- Measure: tokens/character, embedding parameters, sequences/second, validation bits
  per byte, rare-word fragmentation, and multilingual slices.
- Hypothesis: larger vocabularies shorten sequences but enlarge embeddings and create
  sparse rare-token estimates. Token-level perplexities are not directly comparable.
- Falsifier: an 8K vocabulary improves every slice and compute metric despite a corpus
  too small to estimate most tokens reliably.
- Technical question: why is bits-per-byte safer than token perplexity across tokenizers?

### What if document boundaries are removed?

- Compare explicit EOS boundaries with one concatenated token stream.
- Measure loss within five tokens of boundaries and generated topic transitions.
- Explain whether the model learns spurious cross-document transitions.

### What if duplicated examples occupy 20% of training data?

- Track training loss, clean validation loss, exact memorization, and exposure frequency.
- Add duplicates to only one controlled slice; do not contaminate evaluation.
- Explain why lower training loss can coexist with worse generalization.

## 2. Optimization and gradient flow

### What if initialization variance is 10× too large or too small?

- Log per-layer activation mean/std, gradient RMS, update-to-weight ratio, and loss.
- Inspect results before the first update and after 1, 10, and 100 steps.
- Predict which nonlinearities saturate and whether residual depth amplifies variance.
- Falsifier: all layer statistics and convergence remain indistinguishable.

### What if LayerNorm is moved after each residual addition?

- Compare pre-norm and post-norm models at depths 2, 8, and 16.
- Hold parameter count approximately fixed.
- Measure gradient norm by layer, divergence rate, and steps to target loss.
- Explain the identity gradient path in a pre-norm residual block.

### What if AdamW epsilon changes from 1e-8 to 1e-4?

- Measure effective update magnitude for parameters with small second moments.
- Repeat in float32 and mixed precision.
- Derive when epsilon dominates `sqrt(v_hat)`.

### What if gradient clipping is removed?

- Record pre-clip and post-clip norms and the fraction of clipped steps.
- Induce a loss spike using a larger learning rate.
- Distinguish a safeguard from a fix: clipping may hide unstable configuration.

## 3. Context and autoregressive objectives

### What if context length doubles while batch size stays fixed?

- Measure attention activation memory, tokens/second, examples/second, and validation loss.
- Derive the `O(T²)` attention-score term and identify non-attention linear terms.
- Then hold *tokens per batch* fixed. Explain why this is a different experiment.

### What if examples are padded instead of packed?

- Compare useful-token fraction, wall time, loss-mask correctness, and samples/second.
- Assert that padding tokens contribute zero loss.
- Test whether examples can attend across packed document boundaries.

### What if teacher forcing is replaced by sampled previous tokens during training?

- Predict how the target distribution and gradient estimator change.
- Measure convergence and generation errors, controlling total tokens.
- Explain exposure bias without assuming the proposed intervention must help.

## 4. Attention internals

### What if `1/sqrt(head_width)` scaling is removed?

- Before training, log attention-logit variance and attention entropy by layer.
- Repeat for head widths 8, 32, and 128.
- Derive `Var(q·k)` under independent unit-variance components.
- Falsifier: entropy and gradient statistics do not change with head width.

### What if the causal mask has an off-by-one error?

- Create a synthetic dataset where token `t+1` is random and unpredictable from history.
- Compare a correct mask with one that reveals the target token.
- A suspiciously near-zero loss is evidence of leakage, not intelligence.
- Required test: perturb future inputs and prove earlier logits are invariant.

### What if heads share projections?

- Compare independent Q/K/V projections, multi-query attention, and fully shared heads.
- Match parameter count where possible; report KV-cache bytes per generated token.
- Measure quality, training speed, and decoding memory separately.

### What if positional embeddings are removed?

- Use tasks that distinguish permutations with identical token sets.
- Compare learned absolute positions and a relative/rotary implementation.
- Evaluate beyond trained context only if the positional method defines that behavior.

### What if attention weights look interpretable but are not causal?

- Intervene by zeroing or patching selected value vectors and measure logit changes.
- Compare attention visualization with causal effect.
- Explain why a large attention weight need not imply decisive model behavior.

## 5. Architecture ablations

### What if width is doubled versus depth doubled?

- Construct approximately parameter-matched models.
- Report exact parameter counts, FLOPs estimate, peak memory, throughput, and loss.
- Fit loss against consumed tokens, not only steps.
- Identify whether the dataset or training budget is too small for a fair conclusion.

### What if the MLP expansion ratio changes from 4 to 2 or 8?

- Measure parameter allocation between attention, embeddings, and MLP.
- Evaluate syntax-like and memorization-heavy slices separately.

### What if token embeddings and output weights are not tied?

- Measure parameter count, early optimization, final loss, and embedding geometry.
- Explain the inductive bias introduced by weight tying.

### What if GELU is replaced by ReLU, SiLU, or SwiGLU?

- Parameter-match the gated and non-gated MLPs.
- Log activation sparsity, gradient RMS, throughput, and validation loss.

## 6. Training schedules and scale

### What if effective batch size increases 8×?

- Compare constant learning rate and scaled learning-rate variants.
- Hold total tokens fixed and log gradient-noise estimates.
- Separate microbatch size from gradient accumulation and data-parallel batch.
- Technical question: why are eight accumulated microbatches not identical to one large
  microbatch when dropout, normalization, or data order changes?

### What if warmup is removed?

- Inspect update-to-weight ratio and second-moment estimates during the first 100 steps.
- Repeat at two model widths and learning rates.
- Explain why adaptive optimizer statistics are unreliable early in training.

### What if you train longer on less data versus once over more data?

- Hold token budget fixed; change unique-token fraction.
- Measure generalization, memorization, and domain coverage.
- Record the point at which repeated data stops giving comparable returns.

### What if a checkpoint resumes without RNG or data-loader state?

- Compare uninterrupted and resumed runs step-by-step.
- Hash batches and parameters; locate the first divergence.
- Define “reproducible” at bitwise, statistical, and outcome levels.

## 7. Supervised fine-tuning

### What if prompt tokens are included in the loss?

- Compare full-sequence and response-only loss masks.
- Inspect prompt copying, response quality, and token-level loss by region.
- Assert mask alignment after chat-template tokenization.

### What if SFT data is 10× larger but noisier?

- Create controlled label corruption and formatting errors.
- Compare quantity-matched and quality-matched subsets.
- Slice evaluation by instruction type and response length.

### What if domain SFT causes catastrophic forgetting?

- Evaluate base capabilities before and after tuning.
- Vary learning rate, epochs, frozen layers, and general-data mixing one at a time.
- Measure parameter drift and KL divergence from the base model.

## 8. LoRA and quantized tuning

### What if LoRA rank increases from 1 to 64?

- Report trainable parameters, adapter bytes, speed, final loss, and singular values of
  the learned update.
- Determine whether effective rank actually grows with configured rank.

### What if adapters target only Q/V versus every linear layer?

- Keep rank fixed first, then parameter-match a second comparison.
- Measure task performance and base-capability drift.
- Explain why parameter-matched and rank-matched comparisons answer different questions.

### What if LoRA alpha changes while rank changes?

- First hold `alpha/rank` fixed, then hold alpha fixed.
- Log update norm relative to frozen weight norm.
- Explain why changing rank silently changes scale in one design.

### What if the frozen base is quantized to 4-bit?

- Compare memory, throughput, adapter gradients, convergence, and merged-model quality.
- Identify compute dtype, storage dtype, quantization blocks, and double quantization.
- Never claim “4-bit training” without saying which tensors remain higher precision.

## 9. Preference optimization

### What if DPO beta is too small or too large?

- Log chosen/rejected margins, KL from reference, reward accuracy, and task quality.
- Predict the limiting behavior as beta approaches zero or becomes very large under the
  specific loss parameterization used by your implementation.

### What if preferences correlate with response length?

- Create pairs where quality is held constant and only length differs.
- Measure learned length distribution and normalized log-probability variants.
- Distinguish real task preference from annotator or construction bias.

### What if the reference model is changed?

- Compare the SFT checkpoint, base checkpoint, and a stale reference.
- Hold preference pairs and optimization fixed.
- Explain how the reference anchors policy change.

## 10. Evaluation and inference

### What if validation loss improves but task accuracy falls?

- Slice loss by domain, length, and token frequency.
- Inspect generation protocol and exact-match brittleness.
- Test whether calibration, formatting, or decoding caused the disagreement.

### What if temperature changes from 0.2 to 1.5?

- Measure entropy, distinct n-grams, repetition, task success, and pass@k.
- Separate model quality from sampling policy.
- Use identical prompts and multiple sampling seeds.

### What if KV caching produces slightly different logits?

- Compare cached and full-prefix logits at every generation step.
- Check position indices, causal masks, cache concatenation dimension, and dtype.
- Set an explicit numerical tolerance and explain it.

### What if quantization saves memory but makes the model slower?

- Benchmark prefill and decode separately across batch sizes.
- Record kernel availability, dequantization overhead, memory bandwidth, and device.
- Performance claims without hardware and workload descriptions are incomplete.

## 11. Required technical write-up for every major result

Include:

- exact independent/dependent/control variables
- model/data/config hashes and three or more seeds
- parameter, token, FLOPs, memory, and wall-time accounting
- mean and uncertainty, not only the best run
- learning curves aligned by tokens and compute
- at least five inspected examples and a failure taxonomy
- plausible causal mechanism and competing explanation
- falsification test and next experiment
- limitations: what this small-scale result does not establish about large models

