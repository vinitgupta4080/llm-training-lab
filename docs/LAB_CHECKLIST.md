# Practical lab checklist

Use this as the execution board. Commit each checked item with its experiment note.

## 00 — Setup

- [ ] Print Python, PyTorch, device, and dtype information.
- [ ] Demonstrate broadcasting and calculate tensor memory by hand.
- [ ] Show that the same seed reproduces initialization and batches.
- [ ] Benchmark matrix multiplication at three sizes.

## 01 — Tokenization

- [ ] Run `tokenizer_lab.py` and explain multilingual token counts.
- [ ] Add serialization and tests for malformed byte sequences.
- [ ] Implement pair-frequency counting and one BPE merge.
- [ ] Compare compression ratio and vocabulary size on two domains.

## 02 — Autograd and optimization

- [ ] Implement a scalar `Value` node with topological backpropagation.
- [ ] Gradient-check a two-layer MLP with finite differences.
- [ ] Implement SGD, momentum, and AdamW updates directly from equations.
- [ ] Plot loss and gradient norms; trigger and fix an exploding gradient.

## 03 — Language modeling

- [ ] Train the included bigram baseline.
- [ ] Add deterministic train/validation splitting and evaluation mode.
- [ ] Implement greedy, temperature, and top-k sampling.
- [ ] Compare learned transitions with empirical corpus transitions.

## 04 — Transformer

- [ ] Annotate every tensor shape in `CausalSelfAttention.forward`.
- [ ] Write an independent causal-mask test.
- [ ] Overfit one batch, then train on the corpus.
- [ ] Ablate positional embeddings, head count, and residual connections.
- [ ] Calculate parameters and activation memory; compare with measurements.

## 05 — Pretraining

- [ ] Build a streaming token dataset with packing and document boundaries.
- [ ] Add warmup + cosine decay, gradient accumulation, and mixed precision.
- [ ] Save model, optimizer, scheduler, RNG state, step, and config.
- [ ] Prove a resumed run matches an uninterrupted run.
- [ ] Track tokens/second, utilization proxy, memory, and validation loss.

## 06 — SFT

- [ ] Define a versioned conversation schema and chat template.
- [ ] Mask prompt/padding tokens so only target responses contribute loss.
- [ ] Inspect decoded examples alongside token IDs and masks.
- [ ] Compare full fine-tuning with frozen-layer tuning on fixed evaluations.

## 07 — PEFT

- [ ] Train the included `LoRALinear` on a tiny adaptation task.
- [ ] Inject adapters into attention projections and count trainable parameters.
- [ ] Ablate rank, alpha, dropout, and target modules.
- [ ] Merge adapters and assert output equivalence.
- [ ] Explain 8-bit/4-bit base weights and higher-precision adapter computation.

## 08 — Preference optimization

- [ ] Construct chosen/rejected pairs and audit ambiguous preferences.
- [ ] Implement Bradley–Terry reward-model loss.
- [ ] Cache reference-model log probabilities.
- [ ] Implement a DPO-style loss and ablate beta.
- [ ] Test length bias, reward hacking, and degradation on base capabilities.

## 09 — Evaluation

- [ ] Freeze an evaluation manifest before tuning.
- [ ] Report loss, task success, calibration, latency, and safety slices.
- [ ] Add bootstrap confidence intervals and paired comparisons.
- [ ] Check train/eval contamination and manually analyze failures.

## 10 — Inference

- [ ] Add top-p sampling, repetition handling, and stop sequences.
- [ ] Implement a per-layer KV cache and test cached/uncached equivalence.
- [ ] Benchmark latency and throughput by prompt/output length and batch size.
- [ ] Quantize a model and report memory, speed, and quality deltas.

## 11 — Scaling and capstone

- [ ] Estimate parameters, tokens, FLOPs, activation memory, and checkpoint storage.
- [ ] Design data-, tensor-, and pipeline-parallel alternatives.
- [ ] Define monitoring, recovery, data lineage, license, and release gates.
- [ ] Complete the capstone deliverables in `LEARNING_PLAN.md`.

