# Course progress

This is the authoritative resume point. Update it after every meaningful learning session.

## Current state

- Current module: Module 2 — gradients and optimization
- Status: in progress
- Next action: learner explains diminishing returns in the measured 20/200/2,000-step experiment,
  then begins momentum and optimizer comparison.
- Last verified: 2026-07-26; training-loop exercise and controlled learning-rate sweep passing

## Module scoreboard

| Module | Status | Required evidence |
|---|---|---|
| 0 Setup and measurement | Complete | tensor benchmark, memory calculation, reproducibility test |
| 1 Tokenization | Complete | BPE merge experiment and multilingual comparison |
| 2 Autograd and optimization | In progress | scalar autograd, optimizer comparison, gradient profile |
| 3 Language modeling | Scaffolded | validation split, sampling, controlled experiment |
| 4 Transformer | Scaffolded | shape trace, causal test, architecture ablation, profile |
| 5 Pretraining systems | Not started | data pipeline, schedule, exact resume, throughput profile |
| 6 SFT | Not started | chat template, loss-mask tests, full-vs-response loss |
| 7 PEFT | Scaffolded | adapter injection, rank ablation, merge equivalence |
| 8 Preference tuning | Not started | preference loss, beta ablation, bias analysis |
| 9 Evaluation | Not started | frozen manifest, slices, confidence intervals |
| 10 Inference | Not started | KV cache, equivalence tests, latency benchmark |
| 11 Scaling/capstone | Not started | costed systems design and completed capstone |

Status vocabulary: `Not started`, `Scaffolded`, `In progress`, `Blocked`, `Complete`.

## Learner knowledge ledger

### Demonstrated

- Correctly relates dtype bit width to tensor storage: half the bits per element means half
  the storage for the same shape.
- Correctly distinguishes square storage scaling (`N²`, therefore 4× when doubled) from
  matrix-multiplication scaling (`N³`, therefore 8× when doubled).
- Correctly explains that `torch.manual_seed` initializes generator state and subsequent
  random operations consume the sequence and advance that state.
- Correctly predicted that float64 inputs use 2× float32 memory and closely predicted the
  measured CPU matmul latency ratio (4× predicted versus 4.30× measured).
- Correctly explains that dtype fixes the memory ratio while latency depends on system and
  workload factors; still refining which factors apply to local versus distributed execution.
- Correctly explains the `[B,T,C]` hidden-state structure and that a per-channel bias has shape
  `[C]`, matching one token vector before broadcasting across `B` and `T`.
- Correctly created a float32 zero tensor of shape `[2,3,4]` using `torch.zeros` and an explicit
  `dtype=torch.float32` keyword after correcting the initial API syntax.
- Correctly used `numel()` and `element_size()` to calculate storage, and learned that `ndim`
  is an attribute rather than a callable method.
- Correctly retrieved contiguous tensor stride using `hidden.stride()`.
- Correctly constructed a rank-1 float32 bias tensor using `torch.tensor` and reinforced that
  dtype is supplied as the keyword argument `dtype=torch.float32`.
- Correctly used `hidden + bias` to broadcast a `[C]` bias across `[B,T,C]` hidden states.
- Correctly used `bias.expand_as(hidden)` to construct a zero-stride logical view sharing the
  original four-value bias storage.
- Correctly used `hidden.transpose(1,2)` to exchange token/channel axes without copying storage.
- Correctly reset `torch.manual_seed(0)` before matching `torch.randn(5)` calls, completing the
  eight-part Module 0 coding checkpoint.
- Correctly predicted that merging all non-overlapping `a+n` pairs reduces `banana` from six
  base tokens to four BPE tokens.
- Implemented adjacent ordered-pair counting with a `Counter` and index loop after correcting
  range bounds and consistent variable names.
- Correctly explained the BPE training cycle in plain language: count neighboring pairs, select
  the highest-frequency pair, replace each non-overlapping exact match with one new token, then
  recount the changed sequence on the next training iteration.
- Drafted the full count/select/assign/merge decomposition independently. Needs reinforcement on
  Python `Counter`, `+=`, `==`, loop advancement, returning only the selected pair rather than
  `(pair,count)`, preserving unmatched/final tokens, and distinguishing one merge step from the
  outer repeated training loop.
- Integrated the learner's complete one-pass design into runnable code after correcting Python
  syntax and boundary handling; includes count, deterministic selection, merge, and one-pass
  orchestration assertions.
- Correctly recognized BPE training as repeated passes, with the important qualification that
  training stops at its merge budget or when no adjacent pair remains—not necessarily at one token.
- Implemented the repeated `train_bpe` loop using the previously completed count, choose, and
  non-overlapping merge functions; learned rules are saved in deterministic order.
- Correctly implemented `encode_with_merges`: copy the input and replay each learned `(pair,
  new_id)` rule in order without recounting or learning new rules.
- Independently designed an iterative BPE decoder that expands merge rules in reverse training
  order; the idea is correct and avoids explicit recursion.
- Correctly added fail-fast validation for encoded IDs, with only indentation and working-copy
  variable consistency requiring correction.
- Predicted that repeated robot emoji would compress most under an English-trained BPE tokenizer;
  measured results showed zero reduction because encoding cannot invent unseen emoji merge rules.
- Correctly distinguishes tokenizer-training data from evaluation data and explains byte-level
  BPE coverage: evaluation patterns may lack efficient learned merges, but every UTF-8 byte still
  has a base vocabulary ID, so unseen text remains representable without an unknown token.
- Correctly selected row-vector matrix order for batched prediction:
  `[B,T,C_in] @ [C_in,C_out] -> [B,T,C_out]`.
- Correctly explained that backward accumulates into persistent `.grad` storage, while the
  optimizer reads that storage and updates only the parameters registered with it.
- Completed the batched matrix-gradient path: shared `[C_in,C_out]` weight, batched prediction,
  one-token manual check, scalar mean squared loss, and backward into a matching gradient matrix.
- Completed one SGD update and correctly recomputed the forward prediction and mean squared loss
  using the updated in-place weight.

### Needs reinforcement

- Memory units: dtype widths are bits; storage is usually reported in bytes or MiB.
- Scalar gradients are new: reinforce change in output, change in input, their ratio, and the
  distinction between a finite-difference estimate and the exact derivative.
- Forward/backward/update separation: learner initially expected `backward()` to average losses
  and update batched values. Reinforce that the loss expression selects the reduction,
  `backward()` computes and accumulates parameter gradients, and `optimizer.step()` updates
  parameters.
- Optimizers are new: introduce them as update-rule objects that consume stored parameter
  gradients; begin with plain SGD before momentum or Adam.
- Throughput (GFLOP/s) is a new concept.
- Latency versus throughput: learner initially described them as inverses; reinforce that
  this holds only under fixed work and execution conditions, not in general.
- Tensor hierarchy is new: a vector is a rank-1 tensor; scalar, matrix, and higher-rank arrays
  are also tensors.
- Learner currently associates every tensor with an array of 2D matrices; reinforce that this
  describes rank 3 only, while tensor is the general term for rank 0 and above.
- Stride is new: reinforce that each stride is the number of storage elements skipped when
  advancing one position along its corresponding axis.
- Teaching preference: concepts should be introduced as why → practical use → technical
  mechanism, rather than beginning with formal rules.
- Coding preference: explain the entire algorithm and role of each function before incremental
  TODO implementation; do not reveal the system only one function at a time.
- Prerequisite rule: learner could not connect tokenizer tradeoffs to unlearned Transformer
  details. Explain current-module tradeoffs independently and defer model-cost derivations.
- Hidden state terminology is new: reinforce visible token IDs → internal `[B,T,C]`
  representations → visible logits, and distinguish state from permanent memory.
- Channel terminology is new: reinforce that `C` is the number of learned coordinates per
  token vector, shared as an axis across tokens—not a clean one-channel/one-concept mapping.
- Zero-stride broadcast addressing is new: reinforce `offset = Σ index×stride`, where batch
  and token contributions vanish and only the channel selects physical bias storage.
- Contiguous `[B,T,C]` stride calculation: learner proposed `[T,C,1]`; reinforce that advancing
  batch skips the full `[T,C]` plane, so the correct stride is `[T*C,C,1]`.
- Expanded-view accounting: learner expected `expanded_bias.numel()` to remain 4; reinforce
  that `numel()` counts 24 logical positions while shared physical storage remains 4 values.
- Transpose stride permutation: learner understands that strides swap with axes, but used `3`
  where channel stride is `4`; reinforce `[12,4,1] → [12,1,4]` for axes 1 and 2.
- Unicode versus UTF-8: learner correctly predicted five ASCII byte tokens for `hello` but
  predicted one for an emoji, conflating one code point with its four encoded bytes.
- BPE merge matching: learner initially interpreted pair matching as duplicate-value detection;
  reinforce exact equality with the selected ordered adjacent pair, which may contain two
  different token IDs such as `(a,n)`.
- BPE end-to-end mental model still needs reinforcement after the technical explanation; use
  the plain-English shortcut-dictionary analogy before returning to functions and IDs.

### Open questions

- Why GFLOP/s matters and why the middle-sized matrix can achieve higher throughput.
- What may remain nondeterministic even after a random seed is set.

### Newly introduced

- Microbatch gradient equivalence: differentiation distributes over a summed loss, while mean
  losses require scaling by the number of equal accumulation steps (or weighting by actual
  example/token counts when sizes differ).
- Batched token-vector tensors: `[B,T,C]` contains `B` sequences, `T` token positions per
  sequence, and one `C`-dimensional vector at each `[batch,token]` position.
- Multi-GPU training layouts: DDP replicates model/gradient/optimizer state and splits data;
  FSDP shards parameters, gradients, and optimizer state; tensor parallelism splits individual
  layer computations and weights.

## Evidence index

- Repository scaffold and curriculum created.
- Byte tokenizer, bigram model, tiny decoder-only Transformer, and LoRA layer implemented.
- Automated tokenizer, causality, model-shape, LoRA, and experiment-utility tests pass.
- Tiny bigram and Transformer smoke-training runs successfully reduced training loss.

## Session log

### 2026-07-16 — Course construction

- Created the 16-week learning plan and module checklist.
- Added controlled what-if experiments with hypotheses, metrics, and falsifiers.
- Added low-level models, LoRA, experiment utilities, and tests.
- Added the systems/cost optimization track and durable project memory.
- Added a 24-week portfolio schedule, public-repository quality gates, CI, and PR templates.
- Learner preference recorded: every lesson must include a browser-based revision artifact.
- Next: begin Module 0 with measurement fundamentals.

### 2026-07-16 — Module 0 started

- Added the tensor, stride, broadcasting, reproducibility, and measurement lesson.
- Added a warmup/synchronized matrix-multiplication benchmark and correctness test.
- Created the first experiment note without executing the main benchmark before prediction.
- Next: learner records the five predictions and runs the benchmark.

### 2026-07-16 — First Module 0 benchmark

- Recorded the learner's unedited baseline predictions before execution.
- Measured tensor storage, strides, seeded sampling, and CPU matmul throughput.
- Identified beginner topics to reinforce: units, scaling exponents, throughput, and seeds.
- Next: learner explains the corrected results before adding the dtype experiment.

### 2026-07-16 — Module 0 revision visual

- Added an interactive matrix-throughput explainer covering `2N³` work, elapsed time,
  GFLOP/s, output cells, dot products, and the 4× storage versus 8× work scaling rule.
- Made browser-based revision artifacts mandatory for future lessons in `AGENTS.md`.

### 2026-07-16 — Random seed lesson

- Added a beginner explanation distinguishing seed, generator state, deterministic execution,
  and reproducible experiments.
- Added an interactive two-run seed replay showing identical prefixes and state advancement.
- Next: learner experiments with reset/draw order and explains the same-seed condition.

### 2026-07-16 — Module 0B prepared

- Added a dtype lesson separating storage, numerical range, precision, and execution speed.
- Added a controlled CPU float32/float64 benchmark and experiment sheet.
- Added an interactive dtype memory and floating-point field revision visual.
- Next: learner predicts memory and latency ratios before execution.

### 2026-07-16 — Module 0B measured

- Float32: 8 MiB inputs, 1.266 ms median, 1696.25 estimated GFLOP/s.
- Float64: 16 MiB inputs, 5.446 ms median, 394.34 estimated GFLOP/s.
- Learner predicted the memory exactly and latency ratio closely (4× vs measured 4.30×).
- Next: explain why the latency ratio is workload/hardware specific.

### 2026-07-16 — Module 0C started

- Added the distinction between logical tensor view and physical storage.
- Added an interactive transpose/stride visual showing shared storage and changed indexing.
- Next: learner predicts whether transpose immediately allocates copied value storage.

### 2026-07-16 — Module 0D started

- Learner confirmed the transpose/view explanation.
- Added broadcasting rules and an interactive right-aligned shape comparator.
- Next: learner analyzes `[2,3,4] + [2,3]` dimension by dimension.

### 2026-07-16 — Broadcasting lesson restructured

- Reordered the lesson around an LLM bias-addition need, PyTorch usage, then shape/stride rules.
- Added a three-stage revision visual showing logical expansion and zero-stride reuse.
- Made the why → use → technical teaching order a durable course rule.

### 2026-07-16 — Hidden-state terminology

- Added a why/use/technical explanation of internal Transformer representations.
- Added an interactive path from token IDs through layer hidden states to vocabulary logits.
- Next: learner distinguishes an embedding/hidden state from a final output logit.

### 2026-07-16 — Hidden channels

- Added why/use/technical explanation of channels as coordinates of token representations.
- Added an interactive hidden-state table that highlights one channel across batches/tokens.
- Next: learner explains the difference between token count `T` and channel count `C`.

### 2026-07-16 — Zero-stride broadcasting

- Added address-level explanation of a `[4]` bias expanded to logical `[2,3,4]`.
- Added an interactive index-to-storage mapper for stride `[0,0,1]`.
- Next: learner calculates the storage position for logical index `[1,2,3]`.

### 2026-07-16 — Module 0 coding checkpoint prepared

- Learner completed the broadcasting compatibility discussion.
- Added an eight-part learner-owned coding exercise covering the full Module 0 foundation.
- Assertions provide immediate correctness feedback without revealing implementations.
- Next: learner completes TODO 1 and runs the file, then proceeds one failing assertion at a
  time with guidance.

### 2026-07-17 — Module 0 completed

- Learner implemented tensor creation, rank/element/storage accounting, strides, bias
  construction, broadcasting, zero-stride expansion, transpose views, and seeded randomness.
- Module 0 benchmark, dtype experiment, explanations, revision artifacts, and coding assertions
  provide the required correctness, measurement, performance, and reproducibility evidence.
- Next: begin Module 1 with why text must be converted to token IDs.

### 2026-07-17 — Module 0 revision hub

- Consolidated the individual Module 0 visuals into one selectable revision and self-check hub.
- Covers tensors, memory, matmul, seeds, strides, broadcasting, and hidden states/channels.
- Added why, core idea, and technical theory for every topic.
- Added a Git-tracked standalone copy under `revision/module-00-revision.html`.

### 2026-07-17 — Module 1 started

- Added why/how/technical theory for text, Unicode, UTF-8, byte tokens, special tokens, and cost.
- Added the first controlled multilingual byte-token experiment sheet.
- Added an interactive text → code point → UTF-8 byte → token ID → embedding visual.
- Next: learner predicts byte-token counts before running the lab.

### 2026-07-17 — Byte-token experiment measured

- Measured code points versus UTF-8 byte tokens on ASCII, accented Latin, Devanagari, and emoji.
- Confirmed lossless round trips including BOS/EOS framing.
- Observed 5/5, 10/12, 6/18, and 1/4 code-point/byte-token counts respectively.
- Next: learner explains the Devanagari result before pair-merge tokenization.

### 2026-07-17 — BPE lesson started

- Added why/how/technical theory for frequent-pair merges and vocabulary/sequence tradeoffs.
- Added a controlled BPE experiment sheet and interactive merge sequence.
- Next: learner predicts the result of the first `a+n` merge in `banana`.

### 2026-07-17 — BPE coding checkpoint prepared

- Added learner-owned implementations for adjacent-pair counting, non-overlapping merging,
  and deterministic pair selection.

### 2026-07-18 — Repeated BPE training implemented

- Connected the completed one-pass BPE functions with an outer merge-training loop.
- The implementation copies the input, assigns sequential learned IDs, records ordered merge
  rules, and stops when the merge budget is exhausted or fewer than two tokens remain.
- Verified the BPE exercise and all 14 repository tests.
- Next: trace the two `banana` passes and use the saved ordered rules to encode new text.

### 2026-07-18 — BPE training versus encoding introduced

- Added an interactive pass trace showing the two learned `banana` rules and their token IDs.
- Introduced the key distinction: training discovers and records rules, while encoding only
  replays those fixed rules in learned order and never invents a new merge.
- Next: learner predicts the encoded `banana` sequence after replaying both rules.

### 2026-07-19 — BPE encoding completed

- Learner correctly implemented ordered replay of learned merge rules without assistance in
  the final function body.
- Added assertions proving that encoding `banana` reproduces the trained token sequence and
  that unmatched input tokens remain unchanged.
- Next: explain decoding and recursively expand learned IDs back to their original bytes.

### 2026-07-19 — BPE decoding lesson started

- Introduced reverse vocabulary lookup and recursive expansion of learned IDs into base bytes.
- Added a step-through revision visual for `[257, 256, 97]` back to the UTF-8 bytes of `banana`.
- Next: learner predicts the complete base-byte list before implementing the decoder.

### 2026-07-19 — Iterative BPE decoding implemented

- Learner designed reverse-order rule expansion as a valid alternative to recursive decoding.
- Corrected list-versus-dictionary handling, `append`/`extend`, loop advancement, indentation,
  and naming while preserving the learner's algorithm.
- Added round-trip and non-mutation assertions: encoded IDs expand to the original bytes and
  UTF-8 decoding recovers `banana`.
- Next: compare iterative and recursive decoding and define behavior for unknown learned IDs.

### 2026-07-19 — BPE decoder validation completed

- Learner implemented validation that accepts base byte IDs and known learned IDs while rejecting
  unknown IDs before performing partial decoding.
- Added an assertion for the exact `unknown token ID: 999` failure and reverified round-trip safety.
- Next: integrate merges into multilingual byte tokenization and compare token counts.

### 2026-07-19 — Initial public checkpoint prepared

- Audited tracked project content for common credential patterns and files larger than 5 MB;
  none were detected.
- Confirmed generated environments, caches, checkpoints, downloaded data, and run outputs are
  excluded by `.gitignore`.
- Next: create the public GitHub repository under `vinitgupta4080` and publish this checkpoint.

### 2026-07-19 — Initial public checkpoint published

- Created the public repository `vinitgupta4080/llm-training-lab` through GitHub's web interface.
- Published all 49 tracked project files individually through the signed-in `vinitgupta4080`
  web editor after the local Git credential resolved to a forbidden account.
- Added durable workspace and project rules that prohibit publishing through
  `vinitgupta-alation`; stored credentials were not changed.
- Next: resume Module 1 with a multilingual BPE round-trip and token-count comparison.

### 2026-07-19 — Multilingual BPE experiment prepared

- Connected the full lossless path: Unicode text to UTF-8 bytes, ordered BPE encoding,
  reverse-order merge decoding, then UTF-8 text recovery.
- Added a step-through revision visual covering each representation in the round trip.
- Controlled comparison will train merges on an English-heavy corpus and evaluate English,
  accented Latin, Devanagari, and emoji without changing the learned merge table.
- Next: learner predicts which evaluation slice receives the largest token-count reduction.

### 2026-07-19 — Multilingual BPE experiment measured

- Implemented an inspectable `ByteBPETokenizer` with learned IDs beginning at 259, ordered
  encoding, reverse-order decoding, unknown-ID validation, and multilingual tests.
- Measured `banana banana` at 13→3 tokens (76.9% reduction); accented Latin, Devanagari, and
  repeated emoji each received 0% reduction under the English-only merge table.
- All four slices round-tripped exactly, and all 16 automated tests passed.
- Learner's emoji prediction was falsified: evaluation-time repetition does not create a rule
  absent from the tokenizer's training data.
- Next: learner explains the result, then profile runtime scaling with the merge count.

### 2026-07-19 — BPE cost-scaling benchmark prepared

- Added a controlled benchmark for merge budgets 0, 5, 10, 20, and 40 with fixed training and
  evaluation text, warmup, explicit repetitions, median latency, vocabulary, and token counts.
- Added an adjustable revision visual linking merge count to vocabulary and naive scan passes.
- Benchmark remains unexecuted until the learner records four directional predictions.
- Next: learner predicts the direction of all four metrics.

### 2026-07-19 — BPE scaling predictions recorded

- Learner predicted that training latency, encoding latency, and vocabulary size increase with
  merge budget, while token count decreases for familiar evaluation text.
- Next: execute the unchanged benchmark and inspect monotonicity, plateaus, and measurement noise.

### 2026-07-19 — BPE cost scaling measured

- Learner correctly predicted the broad directions: training time, encoding time, and vocabulary
  increased, while familiar-text token count decreased.
- Measured budgets 0/5/10/20/40; the final request learned only 28 rules because no further pair
  remained, so vocabulary reached 287 rather than 299.
- Familiar evaluation text compressed 1,499→401→203→104 tokens, then plateaued at 104; later
  rules did not match that evaluation slice.
- Added an interactive measured-results explorer. All 16 automated tests still pass.
- Next: learner explains both plateaus, then finish Module 1 profiling/checkpoint evidence.

### 2026-07-19 — Module 1 completed

- Learner explained UTF-8 byte coverage, BPE pair counting and ordered merges, the difference
  between training and encoding, reverse-order decoding, and training-versus-evaluation effects.
- Implemented BPE training, encoding, decoding, reserved-ID safety, unknown-ID failure handling,
  and a reusable multilingual byte-BPE tokenizer.
- Controlled multilingual comparison and merge-scaling profile provide correctness, quality,
  timing, vocabulary, and token-count evidence. All 16 tests pass.
- Consolidated the final theory, measured cost results, and self-checks into the Module 1
  interactive revision checkpoint and Git-tracked standalone page.
- Module 2 starts next with scalar slopes, gradients, and why optimization needs them.

### 2026-07-19 — Module 2 scalar-gradient lesson started

- Introduced local sensitivity using `y=x²` at `x=3` and corrected the initial finite-difference
  prediction: increasing x by 0.001 changes y by +0.006001, giving slope 6.001 near exact 6.
- Added an adjustable finite-difference revision tool comparing measured slope with `dy/dx=2x`.
- Next: learner predicts the gradient sign and loss-reducing direction at `x=-3` before autograd.

### 2026-07-19 — Autograd forward/backward model introduced

- Explained the full first experiment before coding: create a leaf tensor, record the forward
  graph, traverse it with backward, and read the gradient stored on the leaf.
- Added an interactive forward/backward graph and the Module 2A lesson scaffold.
- Next: learner predicts all four observable values before running PyTorch.

### 2026-07-19 — Scalar autograd exercise prepared

- Learner confirmed the interpretation of `x` as a simplified trainable parameter and `y=x²`
  as a toy loss function.
- Added a three-TODO exercise with assertions for leaf value, forward loss, backward return,
  stored gradient, and agreement with the manual derivative.
- Next: learner implements tensor creation, forward expression, and backward call.

### 2026-07-19 — First autograd exercise completed

- Learner correctly created a scalar leaf tensor with gradient tracking, computed `x²`, and
  called `backward`; the call was assigned to expose its `None` return contract.
- Assertions verify `x=-3`, `y=9`, `backward_result is None`, and `x.grad=-6`, matching `2x`.
- Next: explain gradient accumulation and the need to clear gradients between training steps.

### 2026-07-19 — Gradient accumulation introduced

- Explained `.grad` as an accumulation bucket: two fresh forward/backward graphs at `x=-3`
  contribute `-6 + -6 = -12` unless the stored gradient is cleared.
- Connected default accumulation to both accidental cross-step leakage and intentional
  microbatch accumulation used when a full batch does not fit in memory.
- Added an interactive accumulation/clear tool and a one-TODO exercise.
- Next: learner clears the gradient before the third backward pass.

### 2026-07-19 — Microbatch equivalence explained

- Connected `.grad` accumulation to full-batch training: gradients of individual loss terms add,
  so summed microbatch gradients match the gradient of the summed full-batch loss.
- Distinguished sum from mean reduction: with `K` equal microbatches, each microbatch mean loss
  must be divided by `K` before `backward()` to match a full-batch mean.
- Added an interactive full-batch-versus-microbatch calculator, including unequal-token and
  stochastic-operation caveats.
- Next: learner predicts the scaling for four equal microbatches, then completes gradient clearing.

### 2026-07-20 — Multi-token vector tensor

- Clarified that batching averages corresponding parameter-gradient positions across examples;
  it does not average different vector dimensions into one scalar.
- Added an interactive `[B,T,C]=[2,3,4]` viewer that exposes each batch's three token vectors.
- Next: learner constructs the displayed tensor in PyTorch and verifies its shape.

### 2026-07-21 — Batched mean-loss exercise prepared

- Extended the one-point model to three `(x,target)` pairs sharing one trainable scalar `w`.
- Traced vectorized predictions, per-point squared losses, scalar mean reduction, the mean
  gradient with respect to `w`, and a single simultaneous parameter update.
- Added a forward/backward/update visual and a four-TODO PyTorch exercise with numeric assertions.
- Next: learner implements the vectorized predictions using the shared weight.

### 2026-07-21 — Batched matrix-gradient exercise prepared

- Generalized the shared scalar weight to a `[C_in,C_out]` weight matrix applied to every token
  vector in a `[B,T,C_in]` input tensor.
- Added a manually checkable `B=2,T=2,C_in=2,C_out=2` exercise covering prediction shape,
  one-token matrix multiplication, scalar MSE, backward, and weight-gradient shape and values.
- Added an interactive trace from one token through the full batch to the shared gradient matrix.
- Next: learner writes the batched matrix multiplication for TODO 1.

### 2026-07-26 — Optimizer introduced

- Paused the batched exercise because `optimizer.step()` had appeared before a formal optimizer
  lesson.
- Introduced the optimizer as the separate mechanism that reads `.grad` and changes parameters;
  plain SGD implements `parameter -= learning_rate * gradient`.
- Added an adjustable one-parameter SGD update visual separating parameter, gradient, learning
  rate, and updated value.
- Next: learner predicts one SGD update, then resumes the batched matrix exercise.

### 2026-07-26 — Training-step memory lifecycle

- Explained the reason for gradient clearing: backward adds into persistent leaf `.grad` storage,
  while `optimizer.step()` reads but deliberately does not erase that storage.
- Distinguished persistent parameter/gradient/optimizer state from temporary forward activations
  and the computation graph, which is normally freed after backward.
- Added a complete three-step SGD trace and an interactive clear/forward/backward/step memory view.
- Next: learner explains why clearing is needed between independent optimizer steps.

### 2026-07-26 — Multi-GPU state placement introduced

- Mapped parameters, gradients, optimizer state, activations, and computation graphs from one GPU
  onto DDP, fully sharded data parallel, and tensor-parallel layouts.
- Explained DDP gradient all-reduce and why identical starting parameters plus identical averaged
  gradients let independent local optimizers remain synchronized.
- Added a four-GPU comparison showing what is replicated versus sharded and where communication
  occurs.
- Next: learner explains why DDP does not reduce per-GPU model-state memory, then returns to the
  single-GPU optimizer exercise.

### 2026-07-26 — First complete multi-step training loop

- Learner completed the five operations repeated during training: clear gradients, calculate
  predictions, calculate scalar mean loss, backpropagate, and update the weight.
- Verified the 20-step loop: loss decreased from `70.500000` to `0.105712`.
- Corrected an overly strict scaffold assertion that expected the ideal weight after only 20 SGD
  steps; the exercise now checks the intended measurable outcome, a substantial loss reduction.
- Next: inspect the loss trajectory and experiment with learning rate and step count.

### 2026-07-26 — Controlled learning-rate sweep

- Added `labs/02_autograd/learning_rate_sweep.py` and tested five SGD learning rates under an
  identical initialization, dataset, optimizer, dtype, and 20-step budget.
- Measured slow learning at `0.0001` and `0.001`, effective learning at `0.01` and `0.03`, and
  divergence at `0.1`; among tested rates, `0.03` had the lowest validation loss after 20 steps.
- Recorded the experiment design, falsifier, competing explanation, measurements, and limitations
  in `experiments/2026-07-26_learning-rate-sweep.md`.
- Added the `learning-rate-experiment.html` revision visual covering update scale, loss curves,
  and slow/effective/unstable regimes.
- Next: learner interprets why “best” is conditional on the tested rates, validation metric, and
  fixed compute budget; then vary only the number of training steps.

### 2026-07-26 — Production learning-rate selection mental model

- Learner correctly explained that a small learning rate can move in the right direction yet fail
  to reach the minimum under a fixed compute budget.
- Added `company-learning-rate-workflow.html`, covering prior recipes, cheap pilot sweeps,
  validation-based selection under equal budgets, and full-run stability monitoring.
- Next: vary only training-step count to demonstrate that the best rate under 20 steps can change
  when the compute budget changes.

### 2026-07-26 — Step-budget and diminishing-returns experiment

- Held learning rate at `0.001` while varying only the budget across 20, 200, and 2,000 steps.
- Measured monotonically decreasing training loss: `9.77733`, `0.105527`, and `0.074096`.
- Corrected a cold-start timing artifact by adding warm-up and reporting the median of five runs;
  measured time scaled approximately 1×, 10×, and 100× with optimizer-step count.
- Added `labs/02_autograd/step_budget_experiment.py` and
  `experiments/2026-07-26_step-budget.md`.
- Updated `revision/module-02-revision.html` with the measured learning-rate sweep, step-budget
  comparison, and diminishing-returns decision rule for GitHub Pages.
- Next: learner interprets diminishing returns, then compares plain SGD with momentum.

### 2026-07-26 — Momentum mental model

- Introduced momentum as an exponentially retained history of first-order gradients, not a second
  derivative; learner recognized that momentum can still oscillate.
- Explained acceleration when gradients agree, partial cancellation when gradients alternate,
  and overshooting when retained velocity, momentum, or learning rate is too large.
- Connected momentum to learning-rate schedules: large productive movement early, smaller and more
  precise updates later, with warmup as the complementary early-training mechanism.
- Added `momentum-step-calculator.html` for stepping through agreeing and alternating gradient
  sequences at momentum `0.9`.
- Implemented `labs/02_autograd/momentum_equivalence_exercise.py`; the learner-created manual
  velocity and weight updates exactly match PyTorch SGD momentum across two vector gradients.
- Verified final velocity `[15.0, 11.2]`, final weight `[0.75, 1.808]`, and all 16 tests passing.
- Clarified that coordinates such as `x` and `y` in a toy loss surface represent two trainable
  parameters, while the earlier MSE reduces errors over batches/tokens/outputs to a scalar function
  of every weight; added `two-parameter-loss-surface.html` to make the mapping explicit.
- Next: learner connects `L(x,y)=x²+50y²` to two squared-error contributions, then compares plain
  SGD and momentum on its steep and shallow parameter directions.

### 2026-07-27 — Momentum hypothesis falsified at aggressive learning rate

- Learner derived gradients `[10,500]` at `[x,y]=[5,5]` and traced steep-direction SGD oscillation
  `5 → -4 → 3.2` at learning rate `0.018`.
- Added `labs/02_autograd/sgd_momentum_zigzag.py` and
  `experiments/2026-07-27_sgd-vs-momentum.md`.
- Under the identical 50-step, LR `0.018` budget, plain SGD reached loss `0.687844`, while momentum
  reached `12.047579`; the original “momentum will be lower” hypothesis was falsified because
  retained velocity amplified overshooting.
- Verified optimizer state: plain SGD stored zero state elements and momentum stored two, matching
  the two-parameter vector.
- Learner correctly explained that LR `0.018` plus retained velocity repeatedly overshot the
  minimum, then predicted momentum would win at the safer shared LR `0.005`.
- Replication confirmed the prediction: at LR `0.005`, plain SGD ended at loss `9.336602` and
  momentum ended at `1.775322` after the same 50 steps.
- Introduced Adam's purpose: the first moment smooths direction, while the second moment tracks
  squared-gradient scale so each parameter receives an adaptive effective step size.
- Added `adam-adaptive-scaling.html`, mapping the steep/shallow gradient `[10,500]` to raw SGD
  scaling versus approximate Adam normalization `[1,1]`.
- Learner manually calculated the first Adam state for `g=10`: raw `m=1`, raw `v=0.1`,
  bias-corrected `m̂=10`, and `v̂=100`, while choosing to retain the compact mental model rather
  than memorize every formula.
- Added and verified `labs/02_autograd/adam_state_inspection.py`; PyTorch stored `exp_avg=[1]`,
  `exp_avg_sq=[0.1]`, step `1`, and updated weight `5 → 4.999`, matching the manual derivation.
- Learner predicted momentum would win a same-LR comparison while correctly warning that optimizer
  ranking depends on many controlled variables.
- Added `labs/02_autograd/optimizer_comparison.py` and
  `experiments/2026-07-27_optimizer-comparison.md`: at LR `0.005` after 50 updates, losses were
  plain SGD `9.150803`, momentum `6.317387`, and Adam `1151.427368`.
- Verified optimizer-state elements for the two-parameter model: plain SGD `0`, momentum `2`, Adam
  `4`; Adam's poor result reflects a mismatched shared numeric LR, not universal inferiority.
- Added `labs/02_autograd/tuned_optimizer_comparison.py` and
  `experiments/2026-07-27_tuned-optimizer-comparison.md`; expanded ranges when initial winners
  appeared at boundaries.
- Under equal 50-step final-run budgets, best tested results were plain SGD LR `0.019`, loss
  `0.552536`; momentum LR `0.017`, loss `0.303671`; Adam LR `0.5`, loss `0.029599`.
- Introduced AdamW as Adam's two adaptive state tensors plus a separate direct shrinkage of weights;
  clarified that Adam's first moment already fills the momentum role.
- Added `weight-decay-sensitivity.html` to connect weight magnitude with output sensitivity through
  `Δoutput = weight × Δinput`, while emphasizing that large weights are not automatically wrong.
- Learner correctly calculated the isolated AdamW decay update:
  `10 × (1 - 0.1 × 0.01) = 9.99`, and distinguished the Adam gradient update from the
  separate decoupled weight-decay update.
- Reinforced that a weight, its gradient, Adam's first/second-moment state, and its update all
  share the weight's shape, while their element values generally differ.
- Added `adamw-zero-gradient.html`, an interactive comparison of a zero gradient tensor versus
  `grad=None`; the former permits the AdamW decay step while PyTorch skips the parameter for the
  latter.
- Learner derived the nonzero-gradient AdamW examples: with `w=10`, LR `0.1`, decay `0.01`,
  the task gradient toward target `8` and decay both reduce the weight to `9.89`; toward target
  `12`, the task update and decay oppose each other and produce `10.09`.
- Clarified that longer training does not cancel weight decay: decay acts every step and an
  excessive value can bias the equilibrium toward weights that are too small, causing underfit.
- Expanded the Module 2 revision hub with detailed SGD, momentum, Adam, AdamW, optimizer-state
  memory, edge cases, measured same-LR and tuned comparisons, search cost, and fair-selection
  criteria.
- Learner reconstructed the repeated AdamW loop in the correct conceptual order
  (clear → forward → backward → update), needing only the exact `zero_grad` syntax.
- Added `labs/02_autograd/weight_decay_sweep.py` and
  `experiments/2026-07-27_weight-decay-sweep.md`.
- The controlled sweep held data, initialization, LR, and 1,000-step budget fixed. Validation
  loss improved from `0.775838` at decay `0` to `0.729880` at decay `0.1`, while training loss
  slightly worsened; expanding the boundary to decay `1.0` raised validation loss to `2.274466`
  and training loss to `1.236641`, demonstrating underfit.
- Search cost was five candidates × 1,000 updates = 5,000 optimizer updates, or 5× one final
  run. The single-seed toy result does not select a universal LLM decay value.
- Learner correctly explained the decay tradeoff: aggressive decay also shrinks useful weights,
  so moderate decay may remove more noise than signal while excessive decay causes underfit.
- Learner derived global norm clipping for gradient `[30,40]`: original norm `50`, scale factor
  `1/50`, clipped gradient `[0.6,0.8]`, and preserved direction.
- Added `labs/02_autograd/gradient_clipping_spike.py` and
  `experiments/2026-07-27_gradient-clipping-spike.md`.
- Learner predicted the unclipped run would make the larger parameter jump. At the controlled
  extreme batch both pre-clip norms were `13107.2`; the unclipped update was `1310.72`, while
  clipping to norm `10` limited the update to `1.0`. The next normal loss was approximately
  `1.716M` unclipped versus `0.1188` clipped.
- Learner explained clipping as a guardrail that preserves direction and limits one poor batch
  from moving the model far from a useful minimum, while leaving the bad-data or instability
  cause unresolved.
- Introduced layer shape composition: `[...C_in] @ [C_in,C_out] -> [...C_out]`; adjacent
  dimensions must match, but internal matrices are square only when the designer preserves width.
- Learner recognized the architecture/cost tradeoff: more parameters increase capacity and
  training memory, but quality also requires sufficient data, compute, optimization, and design.
- Added `labs/02_autograd/initialization_scale_experiment.py` and
  `experiments/2026-07-27_initialization-scale.md`.
- Learner predicted shrink/stable/grow for `0.1×/1×/10×` fan-in initialization. At layer 20,
  activation stds were approximately `1.11e-20`, `1.11`, and `1.11e20`; first-weight gradient
  norms were `0`, `2.23`, and `inf`, respectively.
- Learner connected increased fan-in with overflow risk and learned that a `[C_in,C_out]`
  matrix has fan-in `C_in` and fan-out `C_out`; fan-in controls how many forward contributions
  enter each output, while fan-out controls how many backward contributions return to each input.
- Added `fan-in-fan-out.html`, an interactive forward/backward connection view for a `[3,2]`
  weight matrix.
- Learner calculated Xavier variance `0.4` and standard deviation approximately `0.632` for a
  `[3,2]` rectangular matrix and explained that initialization prevents the signal from fading
  or exploding at the start.
- Distinguished Xavier initialization (starting weight scale), LayerNorm (hidden-activation
  scale on every forward pass), residual connections (direct signal/gradient path), gradient
  clipping (abnormal-gradient guardrail), and learning-rate control (parameter-update size).
- Learner correctly reasoned that a zero weight passes no signal through that connection, while
  a zero weight together with a zero gradient remains unchanged; one inactive connection does
  not imply that an entire output dimension is inactive.
- Expanded `module-02-revision-hub.html` and `revision/module-02-revision.html` with a single
  signal-stability reference table, forward/backward flow, equations, edge cases, and the reason
  each mechanism exists.
- Expanded the signal-stability reference with complete beginner-level math, symbol definitions,
  and worked examples for Xavier variance/standard deviation, LayerNorm mean/variance and learned
  affine transform, residual forward/backward paths, global-norm gradient clipping, and
  learning-rate-scaled parameter movement.
- A premature softmax preview exposed concepts from the future language-modeling module; learner
  correctly flagged the prerequisite jump. Softmax, vocabulary scores, and sampling are deferred
  until the course constructs that prediction problem.
- Began numerical stability using only the known scalar training path
  `prediction=input*weight`, squared-error loss, backward, and an optimizer update.
- Learner explained why the special case `input=1, target=0` reduces the loss to `weight²` and
  why `backward()` calculates `d(loss)/d(weight)` into `weight.grad`.
- Learner correctly predicted that `weight=1e20` produces an infinite float32 loss but a finite
  `2e20` gradient, and learned that `optimizer.step()` does not inspect the loss or automatically
  reject non-finite gradients.
- Added `labs/02_autograd/numerical_range_experiment.py`,
  `experiments/2026-07-28_numerical-range.md`, and `numerical-range-training.html` to compare
  underflow, ordinary range, and overflow at each forward/backward checkpoint.
- Learner independently implemented the reusable finite loss/gradient guard, including
  `grad is None` handling and elementwise `.all()` reduction, and correctly identified weights,
  biases, and normalization affine tensors as parameters.
- Connected data normalization, temporary/dynamic loss scaling, gradient unscaling, clipping,
  learning rate, dtype range, and dtype precision. Learner correctly challenged whether scaling
  alone permits arbitrarily low precision; concluded that scaling helps range but cannot recover
  distinctions or updates lost to rounding.
- Consolidated the measured cases, equations, safety guard, correct operation order, and
  range-versus-precision tradeoffs under a new Numerical range topic in the Module 2 revision hub.
- Next: run a controlled float16/bfloat16/float32 range-and-rounding experiment, then decide which
  tensors need higher precision before the Module 2 checkpoint.

### 2026-07-26 — Public revision site caught up

- Corrected the public revision gap: Module 2 concepts had individual conversation visuals and
  progress entries but no Git-tracked consolidated revision page.
- Added an interactive Module 2 revision hub covering gradients, forward/backward, batched matrix
  loss, microbatch accumulation, optimizer updates, memory lifetime, and a labeled GPU preview.
- Added a styled GitHub Pages homepage linking the complete Module 0/1 hubs and the in-progress
  Module 2 hub.
- Next: resume the single-GPU optimizer exercise; FSDP and tensor-parallel implementation remain
  deliberately deferred to their later systems modules.

### 2026-07-26 — Multi-step training-loop exercise prepared

- Converted the completed one-step matrix-gradient calculation into a 20-step learner-owned
  training loop using the same transparent `[B,T,C]` data and shared weight matrix.
- The loop separates persistent state (weight and optimizer) from per-step state (graph, loss,
  activations, and fresh gradient) and records a loss trajectory.
- Next: learner implements `zero_grad`, then forward, loss, backward, and optimizer step.
- Added assertions using byte IDs for `banana`.
- Next: implement pair counting and proceed one failing assertion at a time.

### 2026-07-17 — Complete BPE mental model

- Added training, deterministic pair selection, vocabulary construction, encoding, and decoding
  as one end-to-end explanation before continuing code.
- Added an interactive six-stage BPE training/encoding cycle.
- Made complete-algorithm-before-TODOs a durable teaching rule.

### 2026-07-18 — Plain-English BPE explanation

- Reframed BPE as learning a shortcut dictionary for frequently neighboring text pieces.
- Added a no-code interactive story: split, notice repetition, create shortcut, reuse, recover.
- Next: learner explains training versus use in plain language before resuming `merge_pair`.

### 2026-07-18 — Module 1 revision hub

- Consolidated Module 1 theory and discussions into an interactive revision hub.
- Covers why tokens, Unicode/UTF-8, byte tokens, BPE purpose, full training algorithm,
  encoding/decoding, deterministic selection, benefits, system costs, risks, and quizzes.
- Added a Git-tracked standalone copy under `revision/module-01-revision.html`.

### 2026-07-18 — BPE deep dive and tradeoff simulator

- Added detailed BPE notes covering purpose, two-bill cost model, full training/encoding/decoding
  algorithm, deterministic ties, IDs, complexity, edge cases, evaluation, and tests.
- Added an interactive vocabulary `V×C` versus sequence `T²` cost simulator.
- Next: learner explains the two bills in plain language, then resumes merge implementation.

### 2026-07-18 — BPE tradeoff simplified

- Reframed the required Module 1 tradeoff as dictionary shortcuts versus encoded piece count.
- Deferred embeddings, logits, attention `T²`, and exact compute implications to Module 4.
- Added a prerequisite-free interactive comparison using `banana` and the `an` shortcut.

### 2026-07-16 — Tensor rank clarification

- Paused dtype prediction to explain tensor as the general object containing scalars,
  vectors, matrices, and higher-rank arrays.
- Added an interactive rank/shape/element-count visual with LLM-specific examples.
- Next: learner identifies rank and element count for `[2, 3, 4]`, then returns to dtype.
