# Course progress

This is the authoritative resume point. Update it after every meaningful learning session.

## Current state

- Current module: Module 2 — gradients and optimization
- Status: in progress
- Next action: introduce a scalar slope as local sensitivity, predict finite-difference results, then compare manual derivatives with autograd
- Last verified: 2026-07-19; Module 1 complete and 16 automated tests passing

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

### Needs reinforcement

- Memory units: dtype widths are bits; storage is usually reported in bytes or MiB.
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
