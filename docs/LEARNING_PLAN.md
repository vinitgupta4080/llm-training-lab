# Learning plan: from zero to LLM training engineer

Suggested pace: 16 weeks at 8–10 hours/week. Faster is fine, but do not skip the
completion checks. “Understand” means you can derive it, implement it, debug it,
and explain what breaks when it changes.

## Phase 1 — Foundations (Weeks 1–3)

### Module 0: environment, tensors, and measurement

Learn shapes, dtypes, devices, broadcasting, memory, random seeds, train/validation
splits, and experiment hygiene. Build tensor exercises and benchmark matrix multiply.

Completion check: explain every dimension in `[batch, time, channels]`, reproduce a
run from a seed, and identify data leakage.

### Module 1: text and tokenization

Learn Unicode, bytes, vocabulary, special tokens, BPE/unigram intuition, encode/decode,
unknown tokens, and how tokenization changes cost and context length. Implement the
included byte tokenizer, then extend it with merges.

Completion check: losslessly round-trip multilingual text and compare token counts.

### Module 2: gradients and optimization

Derive scalar backprop, computational graphs, chain rule, softmax/cross-entropy,
gradient descent, momentum, AdamW, clipping, initialization, and numerical stability.
Manually differentiate a tiny network and compare against autograd.

Completion check: explain `zero_grad`, leaf tensors, vanishing/exploding gradients,
and why weight decay differs from an L2 term under Adam.

## Phase 2 — Language modeling internals (Weeks 4–7)

### Module 3: autoregressive language modeling

Learn next-token likelihood, teacher forcing, context windows, batching, perplexity,
overfitting, sampling temperature, top-k, and train/eval modes. Train the bigram model.

Completion check: derive cross-entropy from maximum likelihood and diagnose a gap
between training and validation loss.

### Module 4: decoder-only Transformer from scratch

Implement embeddings, positional information, causal self-attention, scaled dot
products, multi-head attention, residual connections, LayerNorm/RMSNorm, MLPs, weight
tying, and generation. Trace all tensor shapes and parameter counts.

Completion check: prove future tokens cannot affect an earlier logit, estimate attention
memory, and overfit one batch as a pipeline test.

### Module 5: pretraining systems

Learn data cleaning/deduplication, packing, shuffling, curriculum effects, initialization,
mixed precision, gradient accumulation, learning-rate warmup/decay, checkpointing,
resuming, distributed data parallelism, tensor/pipeline parallelism, ZeRO/FSDP, and
scaling laws. Turn the tiny model into a reproducible pretraining run.

Completion check: calculate tokens/run, effective batch size, model FLOPs, checkpoint
size, and expected memory; resume without changing the learning trajectory.

## Phase 3 — Tuning and alignment (Weeks 8–11)

### Module 6: supervised fine-tuning (SFT)

Learn chat templates, instruction data quality, response-only loss masking, sequence
packing, catastrophic forgetting, domain adaptation, multi-task mixing, and evaluation.

Completion check: inspect the exact token sequence and loss mask for three conversations.

### Module 7: parameter-efficient tuning

Learn LoRA's low-rank update, rank/alpha/dropout, adapter targets, QLoRA concepts,
quantization error, merging, and adapter serving. Implement `LoRALinear`, freeze the
base model, and verify which parameters receive gradients.

Completion check: calculate trainable parameter savings and merge an adapter while
preserving outputs within tolerance.

### Module 8: preference alignment

Learn preference datasets, reward models, Bradley–Terry loss, PPO intuition, KL
regularization, reference policies, DPO-style objectives, rejection sampling, and
common reward-hacking/failure modes.

Completion check: compute a preference loss by hand and explain the role of beta and
the reference model. Treat safety and human-data governance as system requirements.

## Phase 4 — Evaluation and deployment (Weeks 12–14)

### Module 9: evaluation

Build held-out loss, task metrics, exact-match tests, calibration checks, slice-based
analysis, contamination checks, human rubrics, safety testing, and regression gates.
Learn why one benchmark number is never sufficient.

### Module 10: inference

Implement greedy/top-k/temperature sampling and a KV cache. Learn batching, latency vs
throughput, memory bandwidth, quantization, speculative decoding, stop conditions,
prompt injection boundaries, and observability.

### Module 11: scaling and production

Design a multi-GPU run: data/token budget, topology, sharding, fault recovery, logging,
cost, licensing, data lineage, privacy, and release criteria. No costly run is required;
the deliverable is a defensible plan with measured small-scale extrapolations.

## Phase 5 — Capstone (Weeks 15–16)

Choose a small public-domain or self-authored corpus. Pretrain a 1–10M parameter model,
SFT it for one narrow task, optionally add LoRA or preference tuning, and compare every
stage against a fixed evaluation set.

Deliverables:

1. reproducible config, seed, environment, and data manifest
2. learning curves and ablation table
3. base-vs-tuned evaluation with error analysis
4. model card covering intended use, limitations, safety, data, and license
5. five-minute explanation of attention, training loss, and the chosen tuning method

## Weekly routine

1. Read the module goals and derive the key equation on paper.
2. Run the baseline and save its metrics.
3. Implement the missing mechanism without copying a framework implementation.
4. Write at least two tests, including one failure case.
5. Run one controlled ablation and record it in the experiment journal.
6. Explain the result in plain language and answer the completion check.

