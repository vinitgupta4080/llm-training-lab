# Low-level systems, time, memory, and cost optimization

Optimization has four layers: algorithm, numerical representation, kernel execution,
and distributed system. A faster step is not necessarily a cheaper useful model; always
normalize by quality or target loss.

## 1. Cost model to calculate before training

Track these quantities explicitly:

```text
tokens_per_step = microbatch * sequence_length * accumulation * data_parallel_world_size
training_tokens = tokens_per_step * optimizer_steps
parameter_bytes = parameter_count * bytes_per_parameter
time_hours = training_tokens / measured_tokens_per_second / 3600
compute_cost = time_hours * number_of_devices * price_per_device_hour
cost_to_quality = compute_cost required to reach a fixed validation target
```

A rough dense-Transformer training estimate is approximately `6 * parameters * tokens`
FLOPs. Treat it as an order-of-magnitude estimate: attention, embeddings, recomputation,
optimizer, sparsity, padding, and hardware utilization change real cost.

Required distinction:

- theoretical FLOPs: mathematical operations implied by the model
- achieved FLOPs/s: useful executed throughput
- hardware peak FLOPs/s: marketing/theoretical device ceiling
- utilization proxy: achieved divided by peak, documented with assumptions

## 2. Memory accounting

Training memory includes:

- parameters
- gradients
- optimizer state (Adam commonly has two moment tensors)
- master weights when mixed-precision training uses them
- saved activations for backward
- temporary kernel workspaces, allocator fragmentation, and communication buffers

Do not use the simplistic “parameters × dtype” estimate for training.

### What if activation memory dominates?

Vary sequence length and batch size separately. Measure peak allocated and reserved memory.
Then enable activation checkpointing. Record memory saved, extra forward computation,
tokens/second, and cost to the same validation loss.

### What if optimizer state dominates?

Compare AdamW, an 8-bit optimizer, and state sharding. Report optimizer bytes per parameter,
convergence, communication, and checkpoint size. A memory-saving optimizer is not cheaper if
it requires substantially more tokens to reach the target quality.

## 3. Profiling correctly

For every benchmark:

1. state hardware, software versions, power mode, dtype, shapes, and batch
2. run warmup iterations
3. synchronize asynchronous accelerators around timing
4. measure multiple iterations and report median plus variability
5. separate input pipeline, forward, backward, optimizer, communication, and checkpoint I/O
6. profile representative steady-state work, not only the first step

Use a profiler trace only after an end-to-end wall-time baseline exists. First identify
whether the workload is compute-bound, memory-bandwidth-bound, communication-bound, or
input-bound.

## 4. Kernel-level experiments

### What if matrix dimensions are hardware-unfriendly?

- Sweep hidden widths near a target size, including aligned and unaligned dimensions.
- Measure kernel time and achieved throughput with identical token counts.
- Explain tensor-core tile alignment and why parameter count alone predicts poorly.

### What if attention is fused?

- Compare explicit attention with scaled-dot-product/flash-style attention.
- Validate outputs and gradients within dtype-appropriate tolerance.
- Sweep sequence length; record peak memory and prefill throughput.
- Explain why avoiding materialization of the full score matrix changes memory traffic.

### What if operations are fused?

- Compare separate bias/activation/dropout/residual operations with fused alternatives.
- Count kernel launches and memory reads/writes.
- Keep numerical semantics and dropout seeds in mind when testing equivalence.

### What if `torch.compile` or graph capture is enabled?

- Separate compilation time from steady-state time.
- Test static and dynamic shapes and record graph breaks.
- Calculate the number of repeated steps required to amortize compilation.

## 5. Precision and numerical stability

Compare float32, TF32 where applicable, float16, bfloat16, and mixed precision.

Measure:

- memory and throughput
- overflow/underflow, nonfinite gradients, and loss-scaler behavior
- convergence in tokens, not only speed per step
- accumulation and reduction precision

### What if float16 is faster but training diverges?

Inspect the range of activations and gradients. Add dynamic loss scaling, stable softmax,
gradient clipping, or bfloat16 one change at a time. Never label precision as safe from a
short smoke test alone.

## 6. Input-pipeline optimization

Profile tokenization, storage reads, decoding, shuffling, packing, host-to-device transfer,
and accelerator idle gaps.

Experiments:

- online versus offline tokenization
- JSON text versus memory-mapped binary token arrays
- one versus multiple loader workers
- pageable versus pinned host memory
- synchronous versus overlapped transfer
- padded batches versus packed sequences
- random samples versus length buckets

Report useful tokens/second, not examples/second. A highly optimized loader that feeds mostly
padding is not efficient.

## 7. Batch size and gradient accumulation

Increase microbatch size until memory or throughput stops improving, then use accumulation
to reach the desired effective batch. Measure:

- tokens/second and peak memory
- optimizer steps per second
- gradient-noise behavior
- steps and tokens to fixed validation loss

Larger batches improve device utilization only up to saturation and may reduce statistical
efficiency. Optimize total time-to-quality, not maximum instantaneous throughput.

## 8. Distributed training

Study in this order:

1. data parallelism: replicated parameters, synchronized gradients
2. optimizer/gradient/parameter sharding (ZeRO/FSDP family)
3. tensor parallelism: split operations inside a layer
4. pipeline parallelism: split layers and manage pipeline bubbles
5. sequence/context parallelism for long contexts

For each design, calculate per-device parameter/gradient/optimizer/activation memory and
communication bytes. Measure scaling efficiency:

```text
scaling_efficiency = throughput_N / (N * throughput_1)
```

### What if adding GPUs makes training more expensive per token?

Break down compute, collective communication, imbalance, pipeline bubbles, data stalls, and
checkpoint time. Find the smallest device count that satisfies memory and deadline constraints.

## 9. Checkpoint and fault-tolerance economics

Checkpoint interval trades I/O overhead against expected lost work. Measure checkpoint time,
size, resume time, and storage cost. Include model, optimizer, scheduler, scaler, RNG, sampler,
and progress state when exact resume matters.

Explore:

- full versus sharded checkpoints
- synchronous versus asynchronous writes
- frequent lightweight weights-only snapshots versus full recovery checkpoints
- retention policies and object-storage lifecycle cost

## 10. Inference cost optimization

Separate prefill (parallel prompt processing) from decode (one new token per sequence step).
Track time-to-first-token, inter-token latency, output tokens/second, request throughput,
tail latency, KV-cache bytes, and cost per million output tokens.

Experiments:

- static versus continuous batching
- KV caching versus full recomputation
- grouped/multi-query versus multi-head KV caches
- weight-only and weight-plus-activation quantization
- prompt-prefix caching
- speculative decoding and acceptance rate
- tensor parallelism versus single-device replicas

Batching can improve cost while worsening individual latency. Always state the service-level
objective and request distribution.

## 11. Optimization decision order

Use this sequence to avoid polishing the wrong bottleneck:

1. establish correctness and a quality baseline
2. define target quality, latency, and budget
3. measure end-to-end time and peak memory
4. identify the dominant bottleneck
5. remove wasted tokens: padding, duplication, bad data, excessive sequence length
6. choose precision and the largest efficient microbatch
7. use optimized attention/kernels and overlap input
8. add activation checkpointing or sharding only if memory requires it
9. scale devices only after single-device utilization is understood
10. compare cost-to-quality and document quality regressions

## 12. Systems capstone

Produce a costed plan for three budgets: laptop, single accelerator, and multi-accelerator.
For each, provide model size, sequence length, tokens, dtype, batch construction, memory
breakdown, FLOPs estimate, measured utilization assumption, wall time, monetary cost,
checkpoint plan, expected failure modes, and fallback configuration.

Then validate the estimator against one real small-scale run. Report estimation error and
update the model rather than hiding the discrepancy.

