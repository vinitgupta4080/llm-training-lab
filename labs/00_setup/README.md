# Module 0 — Tensors, devices, memory, and trustworthy measurement

## Why this comes first

An LLM is mostly tensor transformations executed repeatedly. Before attention or training,
you must be able to answer:

- What does each tensor dimension mean?
- How many values and bytes does it occupy?
- Is an operation compute-bound or memory-bound?
- Is a timing measurement trustworthy?
- Can a result be reproduced?

## Core model

A tensor has:

- **shape**: number of entries along each axis
- **stride**: steps in storage needed to move one position along each axis
- **dtype**: representation and bytes per stored value
- **device**: processor and memory where operations execute
- **requires_grad**: whether autograd must record operations affecting it

“Tensor” is the general container; scalar, vector, and matrix describe special ranks:

| Rank | Common name | Example shape | Meaning |
|---:|---|---|---|
| 0 | scalar | `[]` | one value |
| 1 | vector | `[4]` | one axis with four values |
| 2 | matrix | `[3, 4]` | two axes, often rows and columns |
| 3+ | higher-rank tensor | `[2, 3, 4]` | three or more axes |

Rank is the number of axes, not the number of elements. A vector with one million values is
still rank 1. A tiny shape `[2, 2, 2]` is rank 3. In LLM code, a common hidden-state tensor is
`[batch, time, channels]`, which is rank 3.

For a dense contiguous tensor:

```text
number_of_elements = product(shape)
storage_bytes = number_of_elements * bytes_per_element
```

Example: `[batch=8, time=512, channels=768]` in float32:

```text
elements = 8 * 512 * 768 = 3,145,728
bytes = 3,145,728 * 4 = 12,582,912 bytes ≈ 12.0 MiB
```

This counts only that tensor's storage. A training operation can also allocate gradients,
saved activations, output tensors, temporary workspaces, and optimizer state.

## Matrix multiplication shape reasoning

For `C = A @ B`:

```text
A: [M, K]
B: [K, N]
C: [M, N]
approximate operations: 2 * M * K * N FLOPs
```

The `K` dimensions must match. Each output value is a dot product of length `K`.

## Broadcasting

PyTorch aligns shapes from the right. Dimensions are compatible when they are equal or one
of them is `1` (missing leading dimensions behave like `1`). Broadcasting often creates a
logical expanded view without copying values, but the following operation still performs
work over the expanded output.

```text
[B, T, C] + [C]       -> [B, T, C]
[B, T, C] + [T, 1]   -> [B, T, C]
[B, T, C] + [B, T]   -> incompatible in general
```

## Measurement protocol

1. Warm up to reduce first-run effects.
2. Repeat measurements; one timing is weak evidence.
3. Synchronize asynchronous devices before starting and stopping a timer.
4. Report shapes, dtype, device, versions, and thread configuration.
5. Use the median and spread, not only the fastest run.
6. Separate theoretical calculations from measured values.

## Random seeds and reproducibility

Computers usually produce **pseudorandom** values: a deterministic algorithm creates a
sequence that looks random. A seed initializes the generator's internal state.

```python
torch.manual_seed(42)
first = torch.randn(3)
torch.manual_seed(42)
second = torch.randn(3)
assert torch.equal(first, second)
```

The core rule is:

```text
same algorithm + same seed + same ordered random calls = same sequence
```

The seed is a starting point, not the random value. Every draw advances generator state.
Two runs with the same seed diverge if one makes an extra random call or changes call order.

In LLM training, randomness affects parameter initialization, batch order, dropout masks,
data augmentation, and sampled tokens. A seed alone does not guarantee an identical training
run: code, data, library versions, hardware, worker scheduling, and nondeterministic parallel
kernels also matter. Floating-point reductions in different orders can differ slightly.

Distinguish three levels:

1. **Seeded randomness:** the intended random sequence can be replayed.
2. **Deterministic execution:** relevant operations return identical results each run.
3. **Reproducible experiment:** code, data, environment, config, and state are recorded.

Exact checkpoint resume requires saving current generator state, not merely the original seed,
because training may already be thousands of random draws into the sequence.

## Lab 0B — dtype is not only memory width

A floating-point dtype allocates bits to three roles: sign, exponent, and fraction. More
exponent bits increase representable range; more fraction bits increase precision. Storage
width is exact, but execution speed depends on hardware and workload.

| dtype | storage | approximate decimal precision |
|---|---:|---:|
| float16 | 2 bytes | 3–4 significant digits |
| float32 | 4 bytes | 6–7 significant digits |
| float64 | 8 bytes | 15–16 significant digits |

Before running `dtype_benchmark.py`, record the float64/float32 memory ratio and predicted
latency ratio in `experiments/2026-07-16_module-00-dtype.md`. Then run:

```bash
python labs/00_setup/dtype_benchmark.py
```

Do not infer speed from byte width alone. Optimized instructions, vector width, cache,
bandwidth, matrix size, and device capabilities determine measured performance.

## Lab 0C — storage, views, strides, and contiguity

A tensor's values live in physical storage. Shape and stride describe how logical indices map
onto that storage. For a contiguous `[3, 4]` matrix, stride `[4, 1]` means moving one row jumps
four storage positions, while moving one column jumps one position.

Transposing can return a view with shape `[4, 3]` and stride `[1, 4]` without copying the 12
values. The view is non-contiguous because its logical iteration order no longer matches the
physical storage order. Some operations accept strided inputs; others make a contiguous copy,
so a cheap transpose can create cost later.

Prediction: does `x.transpose(0, 1)` allocate new value storage immediately? Explain why.

## Lab 0D — broadcasting

### Why it is used

Suppose LLM hidden states have shape `[batch=2, time=3, channels=4]`. A layer has one learned
bias per channel, shape `[4]`. The same four bias values must be added to every token in every
batch. Manually copying that bias into shape `[2,3,4]` would waste code and storage.

### How it is used

PyTorch broadcasts automatically for compatible elementwise operations:

```python
hidden = torch.randn(2, 3, 4)
bias = torch.randn(4)
output = hidden + bias  # shape [2, 3, 4]
```

Other LLM uses include applying attention masks across heads/batches, normalization scale and
bias across tokens, and adding positional information across a batch.

### Technical mechanism

Align dimensions from the right. Each aligned pair must be equal or one must be `1`; missing
leading dimensions behave as `1`.

```text
[2, 3, 4]
      [4]  -> [2, 3, 4]

[2, 3, 4]
   [3, 1]  -> [2, 3, 4]
```

Broadcasting usually creates a logical expanded view rather than physically copying the small
input. Expanded axes can have stride zero, meaning different logical positions reread the same
stored value. The resulting addition still calculates every output element.

For bias storage `[b0,b1,b2,b3]` logically expanded to `[B=2,T=3,C=4]` with stride `[0,0,1]`,
the storage offset for logical index `[batch,token,channel]` is:

```text
offset = batch*0 + token*0 + channel*1 = channel
```

Thus `[0,0,2]`, `[0,1,2]`, and `[1,2,2]` all read storage position 2 (`b2`). No repeated bias
values are stored. Because multiple logical positions alias the same storage, writing in-place
through an expanded view is restricted or unsafe; expansion is mainly used for reading during
the broadcasted operation.

## Why Transformer representations are called hidden states

The model receives visible inputs (token IDs) and produces visible outputs (logits or generated
tokens). Between them, every Transformer layer maintains internal token vectors. These are
called **hidden states** because they are internal latent representations rather than directly
provided labels or final predictions.

```text
token IDs [B,T]
→ embeddings [B,T,C]
→ hidden states after each block [B,T,C]
→ logits [B,T,V]
```

“State” means the current internal representation at a particular layer and token position.
In decoder-only Transformers it is not a single permanent memory: each layer computes a new
hidden-state tensor from the preceding one. Its shape often remains `[B,T,C]` while its values
and encoded information change.

### What is a channel?

A channel is one coordinate of a token's hidden vector. If `C=4`, every token is represented
by four numbers and therefore has four channels. In `[B,T,C]`, selecting `hidden[:,:,2]`
selects channel 2 across every batch and token.

Channels are analogous to columns in a feature table, but their meanings are learned. One
channel usually cannot be labeled simply as “noun” or “animal”; information is commonly
distributed across many channels, and each channel can participate in multiple features.
Changing model width from 128 to 256 means increasing the number of hidden channels per token.

## Lab 0A — predictions before execution

Do not run `tensor_benchmark.py` until you have written predictions for these questions:

1. A tensor has shape `[4, 128, 256]`. How many elements and MiB does it use in float32?
   What about float16?
2. Will float64 matrix multiplication always be exactly twice as slow as float32 on your
   machine? State why or why not.
3. If both matrix dimensions double for square `A @ B`, by what factor do theoretical FLOPs
   grow? By what factor do the input/output element counts grow?
4. Which size—128, 512, or 1024—do you predict will achieve the highest measured GFLOP/s?
5. If the same random seed is reset before creating a tensor, what should match? What might
   still be nondeterministic on accelerators?

Record the answers in `experiments/2026-07-16_module-00-tensor-benchmark.md`.

Then execute:

```bash
python labs/00_setup/tensor_benchmark.py
```

## Completion questions

- Why can a non-contiguous transpose share storage with its input?
- Why does theoretical 8× FLOP growth not guarantee exactly 8× wall time?
- Why is the fastest individual benchmark repetition a biased estimator?
- What must be stored in addition to parameters during AdamW training?
