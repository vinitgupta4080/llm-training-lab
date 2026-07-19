# Experiment: Module 0B dtype and performance

## Prediction — complete before running

- Float64/float32 input-memory ratio: 2×; 8 MiB becomes 16 MiB.
- Predicted float64/float32 latency ratio: 4×.
- Why the memory ratio is predictable: float64 uses 8 bytes per element while float32 uses 4.
- Why the latency ratio is hardware/workload dependent: discussed before prediction; exact
  learner explanation not yet assessed.
- Observation that would falsify the latency prediction: a measured ratio substantially
  different from 4× under the controlled setup.

## Controlled design

- Independent variable: dtype (`float32`, `float64`).
- Controls: CPU, 1024×1024 matrices, seed 42, inputs per dtype, warmups, repetitions,
  thread configuration, and operation.
- Metrics: combined input MiB, median latency, estimated GFLOP/s, latency ratio.
- Limitation: random values are numerically corresponding in distribution, not guaranteed
  to be bitwise-equivalent values after independent dtype generation.

## Results — complete after running

- Hardware/software: CPU; Python/PyTorch environment recorded by Module 0A; square size 1024.
- Output table:

| dtype | Combined input MiB | Median ms | Estimated GFLOP/s |
|---|---:|---:|---:|
| float32 | 8.00 | 1.266 | 1696.25 |
| float64 | 16.00 | 5.446 | 394.34 |

- Prediction versus measurement: 16 MiB prediction was exact. Predicted 4× latency; measured
  4.30×, very close for this run.
- Mechanistic interpretation: storage follows dtype width exactly. The CPU achieved much lower
  float64 arithmetic throughput for this matrix workload; byte width alone does not derive the
  observed 4.30× ratio.
- Measurement limitations: one machine/process, one matrix size, short timings, no reported
  thread affinity or variability table, and independently generated input tensors per dtype.
- Next experiment: learner explains why 4.30× is a measurement rather than a universal dtype
  rule, then explore strides/contiguity and broadcasting.
