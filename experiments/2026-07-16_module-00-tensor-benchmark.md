# Experiment: Module 0 tensor and matrix-multiplication benchmark

## Claim and design

- Date: 2026-07-16
- What-if question: How do tensor dtype and matrix size affect memory and measured throughput?
- Mechanistic hypothesis: float64 matmul will be about 4× slower; doubling square matrix
  dimensions will make the operation about 16× more expensive.
- Observation that would falsify it: measured results substantially different from these
  factors under a controlled comparison.
- Competing explanation: first-run overhead, CPU scheduling, device synchronization, or
  hardware-specific kernel behavior may explain timing differences.
- Independent variable: square matrix size (`128`, `512`, `1024`).
- Dependent variables: median/min/max time and estimated GFLOP/s.
- Controlled variables: device, float32 dtype, square shapes, warmups, repetitions, process.
- Seeds: performance experiment; repeated timings replace training seeds in this first lab.

## Predictions — complete before running

1. `[4, 128, 256]` elements, float32 MiB, and float16 MiB: learner initially calculated
   `4 * 18 * 256` elements and `32 * 4 * 18 * 256` bits (used 18 instead of 128).
2. Will float64 always take exactly 2× float32 time? Why? Predicted 4× slower.
3. FLOP growth and tensor-element growth when square dimensions double: predicted 16×.
4. Predicted size with highest GFLOP/s and reason: unsure; GFLOP/s is new.
5. Expected seeded behavior and possible accelerator nondeterminism: unsure; seeds and
   nondeterminism are new.

## Results — complete after running

- Hardware/software: Python 3.12.2, PyTorch 2.9.1, CPU, 8 PyTorch threads.
- Tensor output: 131,072 elements; 524,288 bytes; 0.500 MiB in float32.
- Benchmark table:

| Size | Median ms | Min ms | Max ms | Estimated GFLOP/s |
|---:|---:|---:|---:|---:|
| 128 | 0.005 | 0.005 | 0.006 | 786.48 |
| 512 | 0.131 | 0.124 | 0.211 | 2047.83 |
| 1024 | 1.481 | 1.227 | 2.167 | 1449.68 |

- Did the predictions hold? Memory prediction used the wrong middle dimension (`18`
  instead of `128`) and did not convert bits to bytes/MiB. The doubling prediction mixed
  quadratic storage growth (4×) and cubic matmul work growth (8×); neither is 16×.
  Float64 was not tested in this first run, so its slowdown prediction remains untested.
- Why might median, minimum, and maximum differ? OS scheduling, CPU frequency, cache state,
  thread scheduling, and other machine activity are plausible causes.
- Theoretical calculation versus measurement: FLOP counts are calculated from matrix shape;
  elapsed times are measured. Their ratio is an estimated achieved throughput.
- Result limitations: short CPU timings, one machine/process, no confidence interval, and no
  float64 comparison. The 128 result is especially sensitive to timer and launch overhead.
- Next falsification or replication experiment: calculate the corrections, explain why size
  512 achieved higher throughput than 128, then add a controlled float32/float64 comparison.

## Learner correction checkpoint

- Float16 memory: correctly explained that halving bits per element halves tensor storage.
- Scaling: correctly explained that doubling both square dimensions creates 4× elements and
  that `2(2N)^3 / 2N^3 = 8`, so matrix-multiplication work grows 8×.
- Remaining checkpoint: distinguish latency from throughput and explain random-seed behavior.
- Latency checkpoint attempt: learner proposed they are inverses. Refine this with the fixed-
  workload condition and examples involving different matrix sizes or concurrent requests.
- Seed checkpoint: learner correctly explained that `torch.manual_seed` initializes generator
  state and each subsequent random operation generates values and advances that state.
