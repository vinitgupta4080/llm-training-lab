# Experiment: naive BPE cost scaling

## Question

How do learned merge count, vocabulary size, training latency, encoding latency, and token count
change when all text and measurement settings remain fixed?

## Controls

- Training text: repeated `banana bandana banana cabana`, fixed at 12,399 UTF-8 bytes.
- Evaluation text: repeated `banana bandana`, fixed at 1,499 UTF-8 bytes.
- Merge budgets: 0, 5, 10, 20, 40.
- Implementation: transparent Python full-scan BPE.
- Measurement: 1 warmup; median of 5 training and 100 encoding repetitions.

## Prediction

- Training latency: goes up.
- Encoding latency: goes up.
- Vocabulary size: goes up.
- Evaluation token count: goes down.

## Results

| Requested merges | Actual rules | Vocabulary | Train median (ms) | Encode median (ms) | Byte tokens | BPE tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 259 | 0.035 | 0.0045 | 1,499 | 1,499 |
| 5 | 5 | 264 | 6.743 | 0.4458 | 1,499 | 401 |
| 10 | 10 | 269 | 9.024 | 0.6419 | 1,499 | 203 |
| 20 | 20 | 279 | 10.088 | 0.7201 | 1,499 | 104 |
| 40 | 28 | 287 | 9.991 | 0.7940 | 1,499 | 104 |

## Interpretation

- Training and encoding latency rose strongly from the byte baseline, supporting the learner's
  directional prediction. Training time was not strictly monotonic at the end because training
  stopped after 28 actual rules and small timing differences are noise.
- Vocabulary grew by one per rule actually learned, not one per requested budget unit.
- Familiar-text token count fell sharply, supporting the learner's prediction, but plateaued at
  104 tokens after 20 rules because later learned rules did not match the evaluation text.
- This transparent implementation scans once per rule. Production tokenizers use optimized data
  structures and native code, so the exact latency values do not generalize.
