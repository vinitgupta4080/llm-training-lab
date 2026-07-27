# Experiment: Learning-rate sweep

## Claim and design

- Date / commit / environment hash: 2026-07-26 / working tree / local CPU environment
- What-if question: How does learning rate affect loss after the same 20-step budget?
- Mechanistic hypothesis: Very small updates learn slowly, moderate updates reduce loss quickly,
  and oversized updates repeatedly overshoot and become unstable.
- Observation that would falsify it: All tested rates produce similar stable loss curves.
- Competing explanation: A run may look better simply because it received more updates or started
  from different weights.
- Independent variable (exactly one): SGD learning rate.
- Dependent variables: Training-loss curve, final training loss, validation loss, stability.
- Controlled variables: Data, validation data, initial weight, optimizer, dtype, device, loss, and
  20 optimizer steps.
- Seeds (minimum three for noisy comparisons): 0, 1, 2. This experiment has no stochastic
  operation, so identical results across seeds are expected and verified.
- Data manifest, version, split, and contamination check: Four fixed training vectors and two
  fixed unseen validation vectors generated from the same known linear mapping; no contamination.
- Model and optimizer configuration: One trainable `2×2` float32 weight; SGD without momentum.

## Accounting

- Total/trainable parameters: 4 / 4.
- Training and evaluation tokens: Not language tokens; 4 training rows and 2 validation rows.
- Estimated FLOPs and peak memory: Tiny educational workload; accounting will be calculated in a
  later profiling exercise.
- Device, dtype, elapsed time, and throughput: CPU / float32; per-run elapsed time is stored in
  `runs/module-02-learning-rate/metrics.json`.

## Results

Run:

```bash
python3 labs/02_autograd/learning_rate_sweep.py
```

Machine-readable loss curves are written to `runs/module-02-learning-rate/metrics.json`.

| Learning rate | Initial train loss | Final train loss | Validation loss | Behavior |
|---:|---:|---:|---:|---|
| 0.0001 | 70.5 | 58.0918 | 31.9640 | Slow |
| 0.001 | 70.5 | 9.77733 | 5.29169 | Slow |
| 0.01 | 70.5 | 0.105712 | 0.0919214 | Effective |
| 0.03 | 70.5 | 0.0980932 | 0.0849651 | Effective |
| 0.1 | 70.5 | 1.24083e25 | 1.09761e26 | Unstable |

All three seed runs were identical, as expected for this deterministic setup.

## Interpretation

- Does the evidence support the hypothesis? Yes. Rates `0.0001` and `0.001` made insufficient
  progress in 20 steps, `0.01` and `0.03` were effective, and `0.1` diverged.
- Mechanism linking the intervention to the observation: `weight -= learning_rate * gradient`;
  learning rate directly scales every update.
- Evidence for/against the competing explanation: All non-learning-rate variables were controlled,
  so different step budgets and initial weights cannot explain the result.
- Limitations and what does not generalize to larger models: This is a convex, deterministic,
  four-parameter problem and cannot establish the best rate for an LLM.
- Falsification or replication experiment: Repeat with random initialization and noisy batches.
- Next single-variable experiment: Keep the selected learning rate fixed and change step count.
