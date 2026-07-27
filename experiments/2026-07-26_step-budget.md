# Experiment: Training-step budget

## Claim and design

- What-if question: Can a small learning rate catch up when it receives more training steps?
- Hypothesis: At fixed learning rate `0.001`, training loss will decrease monotonically as the
  budget grows from 20 to 200 to 2,000 steps, while compute cost grows linearly.
- Falsifier: A larger step budget produces a higher training loss in this deterministic,
  full-batch, convex problem.
- Independent variable: Number of optimizer steps.
- Controlled variables: Learning rate, initial weight, model, data, optimizer, dtype, and device.
- Dependent variables: Final training loss and elapsed time.

## Results

Run:

```bash
python3 labs/02_autograd/step_budget_experiment.py
```

Timing uses one warm-up plus the median of five repetitions.

| Steps | Final training loss | Median elapsed | Relative step cost |
|---:|---:|---:|---:|
| 20 | 9.77733040 | 1.809 ms | 1× |
| 200 | 0.10552691 | 17.975 ms | 10× |
| 2,000 | 0.07409564 | 178.954 ms | 100× |

## Interpretation

- More steps can compensate for a small learning rate, but every extra step consumes compute.
- The first 10× increase in steps reduced loss substantially; the next 10× increase produced a
  much smaller improvement, demonstrating diminishing returns.
- This tiny deterministic result does not imply that stochastic LLM training loss decreases on
  every individual step.
