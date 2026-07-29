# Experiment: float32 overflow and underflow checkpoints

## Claim and design

- Date / commit / environment hash: 2026-07-28; local working tree
- What-if question: What changes when only the magnitude of one float32 trainable weight changes?
- Mechanistic hypothesis: Squaring `1e20` will overflow the loss to `inf`; squaring `1e-30` will
  underflow the loss to zero; the corresponding gradients `2e20` and `2e-30` will remain finite.
- Observation that would falsify it: Either extreme loss remains equal to its mathematical real
  value, or either extreme gradient becomes non-finite/zero.
- Competing explanation: Autograd might differentiate the already-rounded stored loss value and
  therefore return zero for an underflowed loss or `inf` for an overflowed loss.
- Independent variable (exactly one): initial weight magnitude (`1e-30`, `3`, or `1e20`)
- Dependent variables: prediction, loss, gradient, and `isfinite` status at every stage
- Controlled variables: formula `prediction=1*w`, target `0`, squared error, float32, CPU
- Seeds: not applicable; calculation is deterministic and contains no randomness
- Data manifest, version, split, and contamination check: not applicable
- Model and optimizer configuration: one scalar leaf parameter; no optimizer update

## Accounting

- Total/trainable parameters: 1 / 1
- Training and evaluation tokens: not applicable
- Estimated FLOPs and peak memory: constant-size scalar calculation; negligible
- Device, dtype, elapsed time, and throughput: CPU, float32; timing is not meaningful at this size

## Results

| Case | Weight | Prediction | Loss | Gradient | First abnormal stage |
|---|---:|---:|---:|---:|---|
| Underflow | `1e-30` | `1e-30` | `0` | `2e-30` | loss |
| Ordinary | `3` | `3` | `9` | `6` | none |
| Overflow | `1e20` | `1e20` | `inf` | `2e20` | loss |

The prediction and gradient stayed finite in all three cases. The squared loss was the first
operation to leave the useful float32 range in both extreme cases.

## Interpretation

- Does the evidence support the hypothesis? Yes.
- Mechanism linking the intervention to the observation: squaring doubles the exponent, while
  the square derivative multiplies the original input by only two.
- Evidence for/against the competing explanation: the finite gradients falsify the idea that
  backward simply differentiates the rounded scalar loss value. Autograd follows the recorded
  operation and its local derivative rule.
- Limitations and what does not generalize to larger models: deeper graphs can overflow or
  underflow at different operations, and their gradients may also become non-finite or zero.
- Falsification or replication experiment: repeat with float64 while holding the values and
  formula fixed.
- Next single-variable experiment: find the float32 weight magnitude at which the squared loss
  first becomes non-finite.
