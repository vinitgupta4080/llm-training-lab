# Module 2A: scalar gradients and autograd

## Why

Training needs the sensitivity of a scalar loss to every trainable parameter. A gradient tells
how a tiny parameter increase changes the loss locally. Gradient descent moves in the opposite
direction.

## Complete first experiment

1. Create a scalar leaf tensor `x=-3` with gradient tracking enabled.
2. Run the forward expression `y=x**2`.
3. Call `y.backward()` to traverse the recorded computation graph in reverse.
4. Read the accumulated derivative from `x.grad`.
5. Compare it with the manual derivative `2*x` and a finite-difference estimate.

The first coding exercise is added only after the learner predicts the four observable values.
