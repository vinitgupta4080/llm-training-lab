"""Module 2 exercise: one shared weight trained from a batch of points."""

import torch


x = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([2.0, 4.0, 6.0])
w = torch.tensor(1.0, requires_grad=True)

# TODO 1: Calculate all three predictions using the shared weight w.
predictions = None

# TODO 2: Calculate one squared loss per point. Do not reduce yet.
losses = None

# TODO 3: Reduce the three losses to one scalar mean loss.
mean_loss = None

# TODO 4: Run the backward pass from the scalar mean loss.


assert predictions is not None
assert losses is not None
assert mean_loss is not None
assert torch.allclose(predictions, torch.tensor([1.0, 2.0, 3.0]))
assert torch.allclose(losses, torch.tensor([1.0, 4.0, 9.0]))
assert torch.isclose(mean_loss, torch.tensor(14.0 / 3.0))
assert w.grad is not None
assert torch.isclose(w.grad, torch.tensor(-28.0 / 3.0))

print("Batched mean-loss exercise passed.")
