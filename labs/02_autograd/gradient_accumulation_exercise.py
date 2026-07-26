"""Module 2 exercise: observe and clear accumulated gradients."""

import torch


x = torch.tensor(-3.0, requires_grad=True)

y1 = x**2
y1.backward()
first_gradient = x.grad.item()

y2 = x**2
y2.backward()
second_gradient = x.grad.item()

assert first_gradient == -6.0
assert second_gradient == -12.0

# TODO: Clear x.grad before building the next training step.

y3 = x**2
y3.backward()
third_gradient = x.grad.item()

assert third_gradient == -6.0

print("Gradient accumulation exercise passed.")
