"""Module 2 exercise: compare a manual derivative with PyTorch autograd."""

import torch


# TODO 1: Create scalar x=-3.0 with gradient tracking enabled.
x = torch.tensor(-3.0, requires_grad=True)

# TODO 2: Compute y=x**2.
y = x**2

# TODO 3: Run backward and save its return value in backward_result.
backward_result = y.backward()

# These checks state the observable contract without implementing the TODOs.
assert x.item() == -3.0
assert y.item() == 9.0
assert backward_result is None
assert x.grad.item() == -6.0

manual_gradient = 2 * x.item()
assert x.grad.item() == manual_gradient

print("Scalar autograd exercise passed.")
