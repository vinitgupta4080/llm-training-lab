"""Module 2 exercise: [B,T,C_in] input through a shared weight matrix."""

import torch


inputs = torch.tensor(
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]
)
targets = torch.tensor(
    [
        [[2.0, 6.0], [6.0, 12.0]],
        [[10.0, 18.0], [14.0, 24.0]],
    ]
)
weight = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0]],
    requires_grad=True,
)

# TODO 1: Apply the shared weight to every token vector with matrix multiplication.
predictions = None

# TODO 2: Verify one token manually by selecting inputs[0, 0] and multiplying by weight.
first_prediction = None

# TODO 3: Calculate the scalar mean squared error across both batches, both tokens,
# and both output dimensions.
mean_loss = None

# TODO 4: Run backward so weight.grad receives a [C_in, C_out] gradient matrix.


assert predictions is not None
assert first_prediction is not None
assert mean_loss is not None
assert predictions.shape == (2, 2, 2)
assert torch.allclose(first_prediction, torch.tensor([1.0, 2.0]))
assert torch.isclose(mean_loss, torch.tensor(70.5))
assert weight.grad is not None
assert weight.grad.shape == weight.shape
assert torch.allclose(weight.grad, torch.tensor([[-21.0, -50.0], [-25.0, -60.0]]))

print("Batched matrix-gradient exercise passed.")
