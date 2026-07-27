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
predictions = inputs @ weight

# TODO 2: Verify one token manually by selecting inputs[0, 0] and multiplying by weight.
first_prediction = inputs[0, 0] @ weight

# TODO 3: Calculate the scalar mean squared error across both batches, both tokens,
# and both output dimensions.
mean_loss = ((targets - predictions) ** 2).mean()

# TODO 4: Run backward so weight.grad receives a [C_in, C_out] gradient matrix.
mean_loss.backward()

assert predictions is not None
assert first_prediction is not None
assert mean_loss is not None
assert predictions.shape == (2, 2, 2)
assert torch.allclose(first_prediction, torch.tensor([1.0, 2.0]))
assert torch.isclose(mean_loss, torch.tensor(70.5))
assert weight.grad is not None
assert weight.grad.shape == weight.shape
assert torch.allclose(weight.grad, torch.tensor([[-21.0, -50.0], [-25.0, -60.0]]))

loss_before_update = mean_loss.detach()
optimizer = torch.optim.SGD([weight], lr=0.01)

# TODO 5: Use the gradient already stored in weight.grad to update weight once.
optimizer.step()
# TODO 6: Recalculate predictions using the updated weight.
new_predictions = inputs @ weight

# TODO 7: Recalculate the scalar mean squared error.
new_mean_loss = ((new_predictions - targets) ** 2).mean()

assert new_predictions is not None
assert new_mean_loss is not None
assert new_mean_loss < loss_before_update

print(f"Loss before SGD: {loss_before_update.item():.4f}")
print(f"Loss after SGD:  {new_mean_loss.item():.4f}")
print("Batched matrix-gradient and SGD exercise passed.")
