"""Module 2 exercise: repeat clear → forward → backward → update."""

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
optimizer = torch.optim.SGD([weight], lr=0.01)
loss_history: list[float] = []

for step in range(20):
    # TODO 1: Clear the gradient retained from the previous training step.
    optimizer.zero_grad(set_to_none=True)
    # TODO 2: Calculate predictions with the current shared weight.
    predictions = inputs @ weight

    # TODO 3: Calculate one scalar mean squared loss.
    loss = ((targets - predictions) ** 2).mean()

    # TODO 4: Calculate and store weight.grad.
    loss.backward()

    # Save the pre-update loss for measurement.
    loss_history.append(loss.detach().item() if loss is not None else float("nan"))

    # TODO 5: Update weight once.
    optimizer.step()


assert len(loss_history) == 20
assert all(torch.isfinite(torch.tensor(loss_history)))
assert loss_history[-1] < loss_history[0]
assert loss_history[-1] < 0.2

print(f"Initial loss: {loss_history[0]:.6f}")
print(f"Final loss:   {loss_history[-1]:.6f}")
print("Learned weight:")
print(weight.detach())
print("Training-loop exercise passed.")
