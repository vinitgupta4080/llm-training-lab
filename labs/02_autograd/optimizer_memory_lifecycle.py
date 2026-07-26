"""Trace what persists in memory during a complete SGD training step."""

import torch


x = torch.tensor(-3.0, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

for step in range(3):
    print(f"\n--- training step {step} ---")
    print(f"before clear:   x={x.item():.4f}, x.grad={x.grad}")

    # Discard the gradient retained from the previous training step.
    optimizer.zero_grad(set_to_none=True)
    print(f"after clear:    x={x.item():.4f}, x.grad={x.grad}")

    # Forward pass: creates a loss tensor and a computation graph linking it to x.
    loss = x**2
    print(
        f"after forward:  x={x.item():.4f}, loss={loss.item():.4f}, "
        f"grad_fn={type(loss.grad_fn).__name__}"
    )

    # Backward pass: traverses the graph and accumulates the derivative into x.grad.
    loss.backward()
    print(f"after backward: x={x.item():.4f}, x.grad={x.grad.item():.4f}")

    # SGD reads x.grad and updates x. It deliberately leaves x.grad unchanged.
    optimizer.step()
    print(f"after step:     x={x.item():.4f}, x.grad={x.grad.item():.4f}")

assert torch.isclose(x.detach(), torch.tensor(-1.5360))
print("\nOptimizer memory-lifecycle trace passed.")
