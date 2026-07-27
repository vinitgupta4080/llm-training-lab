"""Hold learning rate fixed and measure the effect of more optimizer steps."""

from __future__ import annotations

import time
from statistics import median

import torch


LEARNING_RATE = 0.001
STEP_BUDGETS = [20, 200, 2_000]

inputs = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
)
targets = torch.tensor(
    [[[2.0, 6.0], [6.0, 12.0]], [[10.0, 18.0], [14.0, 24.0]]]
)


def run(step_budget: int) -> tuple[float, float, list[float]]:
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=LEARNING_RATE)
    losses: list[float] = []
    started = time.perf_counter()

    for _ in range(step_budget):
        optimizer.zero_grad(set_to_none=True)
        loss = ((inputs @ weight - targets) ** 2).mean()
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()

    return losses[-1], time.perf_counter() - started, losses


def main() -> None:
    print("steps    final loss       elapsed (ms)    cost vs 20")
    results: dict[int, tuple[float, float, list[float]]] = {}
    run(2)  # Warm up one-time PyTorch setup before measuring.
    for steps in STEP_BUDGETS:
        repetitions = [run(steps) for _ in range(5)]
        final_loss, _, losses = repetitions[0]
        elapsed = median(item[1] for item in repetitions)
        results[steps] = (final_loss, elapsed, losses)
        print(f"{steps:<8} {final_loss:<16.8f} {elapsed * 1_000:<15.3f} {steps / 20:>6.0f}x")

    for _, _, losses in results.values():
        assert all(after <= before for before, after in zip(losses, losses[1:]))
    assert results[2_000][0] < results[200][0] < results[20][0]


if __name__ == "__main__":
    main()
