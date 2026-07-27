"""Controlled learning-rate sweep for the tiny matrix model."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch


LEARNING_RATES = [0.0001, 0.001, 0.01, 0.03, 0.1]
SEEDS = [0, 1, 2]
STEPS = 20
OUTPUT = Path("runs/module-02-learning-rate/metrics.json")

inputs = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
)
targets = torch.tensor(
    [[[2.0, 6.0], [6.0, 12.0]], [[10.0, 18.0], [14.0, 24.0]]]
)
validation_inputs = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
validation_targets = torch.tensor([[[4.0, 9.0], [8.0, 15.0]]])


def mean_squared_error(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return ((predictions - labels) ** 2).mean()


def run_once(learning_rate: float, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=learning_rate)
    training_losses: list[float] = []
    started = time.perf_counter()

    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = mean_squared_error(inputs @ weight, targets)
        training_losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()

    elapsed_seconds = time.perf_counter() - started
    with torch.no_grad():
        validation_loss = mean_squared_error(
            validation_inputs @ weight, validation_targets
        ).item()

    final_loss = training_losses[-1]
    if not math.isfinite(final_loss) or final_loss > training_losses[0]:
        behavior = "unstable"
    elif final_loss < 0.2:
        behavior = "effective"
    else:
        behavior = "slow"

    return {
        "learning_rate": learning_rate,
        "seed": seed,
        "steps": STEPS,
        "initial_training_loss": training_losses[0],
        "final_training_loss": final_loss,
        "validation_loss": validation_loss,
        "behavior": behavior,
        "elapsed_seconds": elapsed_seconds,
        "loss_curve": training_losses,
        "final_weight": weight.detach().tolist(),
    }


def main() -> None:
    results = [
        run_once(learning_rate, seed)
        for learning_rate in LEARNING_RATES
        for seed in SEEDS
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")

    print("lr       train start    train final    validation    behavior")
    for learning_rate in LEARNING_RATES:
        result = next(
            item
            for item in results
            if item["learning_rate"] == learning_rate and item["seed"] == 0
        )
        print(
            f"{learning_rate:<8g} "
            f"{result['initial_training_loss']:>11.6g} "
            f"{result['final_training_loss']:>14.6g} "
            f"{result['validation_loss']:>13.6g} "
            f"{result['behavior']}"
        )

    assert all(
        results[index]["loss_curve"] == results[index + 1]["loss_curve"]
        for index in range(0, len(results), len(SEEDS))
        for _ in [0]
        if index + 1 < len(results)
    ), "Seeds should not affect this deterministic experiment."


if __name__ == "__main__":
    main()
