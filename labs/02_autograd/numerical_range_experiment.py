"""Inspect overflow and underflow at each stage of a tiny training calculation."""

from __future__ import annotations

import torch


def gradients_are_safe(
    loss: torch.Tensor,
    parameters: list[torch.Tensor],
) -> bool:
    """Return whether a scalar loss and all available gradients are finite."""
    if not torch.isfinite(loss):
        return False

    for parameter in parameters:
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            return False

    return True


def inspect_weight(weight_value: float) -> dict[str, float | bool]:
    """Run L=(1*w-0)^2 and report which stages remain finite."""
    weight = torch.tensor(
        weight_value,
        dtype=torch.float32,
        requires_grad=True,
    )
    input_value = torch.tensor(1.0, dtype=torch.float32)
    target = torch.tensor(0.0, dtype=torch.float32)

    prediction = input_value * weight
    loss = (prediction - target) ** 2
    loss.backward()

    return {
        "weight": weight.item(),
        "prediction": prediction.item(),
        "loss": loss.item(),
        "gradient": weight.grad.item(),
        "weight_finite": bool(torch.isfinite(weight)),
        "prediction_finite": bool(torch.isfinite(prediction)),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradient_finite": bool(torch.isfinite(weight.grad)),
    }


def main() -> None:
    cases = {
        "underflowing loss": 1e-30,
        "ordinary": 3.0,
        "overflowing loss": 1e20,
    }

    for name, value in cases.items():
        result = inspect_weight(value)
        print(f"\n{name}")
        for key, measured_value in result.items():
            print(f"  {key:>18}: {measured_value}")

    underflow = inspect_weight(1e-30)
    ordinary = inspect_weight(3.0)
    overflow = inspect_weight(1e20)

    assert underflow["loss"] == 0.0
    assert underflow["gradient_finite"]
    assert ordinary["loss"] == 9.0
    assert ordinary["gradient"] == 6.0
    assert overflow["loss"] == float("inf")
    assert overflow["gradient_finite"]

    safe_weight = torch.tensor(3.0, requires_grad=True)
    safe_loss = safe_weight**2
    safe_loss.backward()
    assert gradients_are_safe(safe_loss, [safe_weight])

    unsafe_weight = torch.tensor(1e20, requires_grad=True)
    unsafe_loss = unsafe_weight**2
    unsafe_loss.backward()
    assert not gradients_are_safe(unsafe_loss, [unsafe_weight])

    unused_parameter = torch.tensor(1.0, requires_grad=True)
    assert unused_parameter.grad is None
    assert gradients_are_safe(safe_loss, [safe_weight, unused_parameter])


if __name__ == "__main__":
    main()
