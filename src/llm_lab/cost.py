"""Transparent first-order estimates for training-system planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingCostEstimate:
    parameters: int
    tokens: int
    devices: int
    measured_tokens_per_second: float
    price_per_device_hour: float = 0.0
    bytes_per_parameter: float = 2.0

    def __post_init__(self):
        if min(self.parameters, self.tokens, self.devices) <= 0:
            raise ValueError("parameters, tokens, and devices must be positive")
        if self.measured_tokens_per_second <= 0 or self.bytes_per_parameter <= 0:
            raise ValueError("throughput and byte width must be positive")
        if self.price_per_device_hour < 0:
            raise ValueError("price cannot be negative")

    @property
    def approximate_training_flops(self) -> int:
        return 6 * self.parameters * self.tokens

    @property
    def parameter_bytes(self) -> float:
        return self.parameters * self.bytes_per_parameter

    @property
    def wall_hours(self) -> float:
        return self.tokens / self.measured_tokens_per_second / 3600

    @property
    def device_hours(self) -> float:
        return self.wall_hours * self.devices

    @property
    def estimated_cost(self) -> float:
        return self.device_hours * self.price_per_device_hour


def activation_bytes(
    *, batch: int, sequence: int, width: int, layers: int, bytes_per_value: int = 2,
    saved_values_per_token_per_layer: int = 16,
) -> int:
    """Educational estimate; measure real peak memory because kernels differ."""
    values = (batch, sequence, width, layers, bytes_per_value, saved_values_per_token_per_layer)
    if any(value <= 0 for value in values):
        raise ValueError("all activation estimate inputs must be positive")
    return batch * sequence * width * layers * bytes_per_value * saved_values_per_token_per_layer

