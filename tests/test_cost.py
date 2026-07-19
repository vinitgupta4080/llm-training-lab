import pytest

from llm_lab.cost import TrainingCostEstimate, activation_bytes


def test_training_cost_estimate():
    estimate = TrainingCostEstimate(
        parameters=1_000_000,
        tokens=3_600_000,
        devices=2,
        measured_tokens_per_second=1_000,
        price_per_device_hour=2.5,
    )
    assert estimate.approximate_training_flops == 21_600_000_000_000
    assert estimate.parameter_bytes == 2_000_000
    assert estimate.wall_hours == 1.0
    assert estimate.device_hours == 2.0
    assert estimate.estimated_cost == 5.0


def test_activation_estimate_and_invalid_input():
    assert activation_bytes(batch=2, sequence=8, width=16, layers=4) == 32_768
    with pytest.raises(ValueError):
        activation_bytes(batch=0, sequence=8, width=16, layers=4)

