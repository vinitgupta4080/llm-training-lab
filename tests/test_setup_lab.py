import importlib.util
from pathlib import Path

import torch


path = Path(__file__).parents[1] / "labs/00_setup/tensor_benchmark.py"
spec = importlib.util.spec_from_file_location("tensor_benchmark", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_benchmark_returns_positive_measurements():
    result = module.benchmark_matmul(8, torch.device("cpu"), torch.float32, warmups=1, repeats=2)
    assert result["median_ms"] > 0
    assert result["gflops_per_second"] > 0
