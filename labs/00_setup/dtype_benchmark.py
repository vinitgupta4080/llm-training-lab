"""Module 0B: controlled float32 versus float64 CPU comparison."""

from __future__ import annotations

import statistics
import time

import torch


def benchmark(size: int, dtype: torch.dtype, warmups: int = 5, repeats: int = 20) -> dict:
    torch.manual_seed(42)
    left = torch.randn(size, size, dtype=dtype)
    right = torch.randn(size, size, dtype=dtype)

    for _ in range(warmups):
        _ = left @ right

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ = left @ right
        samples.append(time.perf_counter() - started)

    elapsed = statistics.median(samples)
    elements = left.numel() + right.numel()
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "input_mib": elements * left.element_size() / 2**20,
        "median_ms": elapsed * 1_000,
        "estimated_gflops_s": 2 * size**3 / elapsed / 1e9,
    }


def main() -> None:
    size = 1024
    results = [benchmark(size, dtype) for dtype in (torch.float32, torch.float64)]
    print(f"CPU square matmul size: {size}")
    print("dtype    input_MiB  median_ms  estimated_GFLOP/s")
    for row in results:
        print(
            f"{row['dtype']:7}  {row['input_mib']:9.2f}  {row['median_ms']:9.3f}  "
            f"{row['estimated_gflops_s']:18.2f}"
        )
    ratio = results[1]["median_ms"] / results[0]["median_ms"]
    print(f"\nMeasured float64/float32 latency ratio: {ratio:.2f}×")
    print("The ratio is a property of this workload and machine, not a dtype law.")


if __name__ == "__main__":
    main()

