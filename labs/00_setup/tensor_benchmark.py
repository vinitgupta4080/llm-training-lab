"""Module 0 lab: inspect tensors and benchmark matrix multiplication."""

from __future__ import annotations

import platform
import statistics
import time

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_matmul(
    size: int, device: torch.device, dtype: torch.dtype, warmups: int = 3, repeats: int = 10
) -> dict[str, float]:
    left = torch.randn(size, size, device=device, dtype=dtype)
    right = torch.randn(size, size, device=device, dtype=dtype)

    for _ in range(warmups):
        _ = left @ right
    synchronize(device)

    timings = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        _ = left @ right
        synchronize(device)
        timings.append(time.perf_counter() - started)

    median_seconds = statistics.median(timings)
    theoretical_flops = 2 * size**3
    return {
        "median_ms": median_seconds * 1_000,
        "min_ms": min(timings) * 1_000,
        "max_ms": max(timings) * 1_000,
        "gflops_per_second": theoretical_flops / median_seconds / 1e9,
    }


def main() -> None:
    device = preferred_device()
    print(f"Python:  {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {device}")
    print(f"Threads: {torch.get_num_threads()}")

    tensor = torch.zeros((4, 128, 256), dtype=torch.float32, device=device)
    storage_bytes = tensor.numel() * tensor.element_size()
    print("\nTensor inspection")
    print(f"shape={tuple(tensor.shape)} stride={tensor.stride()} contiguous={tensor.is_contiguous()}")
    print(f"elements={tensor.numel():,} bytes={storage_bytes:,} MiB={storage_bytes / 2**20:.3f}")
    transposed = tensor.transpose(0, 1)
    print(
        f"transpose shape={tuple(transposed.shape)} stride={transposed.stride()} "
        f"contiguous={transposed.is_contiguous()}"
    )

    print("\nReproducibility check")
    torch.manual_seed(42)
    first = torch.randn(5, device=device)
    torch.manual_seed(42)
    second = torch.randn(5, device=device)
    print(f"same seed gives same sampled values: {torch.equal(first, second)}")

    print("\nSquare matrix multiplication (float32)")
    print("size  median_ms  min_ms  max_ms  estimated_GFLOP/s")
    for size in (128, 512, 1024):
        result = benchmark_matmul(size, device, torch.float32)
        print(
            f"{size:4d}  {result['median_ms']:9.3f}  {result['min_ms']:6.3f}  "
            f"{result['max_ms']:6.3f}  {result['gflops_per_second']:17.2f}"
        )

    print("\nInterpretation is your job: compare results with your written predictions.")


if __name__ == "__main__":
    main()

