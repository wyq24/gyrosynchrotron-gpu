import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache-codex")

from mcmc_example.mcmc_backend_gpu_batched import build_legacy_8d_batched_backend
from mcmc_example.spec_utils import simulate_spectrum_optimized


BASE_PARAMS_8D = np.array([5.0, 2.0, 1.0, 10.0, 5.0, 4.0, 45.0, 10.0], dtype=np.float64)


def _make_batch(batch_size: int) -> np.ndarray:
    batch = np.repeat(BASE_PARAMS_8D[None, :], batch_size, axis=0)
    offsets = np.linspace(-1.0, 1.0, batch_size, dtype=np.float64)
    batch[:, 0] += 0.8 * offsets
    batch[:, 1] += 0.3 * offsets
    batch[:, 3] += 0.2 * offsets
    batch[:, 4] += 0.3 * offsets
    batch[:, 6] += 8.0 * offsets
    batch[:, 7] += 4.0 * offsets
    return batch


def _time_legacy_loop(params_batch: np.ndarray, repeats: int, lib_path: str) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        for row in params_batch:
            simulate_spectrum_optimized(row, libname=lib_path)
    return time.perf_counter() - start


def _time_batched_backend(params_batch: np.ndarray, repeats: int, lib_path: str, backend_name: str) -> float:
    backend = build_legacy_8d_batched_backend(
        lib_path=lib_path,
        batch_capacity=params_batch.shape[0],
        backend=backend_name,
    )
    backend.simulate_batch(params_batch)

    start = time.perf_counter()
    for _ in range(repeats):
        backend.simulate_batch(params_batch)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark old single-call spectra vs new batched wrapper batches.")
    parser.add_argument("--lib", default="source/MWTransferArr.so", help="Path to MWTransferArr shared library")
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda"], help="Native backend")
    parser.add_argument("--batch-size", type=int, default=32, help="Likelihood-style batch size")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated timing loops")
    args = parser.parse_args()

    params_batch = _make_batch(args.batch_size)
    legacy_seconds = _time_legacy_loop(params_batch, args.repeats, args.lib)
    batched_seconds = _time_batched_backend(params_batch, args.repeats, args.lib, args.backend)

    print("MCMC forward benchmark")
    print(f"backend: {args.backend}")
    print(f"batch_size: {args.batch_size}")
    print(f"repeats: {args.repeats}")
    print(f"legacy_single_call_seconds: {legacy_seconds:.6f}")
    print(f"new_batched_seconds: {batched_seconds:.6f}")
    print(f"speedup_vs_legacy: {legacy_seconds / batched_seconds:.2f}x")


if __name__ == "__main__":
    main()
