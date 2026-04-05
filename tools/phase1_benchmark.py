import argparse
import statistics
import time
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import GScodes
import phase1_workloads
import runtime_env


def _timeit(fn, repeats):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def benchmark_real_single(libname, repeats):
    libname = phase1_workloads.default_library_path(libname)
    batch = phase1_workloads.build_real_wrapper_workload(batch_size=1)
    mwfunc = GScodes.initGET_MW(libname)
    lparms, rparms, parms, dummy, out = GScodes._build_powerlaw_iso_single_inputs(batch, 0)

    def run_once():
        out.fill(0.0)
        return mwfunc(lparms, rparms, parms, dummy, dummy, dummy, out)

    samples = _timeit(run_once, repeats)
    return {
        "scenario": "W1-small-real",
        "repeats": repeats,
        "median_seconds": statistics.median(samples),
        "p95_seconds": float(np.percentile(samples, 95)),
        "calls_per_second": 1.0 / statistics.median(samples),
    }


def benchmark_real_batch(libname, repeats, batch_sizes):
    libname = phase1_workloads.default_library_path(libname)
    rows = []
    for batch_size in batch_sizes:
        batch = phase1_workloads.build_real_wrapper_sweep(batch_size)

        def run_once():
            GScodes.run_powerlaw_iso_batch_cpu(libname, batch)

        samples = _timeit(run_once, repeats)
        median_seconds = statistics.median(samples)
        rows.append(
            {
                "scenario": "W2-batched-real",
                "batch_size": batch_size,
                "repeats": repeats,
                "median_seconds": median_seconds,
                "p95_seconds": float(np.percentile(samples, 95)),
                "spectra_per_second": batch_size / median_seconds,
                "output_threads": batch_size * batch.nfreq * 2,
            }
        )
    return rows


def benchmark_array_secondary(libname, repeats):
    libname = phase1_workloads.default_library_path(libname)
    mwfunc = GScodes.initGET_MW(libname)
    baseline = phase1_workloads.build_repo_array_baseline()

    def run_once():
        baseline["rl"].fill(0.0)
        return mwfunc(
            baseline["lparms"],
            baseline["rparms"],
            baseline["parms"],
            baseline["e_arr"],
            baseline["mu_arr"],
            baseline["f_arr"],
            baseline["rl"],
        )

    samples = _timeit(run_once, repeats)
    return {
        "scenario": "W3-secondary-array",
        "repeats": repeats,
        "median_seconds": statistics.median(samples),
        "p95_seconds": float(np.percentile(samples, 95)),
        "calls_per_second": 1.0 / statistics.median(samples),
        "above_nu_cr_fraction": baseline["above_nu_cr_fraction"],
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Phase 1 workloads and CPU reference batch path.")
    parser.add_argument("--lib", help="Path to the MWTransferArr shared library.")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing repeats per workload.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 256],
        help="Batch sizes for W2-batched-real.",
    )
    args = parser.parse_args()

    runtime_env.print_runtime_summary(args.lib)
    real_single = benchmark_real_single(args.lib, args.repeats)
    real_batch = benchmark_real_batch(args.lib, args.repeats, args.batch_sizes)
    array_secondary = benchmark_array_secondary(args.lib, max(1, min(args.repeats, 3)))

    print("Phase 1 benchmark summary")
    print(
        f"W1-small-real: median={real_single['median_seconds']:.6f}s "
        f"p95={real_single['p95_seconds']:.6f}s calls/s={real_single['calls_per_second']:.2f}"
    )
    for row in real_batch:
        print(
            f"W2-batched-real batch={row['batch_size']:4d}: median={row['median_seconds']:.6f}s "
            f"p95={row['p95_seconds']:.6f}s spectra/s={row['spectra_per_second']:.2f} "
            f"output_threads={row['output_threads']}"
        )
    print(
        f"W3-secondary-array: median={array_secondary['median_seconds']:.6f}s "
        f"p95={array_secondary['p95_seconds']:.6f}s calls/s={array_secondary['calls_per_second']:.2f} "
        f"above_nu_cr_fraction={array_secondary['above_nu_cr_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
