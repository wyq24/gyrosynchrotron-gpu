import argparse
import statistics
import time
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = REPO_ROOT / "examples"

for path in (REPO_ROOT, TOOLS_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jax_ffi.cli_reentry import maybe_reexec_tool_main
from jax_ffi.jax_runtime import device_label, maybe_device_put, resolve_jax_device

maybe_reexec_tool_main(
    "jax_phase1_benchmark",
    script_path=__file__,
    repo_root=REPO_ROOT,
    extra_paths=(TOOLS_DIR, EXAMPLES_DIR),
)

import GScodes
import jax
import phase1_workloads
from jax_ffi import mw_approx_batch_contract as contract
from jax_ffi.mw_approx_batch_jax import mw_approx_batch_legacy_spectrum_jax


def _timeit(fn, repeats):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _summarize(path_name, batch_size, samples, *, output_threads, compile_seconds=None, device=None):
    median_seconds = statistics.median(samples)
    row = {
        "path": path_name,
        "batch_size": batch_size,
        "median_seconds": median_seconds,
        "p95_seconds": float(np.percentile(samples, 95)),
        "spectra_per_second": batch_size / median_seconds,
        "output_threads": output_threads,
    }
    if compile_seconds is not None:
        row["compile_seconds"] = float(compile_seconds)
    if device is not None:
        row["device"] = device
    return row


def benchmark_reference_wrapper(lib_path, backend, repeats, batch_sizes, workload):
    rows = []
    for batch_size in batch_sizes:
        batch = phase1_workloads.build_supported_workload(workload, batch_size)

        def run_once():
            GScodes.run_powerlaw_iso_batch_wrapper(
                lib_path,
                batch,
                backend=backend,
                precision="fp64",
                npoints=16,
                q_on=True,
            )

        rows.append(
            _summarize(
                f"native-{backend}-integrated-wrapper-fp64",
                batch_size,
                _timeit(run_once, repeats),
                output_threads=batch_size * batch.nfreq * 2,
            )
        )
    return rows


def benchmark_jax_observable(xla_lib_path, jax_device, repeats, batch_sizes, workload):
    rows = []
    for batch_size in batch_sizes:
        batch = phase1_workloads.build_supported_workload(workload, batch_size)
        full_params = contract.full_parameter_batch_from_wrapper_batch(batch)
        params = maybe_device_put(full_params, jax_device)

        def forward_call(param_batch):
            spectra, _ = mw_approx_batch_legacy_spectrum_jax(
                param_batch,
                lib_path=xla_lib_path,
                nfreq=batch.nfreq,
                nu0_hz=batch.nu0_hz,
                dlog10_nu=batch.dlog10_nu,
                npoints=16,
                q_on=True,
                d_sun_au=1.0,
            )
            return spectra

        compiled = jax.jit(forward_call)

        t0 = time.perf_counter()
        compiled(params).block_until_ready()
        compile_seconds = time.perf_counter() - t0

        warm_samples = _timeit(lambda: compiled(params).block_until_ready(), repeats)
        rows.append(
            _summarize(
                f"jax-ffi-{device_label(jax_device)}-observable-fp64",
                batch_size,
                warm_samples,
                output_threads=batch_size * batch.nfreq * 2,
                compile_seconds=compile_seconds,
                device=device_label(jax_device),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark the staged JAX FFI observable path.")
    parser.add_argument("--lib", default=str(REPO_ROOT / "source" / "MWTransferArr.so"))
    parser.add_argument("--xla-lib", default=str(REPO_ROOT / "source" / "MWTransferArrXLA.so"))
    parser.add_argument("--workload", choices=["real-sweep", "stress-sweep"], default="real-sweep")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--reference-backend", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--jax-device-platform",
        default="gpu",
        choices=["default", "cpu", "gpu", "cuda"],
        help="Device placement for the JAX inputs and compiled custom call.",
    )
    args = parser.parse_args()

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX x64 mode is required for the FP64 benchmark path.")

    jax_device = resolve_jax_device(args.jax_device_platform)
    reference_rows = benchmark_reference_wrapper(args.lib, args.reference_backend, args.repeats, args.batch_sizes, args.workload)
    jax_rows = benchmark_jax_observable(args.xla_lib, jax_device, args.repeats, args.batch_sizes, args.workload)

    by_batch = {row["batch_size"]: row for row in reference_rows}
    print("JAX phase-1 benchmark summary")
    print(f"reference_backend: {args.reference_backend}")
    print(f"jax_default_backend: {jax.default_backend()}")
    print(f"jax_input_device: {device_label(jax_device)}")

    for row in jax_rows:
        baseline = by_batch[row["batch_size"]]
        speedup = baseline["median_seconds"] / row["median_seconds"]
        print(
            f"{row['path']} batch={row['batch_size']:4d}: "
            f"compile={row['compile_seconds']:.6f}s "
            f"warm_median={row['median_seconds']:.6f}s "
            f"warm_p95={row['p95_seconds']:.6f}s "
            f"spectra/s={row['spectra_per_second']:.2f} "
            f"speedup_vs_{baseline['path']}={speedup:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
