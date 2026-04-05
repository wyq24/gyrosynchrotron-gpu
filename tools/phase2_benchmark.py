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
import phase2_validate
import runtime_env


def _timeit(fn, repeats):
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _build_batch(workload, batch_size):
    return phase1_workloads.build_supported_workload(workload, batch_size)


def _summarize_samples(path_name, batch_size, samples, output_threads):
    median_seconds = statistics.median(samples)
    return {
        "path": path_name,
        "batch_size": batch_size,
        "median_seconds": median_seconds,
        "p95_seconds": float(np.percentile(samples, 95)),
        "spectra_per_second": batch_size / median_seconds,
        "output_threads": output_threads,
    }


def _median_profile_rows(path_name, batch_size, profiles, output_threads):
    keys = [
        "binding_seconds",
        "packing_seconds",
        "native_call_seconds",
        "call_boundary_seconds",
        "postprocess_seconds",
        "total_seconds",
    ]
    row = {
        "path": path_name,
        "batch_size": batch_size,
        "output_threads": output_threads,
    }
    for key in keys:
        row[key] = statistics.median(getattr(profile, key) for profile in profiles)

    native_keys = [
        "total_seconds",
        "setup_seconds",
        "h2d_seconds",
        "device_alloc_seconds",
        "device_zero_seconds",
        "backend_compute_seconds",
        "sync_seconds",
        "d2h_seconds",
        "cleanup_seconds",
        "internal_overhead_seconds",
    ]
    for key in native_keys:
        row[f"native_{key}"] = statistics.median(getattr(profile.native_timing, key) for profile in profiles)
    return row


def benchmark_native_backend(libname, backend, precision, repeats, batch_sizes, workload):
    libname = phase1_workloads.default_library_path(libname)
    if backend == "cuda" and not GScodes.cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend is not available in the loaded library: {libname}. "
            "Build with CUDA=1 and run on a machine with accessible NVIDIA runtime support."
        )

    rows = []
    for batch_size in batch_sizes:
        batch = _build_batch(workload, batch_size)

        def run_once():
            GScodes.run_powerlaw_iso_batch_native(
                libname,
                batch,
                backend=backend,
                precision=precision,
                npoints=16,
                q_on=True,
            )

        rows.append(
            _summarize_samples(
                f"native-{backend}-backend-{precision}",
                batch_size,
                _timeit(run_once, repeats),
                batch_size * batch.nfreq * 2,
            )
        )
    return rows


def benchmark_native_wrapper(libname, backend, precision, repeats, batch_sizes, workload):
    libname = phase1_workloads.default_library_path(libname)
    if backend == "cuda" and not GScodes.cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend is not available in the loaded library: {libname}. "
            "Build with CUDA=1 and run on a machine with accessible NVIDIA runtime support."
        )

    rows = []
    for batch_size in batch_sizes:
        batch = _build_batch(workload, batch_size)

        def run_once():
            kernel = GScodes.run_powerlaw_iso_batch_native(
                libname,
                batch,
                backend=backend,
                precision=precision,
                npoints=16,
                q_on=True,
            )
            GScodes.local_jk_to_single_voxel_rl(batch, kernel)

        rows.append(
            _summarize_samples(
                f"native-{backend}-wrapper-{precision}",
                batch_size,
                _timeit(run_once, repeats),
                batch_size * batch.nfreq * 2,
            )
        )
    return rows


def benchmark_native_integrated_wrapper(libname, backend, precision, repeats, batch_sizes, workload):
    libname = phase1_workloads.default_library_path(libname)
    if backend == "cuda" and not GScodes.cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend is not available in the loaded library: {libname}. "
            "Build with CUDA=1 and run on a machine with accessible NVIDIA runtime support."
        )

    rows = []
    for batch_size in batch_sizes:
        batch = _build_batch(workload, batch_size)

        def run_once():
            GScodes.run_powerlaw_iso_batch_wrapper(
                libname,
                batch,
                backend=backend,
                precision=precision,
                npoints=16,
                q_on=True,
            )

        rows.append(
            _summarize_samples(
                f"native-{backend}-integrated-wrapper-{precision}",
                batch_size,
                _timeit(run_once, repeats),
                batch_size * batch.nfreq * 2,
            )
        )
    return rows


def benchmark_legacy_wrapper(libname, repeats, batch_sizes, workload):
    libname = phase1_workloads.default_library_path(libname)
    rows = []
    for batch_size in batch_sizes:
        batch = _build_batch(workload, batch_size)
        mwfunc = GScodes.initGET_MW(libname)
        call_inputs = [phase2_validate._build_single_reference_inputs(batch, idx) for idx in range(batch.batch_size)]

        def run_once():
            for lparms, rparms, parms, dummy, out in call_inputs:
                out.fill(0.0)
                mwfunc(lparms, rparms, parms, dummy, dummy, dummy, out)

        rows.append(
            _summarize_samples(
                "legacy-singlecall-wrapper-fp64",
                batch_size,
                _timeit(run_once, repeats),
                batch_size * batch.nfreq * 2,
            )
        )
    return rows


def benchmark_compare_fp64(libname, repeats, batch_sizes, workload):
    rows = []
    rows.extend(benchmark_legacy_wrapper(libname, repeats, batch_sizes, workload))
    rows.extend(benchmark_native_backend(libname, "cpu", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_backend(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_wrapper(libname, "cpu", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_wrapper(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_integrated_wrapper(libname, "cpu", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_integrated_wrapper(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    return rows


def benchmark_compare_fp32(libname, repeats, batch_sizes, workload):
    rows = []
    rows.extend(benchmark_legacy_wrapper(libname, repeats, batch_sizes, workload))
    rows.extend(benchmark_native_backend(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_backend(libname, "cuda", "fp32", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_wrapper(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_wrapper(libname, "cuda", "fp32", repeats, batch_sizes, workload))
    return rows


def benchmark_compare_integrated_fp64(libname, repeats, batch_sizes, workload):
    rows = []
    rows.extend(benchmark_legacy_wrapper(libname, repeats, batch_sizes, workload))
    rows.extend(benchmark_native_backend(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_wrapper(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    rows.extend(benchmark_native_integrated_wrapper(libname, "cuda", "fp64", repeats, batch_sizes, workload))
    return rows


def benchmark_workflow_breakdown(libname, backend, precision, repeats, batch_sizes, workload):
    libname = phase1_workloads.default_library_path(libname)
    if backend == "cuda" and not GScodes.cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend is not available in the loaded library: {libname}. "
            "Build with CUDA=1 and run on a machine with accessible NVIDIA runtime support."
        )

    rows = []
    for batch_size in batch_sizes:
        batch = _build_batch(workload, batch_size)
        backend_profiles = []
        wrapper_profiles = []
        for _ in range(repeats):
            _, _, backend_profile = GScodes.run_powerlaw_iso_batch_native_profiled(
                libname,
                batch,
                backend=backend,
                precision=precision,
                npoints=16,
                q_on=True,
                include_postprocess=False,
            )
            backend_profiles.append(backend_profile)
            _, _, wrapper_profile = GScodes.run_powerlaw_iso_batch_native_profiled(
                libname,
                batch,
                backend=backend,
                precision=precision,
                npoints=16,
                q_on=True,
                include_postprocess=True,
            )
            wrapper_profiles.append(wrapper_profile)

        rows.append(
            _median_profile_rows(
                f"native-{backend}-backend-breakdown-{precision}",
                batch_size,
                backend_profiles,
                batch_size * batch.nfreq * 2,
            )
        )
        rows.append(
            _median_profile_rows(
                f"native-{backend}-wrapper-breakdown-{precision}",
                batch_size,
                wrapper_profiles,
                batch_size * batch.nfreq * 2,
            )
        )
    return rows


def _speedup_rows(rows):
    by_batch = {}
    for row in rows:
        by_batch.setdefault(row["batch_size"], {})[row["path"]] = row

    output = []
    for batch_size in sorted(by_batch):
        group = by_batch[batch_size]
        baseline = group.get("legacy-singlecall-wrapper-fp64")
        if baseline is None:
            continue
        for path, row in sorted(group.items()):
            if path == baseline["path"]:
                continue
            output.append(
                {
                    "batch_size": batch_size,
                    "baseline_path": baseline["path"],
                    "path": path,
                    "speedup_vs_legacy_wrapper": baseline["median_seconds"] / row["median_seconds"],
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark the supported narrow native batch path.")
    parser.add_argument("--lib", help="Path to the MWTransferArr shared library.")
    parser.add_argument(
        "--mode",
        choices=["native-backend", "native-wrapper", "compare-fp64", "compare-fp32", "compare-integrated-fp64", "compare-fused-fp64", "breakdown"],
        default="native-backend",
        help="Benchmark a single native backend or run the full FP64 comparison suite.",
    )
    parser.add_argument(
        "--workload",
        choices=["real-sweep", "stress-sweep"],
        default="real-sweep",
        help="Supported workload builder to benchmark.",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Native batch backend to benchmark in native-backend mode.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp64", "fp32"],
        default="fp64",
        help="Precision mode for native-backend mode.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing repeats per batch size.")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128, 256, 1024],
        help="Batch sizes to benchmark.",
    )
    args = parser.parse_args()

    runtime_env.print_runtime_summary(args.lib)
    try:
        if args.mode == "compare-fp64":
            rows = benchmark_compare_fp64(args.lib, args.repeats, args.batch_sizes, args.workload)
            print("Phase 2 FP64 benchmark comparison")
            for row in rows:
                print(
                    f"{row['path']} batch={row['batch_size']:4d}: "
                    f"median={row['median_seconds']:.6f}s "
                    f"p95={row['p95_seconds']:.6f}s "
                    f"spectra/s={row['spectra_per_second']:.2f} "
                    f"output_threads={row['output_threads']}"
                )
            for row in _speedup_rows(rows):
                print(
                    f"speedup batch={row['batch_size']:4d} {row['path']} vs {row['baseline_path']}: "
                    f"{row['speedup_vs_legacy_wrapper']:.2f}x"
                )
            return
        if args.mode in {"compare-integrated-fp64", "compare-fused-fp64"}:
            rows = benchmark_compare_integrated_fp64(args.lib, args.repeats, args.batch_sizes, args.workload)
            print("Phase 2 integrated wrapper FP64 benchmark comparison")
            for row in rows:
                print(
                    f"{row['path']} batch={row['batch_size']:4d}: "
                    f"median={row['median_seconds']:.6f}s "
                    f"p95={row['p95_seconds']:.6f}s "
                    f"spectra/s={row['spectra_per_second']:.2f} "
                    f"output_threads={row['output_threads']}"
                )
            for row in _speedup_rows(rows):
                print(
                    f"speedup batch={row['batch_size']:4d} {row['path']} vs {row['baseline_path']}: "
                    f"{row['speedup_vs_legacy_wrapper']:.2f}x"
                )
            return
        if args.mode == "compare-fp32":
            rows = benchmark_compare_fp32(args.lib, args.repeats, args.batch_sizes, args.workload)
            print("Phase 2 FP32 benchmark comparison")
            for row in rows:
                print(
                    f"{row['path']} batch={row['batch_size']:4d}: "
                    f"median={row['median_seconds']:.6f}s "
                    f"p95={row['p95_seconds']:.6f}s "
                    f"spectra/s={row['spectra_per_second']:.2f} "
                    f"output_threads={row['output_threads']}"
                )
            for row in _speedup_rows(rows):
                print(
                    f"speedup batch={row['batch_size']:4d} {row['path']} vs {row['baseline_path']}: "
                    f"{row['speedup_vs_legacy_wrapper']:.2f}x"
                )
            by_batch = {}
            for row in rows:
                by_batch.setdefault(row["batch_size"], {})[row["path"]] = row
            for batch_size in sorted(by_batch):
                group = by_batch[batch_size]
                for kind in ["backend", "wrapper"]:
                    fp64 = group.get(f"native-cuda-{kind}-fp64")
                    fp32 = group.get(f"native-cuda-{kind}-fp32")
                    if fp64 and fp32:
                        print(
                            f"speedup batch={batch_size:4d} native-cuda-{kind}-fp32 vs native-cuda-{kind}-fp64: "
                            f"{fp64['median_seconds'] / fp32['median_seconds']:.2f}x"
                        )
            return
        if args.mode == "breakdown":
            rows = benchmark_workflow_breakdown(args.lib, args.backend, args.precision, args.repeats, args.batch_sizes, args.workload)
            print("Phase 2 workflow breakdown")
            for row in rows:
                print(
                    f"{row['path']} batch={row['batch_size']:4d}: "
                    f"total={row['total_seconds']:.6f}s "
                    f"binding={row['binding_seconds']:.6f}s "
                    f"packing={row['packing_seconds']:.6f}s "
                    f"native_call={row['native_call_seconds']:.6f}s "
                    f"call_boundary={row['call_boundary_seconds']:.6f}s "
                    f"postprocess={row['postprocess_seconds']:.6f}s "
                    f"native_total={row['native_total_seconds']:.6f}s "
                    f"setup={row['native_setup_seconds']:.6f}s "
                    f"h2d={row['native_h2d_seconds']:.6f}s "
                    f"alloc={row['native_device_alloc_seconds']:.6f}s "
                    f"zero={row['native_device_zero_seconds']:.6f}s "
                    f"compute={row['native_backend_compute_seconds']:.6f}s "
                    f"sync={row['native_sync_seconds']:.6f}s "
                    f"d2h={row['native_d2h_seconds']:.6f}s "
                    f"cleanup={row['native_cleanup_seconds']:.6f}s "
                    f"native_internal_overhead={row['native_internal_overhead_seconds']:.6f}s"
                )
            return

        if args.mode == "native-wrapper":
            rows = benchmark_native_wrapper(args.lib, args.backend, args.precision, args.repeats, args.batch_sizes, args.workload)
        else:
            rows = benchmark_native_backend(args.lib, args.backend, args.precision, args.repeats, args.batch_sizes, args.workload)
    except RuntimeError as exc:
        print(f"Phase 2 benchmark unavailable: {exc}")
        raise SystemExit(2)

    print("Phase 2 native batch benchmark summary")
    for row in rows:
        print(
            f"{row['path']} batch={row['batch_size']:4d}: "
            f"median={row['median_seconds']:.6f}s "
            f"p95={row['p95_seconds']:.6f}s "
            f"spectra/s={row['spectra_per_second']:.2f} "
            f"output_threads={row['output_threads']}"
        )


if __name__ == "__main__":
    main()
