import argparse
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


WRAPPER_COMPARE_ROWS = {
    "freq": 0,
    "left": 5,
    "right": 6,
}


def _compute_error_report(candidate, reference, freq_hz, top_k):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    if candidate.ndim != 2:
        raise ValueError("candidate and reference must be 2D arrays with shape (nfreq, batch)")

    floor = max(float(np.max(np.abs(reference))) * 1.0e-12, 1.0e-300)
    abs_err = np.abs(candidate - reference)
    denom = np.maximum(np.abs(reference), floor)
    rel_err = abs_err / denom

    cases = []
    if top_k > 0:
        flat_order = np.argsort(rel_err.ravel())[::-1]
        for flat_index in flat_order[:top_k]:
            freq_index, batch_index = np.unravel_index(flat_index, rel_err.shape)
            ref_value = float(reference[freq_index, batch_index])
            cand_value = float(candidate[freq_index, batch_index])
            cases.append(
                {
                    "batch_index": int(batch_index),
                    "freq_index": int(freq_index),
                    "freq_hz": float(freq_hz[freq_index]),
                    "cpu_value": ref_value,
                    "gpu_value": cand_value,
                    "abs_error": float(abs_err[freq_index, batch_index]),
                    "rel_error": float(rel_err[freq_index, batch_index]),
                    "above_floor": bool(abs(ref_value) > floor),
                }
            )

    return {
        "floor": floor,
        "reference_max_abs": float(np.max(np.abs(reference))),
        "max_abs_error": float(np.max(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "p999_rel_error": float(np.quantile(rel_err, 0.999)),
        "median_rel_error": float(np.median(rel_err)),
        "count_total": int(rel_err.size),
        "count_above_floor": int(np.count_nonzero(np.abs(reference) > floor)),
        "count_at_or_below_floor": int(np.count_nonzero(np.abs(reference) <= floor)),
        "cases": cases,
    }


def _build_single_reference_inputs(batch, index):
    lparms, rparms, parms, dummy, rl = GScodes._build_powerlaw_iso_single_inputs(batch, index)
    lparms[4] = 16
    lparms[5] = 1
    lparms[6] = 0
    lparms[7] = 1
    parms[5, 0] = 6.0
    return lparms, rparms, parms, dummy, rl


def _build_batch(workload, batch_size):
    return phase1_workloads.build_supported_workload(workload, batch_size)


def _run_single_reference_rl(libname, batch):
    mwfunc = GScodes.initGET_MW(libname)
    rl = np.zeros((7, batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    status = np.zeros(batch.batch_size, dtype=np.int32)
    for idx in range(batch.batch_size):
        lparms, rparms, parms, dummy, out = _build_single_reference_inputs(batch, idx)
        status[idx] = int(mwfunc(lparms, rparms, parms, dummy, dummy, dummy, out))
        rl[:, :, idx] = out
    return GScodes.BatchRunResult(status=status, rl=rl, native_freq_hz=batch.native_freq_hz)


def _run_native_wrapper(libname, batch, backend, precision):
    kernel = GScodes.run_powerlaw_iso_batch_native(
        libname,
        batch,
        backend=backend,
        precision=precision,
        npoints=16,
        q_on=True,
    )
    wrapper = GScodes.local_jk_to_single_voxel_rl(batch, kernel)
    return kernel, wrapper


def validate_cpu_reference(libname, batch_size, workload, top_k):
    libname = phase1_workloads.default_library_path(libname)
    batch = _build_batch(workload, batch_size)
    kernel_result, cpu_rl = _run_native_wrapper(libname, batch, backend="cpu", precision="fp64")
    reference_rl = _run_single_reference_rl(libname, batch)

    if not np.array_equal(kernel_result.status, reference_rl.status):
        raise AssertionError("CPU native status and single-call reference status arrays differ")

    metrics = {}
    for name, row_index in WRAPPER_COMPARE_ROWS.items():
        metrics[name] = _compute_error_report(
            cpu_rl.rl[row_index, :, :],
            reference_rl.rl[row_index, :, :],
            cpu_rl.native_freq_hz,
            top_k,
        )

    return {
        "library": libname,
        "batch_size": batch_size,
        "workload": workload,
        "metrics": metrics,
    }


def validate_cuda_against_cpu(libname, batch_size, precision, workload, top_k):
    libname = phase1_workloads.default_library_path(libname)
    if not GScodes.cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend is not available in the loaded library: {libname}. "
            "Build with CUDA=1 and run on a machine with accessible NVIDIA runtime support."
        )

    batch = _build_batch(workload, batch_size)
    cpu = GScodes.run_powerlaw_iso_batch_native(libname, batch, backend="cpu", precision="fp64", npoints=16, q_on=True)
    gpu = GScodes.run_powerlaw_iso_batch_native(libname, batch, backend="cuda", precision=precision, npoints=16, q_on=True)

    if not np.array_equal(cpu.status, gpu.status):
        raise AssertionError("CPU and CUDA status arrays differ")

    metrics = {}
    for name in ["jx", "kx", "jo", "ko"]:
        metrics[name] = _compute_error_report(
            getattr(gpu, name),
            getattr(cpu, name),
            cpu.native_freq_hz,
            top_k,
        )

    return {
        "library": libname,
        "batch_size": batch_size,
        "workload": workload,
        "precision": precision,
        "metrics": metrics,
    }


def validate_wrapper_against_reference(libname, batch_size, backend, precision, workload, top_k):
    libname = phase1_workloads.default_library_path(libname)
    if backend == "cuda" and not GScodes.cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend is not available in the loaded library: {libname}. "
            "Build with CUDA=1 and run on a machine with accessible NVIDIA runtime support."
        )

    batch = _build_batch(workload, batch_size)
    kernel, wrapper_rl = _run_native_wrapper(libname, batch, backend=backend, precision=precision)
    reference_rl = _run_single_reference_rl(libname, batch)

    if not np.array_equal(kernel.status, reference_rl.status):
        raise AssertionError("Native wrapper path and single-call reference status arrays differ")

    metrics = {}
    for name, row_index in WRAPPER_COMPARE_ROWS.items():
        metrics[name] = _compute_error_report(
            wrapper_rl.rl[row_index, :, :],
            reference_rl.rl[row_index, :, :],
            wrapper_rl.native_freq_hz,
            top_k,
        )

    return {
        "library": libname,
        "batch_size": batch_size,
        "workload": workload,
        "backend": backend,
        "precision": precision,
        "metrics": metrics,
    }


def _print_metric_report(name, metrics, print_cases):
    print(
        f"{name}: floor={metrics['floor']:.3e} ref_max={metrics['reference_max_abs']:.3e} "
        f"max_abs={metrics['max_abs_error']:.3e} max_rel={metrics['max_rel_error']:.3e} "
        f"p999_rel={metrics['p999_rel_error']:.3e} median_rel={metrics['median_rel_error']:.3e} "
        f"above_floor={metrics['count_above_floor']} at_or_below_floor={metrics['count_at_or_below_floor']}"
    )
    if print_cases:
        for case in metrics["cases"]:
            print(
                f"  case batch={case['batch_index']} freq={case['freq_index']} freq_hz={case['freq_hz']:.6e} "
                f"cpu={case['cpu_value']:.15e} gpu={case['gpu_value']:.15e} "
                f"abs_err={case['abs_error']:.3e} rel_err={case['rel_error']:.3e} "
                f"above_floor={case['above_floor']}"
            )


def _print_validation_result(header, result, report_names):
    print(header)
    print(f"library: {result['library']}")
    print(f"workload: {result['workload']}")
    print(f"batch_size: {result['batch_size']}")
    if "backend" in result:
        print(f"backend: {result['backend']}")
    if "precision" in result:
        print(f"precision: {result['precision']}")
    for name, metrics in result["metrics"].items():
        _print_metric_report(name, metrics, name in report_names)


def _select_batch_sizes(args):
    if args.suite_batch_sizes:
        return args.suite_batch_sizes
    return [args.batch_size]


def _run_target(args, batch_size):
    if args.target == "cpu-reference":
        return validate_cpu_reference(args.lib, batch_size, args.workload, args.top_k)
    if args.target == "wrapper-cpu-reference":
        return validate_wrapper_against_reference(args.lib, batch_size, "cpu", "fp64", args.workload, args.top_k)
    if args.target == "cuda-fp64":
        return validate_cuda_against_cpu(args.lib, batch_size, "fp64", args.workload, args.top_k)
    if args.target == "cuda-fp32":
        return validate_cuda_against_cpu(args.lib, batch_size, "fp32", args.workload, args.top_k)
    if args.target == "wrapper-cuda-fp64":
        return validate_wrapper_against_reference(args.lib, batch_size, "cuda", "fp64", args.workload, args.top_k)
    if args.target == "wrapper-cuda-fp32":
        return validate_wrapper_against_reference(args.lib, batch_size, "cuda", "fp32", args.workload, args.top_k)
    raise ValueError(f"Unsupported target {args.target!r}")


def main():
    parser = argparse.ArgumentParser(description="Validate the native approximate batch backend on the supported narrow path.")
    parser.add_argument("--lib", help="Path to the MWTransferArr shared library.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of spectra in the validation batch.")
    parser.add_argument(
        "--suite-batch-sizes",
        type=int,
        nargs="+",
        help="Run the selected validation target across multiple batch sizes.",
    )
    parser.add_argument(
        "--workload",
        choices=["real-sweep", "stress-sweep"],
        default="real-sweep",
        help="Supported workload builder to validate.",
    )
    parser.add_argument(
        "--target",
        choices=[
            "cpu-reference",
            "wrapper-cpu-reference",
            "cuda-fp64",
            "cuda-fp32",
            "wrapper-cuda-fp64",
            "wrapper-cuda-fp32",
        ],
        default="cpu-reference",
        help="Validation target to run.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many worst cases to retain per reported metric.")
    parser.add_argument(
        "--report-cases",
        nargs="+",
        default=[],
        help="Metric names whose top-k worst cases should be printed, for example jo or left.",
    )
    args = parser.parse_args()

    runtime_env.print_runtime_summary(args.lib)
    try:
        for batch_size in _select_batch_sizes(args):
            result = _run_target(args, batch_size)
            _print_validation_result("Phase 2 validation result", result, set(args.report_cases))
    except RuntimeError as exc:
        print(f"Phase 2 validation unavailable: {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
