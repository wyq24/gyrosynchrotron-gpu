import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import GScodes
import phase1_workloads
import phase2_benchmark
import phase2_validate


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _plot_speedups(rows, output_path, title, path_names):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 4.5))
    for path_name, label in path_names:
        points = []
        for row in rows:
            if row["path"] != path_name:
                continue
            points.append((row["batch_size"], row["speedup_vs_legacy_wrapper"]))
        points.sort()
        if not points:
            continue
        batch_sizes = [item[0] for item in points]
        speedups = [item[1] for item in points]
        plt.plot(batch_sizes, speedups, marker="o", linewidth=2, label=label)
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlabel("Batch Size")
    plt.ylabel("Speedup vs Legacy Single-Call Wrapper")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_fp64_error_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{row['scope']}:{row['workload']}:{row['metric']}" for row in rows]
    max_rel = [row["max_rel_error"] for row in rows]
    p999_rel = [row["p999_rel_error"] for row in rows]

    y = np.arange(len(labels))
    plt.figure(figsize=(10.0, 6.5))
    plt.barh(y - 0.18, max_rel, height=0.32, label="Max Rel Error")
    plt.barh(y + 0.18, p999_rel, height=0.32, label="P999 Rel Error")
    plt.xscale("log")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Relative Error")
    plt.title("CPU vs CUDA FP64 Fidelity Summary")
    plt.grid(True, axis="x", which="both", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_overlay(output_path, title, freq_hz, cpu_left, cpu_right, gpu_left, gpu_right):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freq_ghz = np.asarray(freq_hz, dtype=np.float64) / 1.0e9
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    axes[0].plot(freq_ghz, cpu_left, linewidth=2, label="CPU Reference")
    axes[0].plot(freq_ghz, gpu_left, linestyle="--", linewidth=2, label="CUDA FP64")
    axes[0].set_title("Left")
    axes[0].set_xlabel("Frequency [GHz]")
    axes[0].set_ylabel("Flux")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].plot(freq_ghz, cpu_right, linewidth=2, label="CPU Reference")
    axes[1].plot(freq_ghz, gpu_right, linestyle="--", linewidth=2, label="CUDA FP64")
    axes[1].set_title("Right")
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _fidelity_rows(libname, batch_sizes):
    rows = []
    for workload in ["real-sweep", "stress-sweep"]:
        for batch_size in batch_sizes:
            local = phase2_validate.validate_cuda_against_cpu(libname, batch_size, "fp64", workload, 0)
            wrapper = phase2_validate.validate_wrapper_against_reference(libname, batch_size, "cuda", "fp64", workload, 0)
            for metric_name, metrics in local["metrics"].items():
                rows.append(
                    {
                        "scope": "local",
                        "workload": workload,
                        "batch_size": batch_size,
                        "metric": metric_name,
                        "max_abs_error": metrics["max_abs_error"],
                        "max_rel_error": metrics["max_rel_error"],
                        "p999_rel_error": metrics["p999_rel_error"],
                        "median_rel_error": metrics["median_rel_error"],
                    }
                )
            for metric_name, metrics in wrapper["metrics"].items():
                rows.append(
                    {
                        "scope": "wrapper",
                        "workload": workload,
                        "batch_size": batch_size,
                        "metric": metric_name,
                        "max_abs_error": metrics["max_abs_error"],
                        "max_rel_error": metrics["max_rel_error"],
                        "p999_rel_error": metrics["p999_rel_error"],
                        "median_rel_error": metrics["median_rel_error"],
                    }
                )
    return rows


def _worst_case_rows(rows):
    summary = {}
    for row in rows:
        key = (row["scope"], row["workload"], row["metric"])
        current = summary.get(key)
        if current is None or row["max_rel_error"] > current["max_rel_error"]:
            summary[key] = row
    return list(summary.values())


def _overlay_rows(libname, workload, batch_size, index):
    batch = phase1_workloads.build_supported_workload(workload, batch_size)
    reference = phase2_validate._run_single_reference_rl(libname, batch)
    _, gpu = phase2_validate._run_native_wrapper(libname, batch, "cuda", "fp64")
    return {
        "title": f"{workload} representative spectrum (batch={batch_size}, index={index})",
        "freq_hz": reference.native_freq_hz,
        "cpu_left": reference.rl[5, :, index],
        "cpu_right": reference.rl[6, :, index],
        "gpu_left": gpu.rl[5, :, index],
        "gpu_right": gpu.rl[6, :, index],
    }


def _coverage_matrix_text():
    return """# Validation Coverage Matrix

| Area | Status | Notes |
| --- | --- | --- |
| Python batch wrapper (`pyMW_Approx_Batch` + `local_jk_to_single_voxel_rl`) | Validated | Narrow analytical `PLW + ISO`, `Nz=1`, approximate GS only |
| Native CPU batch backend FP64 | Validated | Reference backend for the supported packed inputs |
| Native CUDA batch backend FP64 | Validated | Build, local `j/k`, wrapper RL, and benchmarked on the supported narrow path |
| Native CUDA batch backend FP32 | Evaluation only | Finite after local stability fixes, but still not accepted as a correctness mode |
| Legacy single-call CPU wrapper (`pyGET_MW`) | Preserved | Baseline path kept unchanged |
| Exact GS | Out of scope | Not validated here |
| Array DF | Out of scope | Not validated here |
| Transport | Out of scope | Not validated here |
| IDL | Out of scope | Not validated here |
| Broader API redesign | Out of scope | Not attempted here |
"""


def main():
    parser = argparse.ArgumentParser(description="Generate paper-grade figures and tables for the current validated narrow scope.")
    parser.add_argument("--lib", help="Path to the MWTransferArr shared library.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "phase2_paper"), help="Output directory.")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512], help="Batch sizes to include.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    libname = phase1_workloads.default_library_path(args.lib)

    benchmark_rows = phase2_benchmark.benchmark_compare_fp64(libname, args.repeats, args.batch_sizes, "real-sweep")
    speedup_rows = phase2_benchmark._speedup_rows(benchmark_rows)
    breakdown_rows = phase2_benchmark.benchmark_workflow_breakdown(libname, "cuda", "fp64", args.repeats, args.batch_sizes, "real-sweep")
    fidelity_rows = _fidelity_rows(libname, args.batch_sizes)
    fidelity_worst_rows = _worst_case_rows(fidelity_rows)

    _write_csv(
        output_dir / "benchmark_compare_fp64.csv",
        benchmark_rows,
        ["path", "batch_size", "median_seconds", "p95_seconds", "spectra_per_second", "output_threads"],
    )
    _write_csv(
        output_dir / "speedup_vs_legacy.csv",
        speedup_rows,
        ["batch_size", "baseline_path", "path", "speedup_vs_legacy_wrapper"],
    )
    _write_csv(
        output_dir / "wrapper_backend_breakdown_fp64.csv",
        breakdown_rows,
        [
            "path",
            "batch_size",
            "output_threads",
            "binding_seconds",
            "packing_seconds",
            "native_call_seconds",
            "call_boundary_seconds",
            "postprocess_seconds",
            "total_seconds",
            "native_total_seconds",
            "native_setup_seconds",
            "native_h2d_seconds",
            "native_device_alloc_seconds",
            "native_device_zero_seconds",
            "native_backend_compute_seconds",
            "native_sync_seconds",
            "native_d2h_seconds",
            "native_cleanup_seconds",
            "native_internal_overhead_seconds",
        ],
    )
    _write_csv(
        output_dir / "fp64_fidelity_summary.csv",
        fidelity_rows,
        ["scope", "workload", "batch_size", "metric", "max_abs_error", "max_rel_error", "p999_rel_error", "median_rel_error"],
    )
    _write_csv(
        output_dir / "fp64_fidelity_worstcase.csv",
        fidelity_worst_rows,
        ["scope", "workload", "batch_size", "metric", "max_abs_error", "max_rel_error", "p999_rel_error", "median_rel_error"],
    )
    _write_text(output_dir / "validation_coverage_matrix.md", _coverage_matrix_text())
    _write_text(
        output_dir / "fp32_status.txt",
        "FP32 remains evaluation-only after the local normalization fix. It is finite on the supported path but is not accepted as a correctness mode.",
    )

    _plot_speedups(
        speedup_rows,
        output_dir / "backend_speedup_vs_batch.png",
        "Backend-Level Speedup vs Batch Size",
        [
            ("native-cpu-backend-fp64", "CPU batch backend FP64"),
            ("native-cuda-backend-fp64", "CUDA batch backend FP64"),
        ],
    )
    _plot_speedups(
        speedup_rows,
        output_dir / "wrapper_speedup_vs_batch.png",
        "Wrapper-Level Speedup vs Batch Size",
        [
            ("native-cpu-wrapper-fp64", "CPU wrapper FP64"),
            ("native-cuda-wrapper-fp64", "CUDA wrapper FP64"),
        ],
    )
    _plot_fp64_error_summary(fidelity_worst_rows, output_dir / "fp64_error_summary.png")

    real_overlay = _overlay_rows(libname, "real-sweep", 8, 4)
    stress_overlay = _overlay_rows(libname, "stress-sweep", 8, 5)
    _plot_overlay(
        output_dir / "spectrum_overlay_real.png",
        real_overlay["title"],
        real_overlay["freq_hz"],
        real_overlay["cpu_left"],
        real_overlay["cpu_right"],
        real_overlay["gpu_left"],
        real_overlay["gpu_right"],
    )
    _plot_overlay(
        output_dir / "spectrum_overlay_stress.png",
        stress_overlay["title"],
        stress_overlay["freq_hz"],
        stress_overlay["cpu_left"],
        stress_overlay["cpu_right"],
        stress_overlay["gpu_left"],
        stress_overlay["gpu_right"],
    )

    print(f"Paper outputs written to {output_dir}")


if __name__ == "__main__":
    main()
