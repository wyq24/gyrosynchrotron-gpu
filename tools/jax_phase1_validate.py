import argparse
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
from jax_ffi.jax_runtime import device_label, maybe_device_put, normalize_platform_name, resolve_jax_device

maybe_reexec_tool_main(
    "jax_phase1_validate",
    script_path=__file__,
    repo_root=REPO_ROOT,
    extra_paths=(TOOLS_DIR, EXAMPLES_DIR),
)

import GScodes
import jax
import phase1_workloads
from jax_ffi import mw_approx_batch_contract as contract
from jax_ffi.mw_approx_batch_jax import mw_approx_batch_legacy_spectrum_jax, mw_approx_batch_rl_jax


STRICT_THRESHOLDS = {
    "name": "direct-reference",
    "rl_rtol": 5.0e-12,
    "rl_atol": 1.0e-10,
    "tb_rtol": 5.0e-12,
    "tb_atol": 1.0e-10,
}

# The current CUDA FFI handler routes through the existing validated native CUDA path.
# When comparing GPU JAX results directly against the authoritative CPU reference, the
# expected discrepancy floor is therefore the native CUDA-vs-CPU FP64 envelope rather
# than bitwise-or-near-bitwise agreement.
CPU_REFERENCE_CUDA_ENVELOPE_THRESHOLDS = {
    "name": "cpu-reference-cuda-envelope",
    "rl_rtol": 2.0e-9,
    "rl_atol": 2.0e-9,
    "tb_rtol": 2.0e-10,
    "tb_atol": 1.0e-2,
}


def _error_metrics(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise AssertionError(f"shape mismatch {candidate.shape} vs {reference.shape}")
    abs_err = np.abs(candidate - reference)
    ref_scale = np.maximum(np.abs(reference), 1.0e-300)
    rel_err = abs_err / ref_scale
    return {
        "max_abs": float(np.max(abs_err)),
        "max_rel": float(np.max(rel_err)),
    }


def _assert_threshold(name, metrics, *, rtol, atol):
    if metrics["max_abs"] > atol and metrics["max_rel"] > rtol:
        raise AssertionError(
            f"{name}: max_abs={metrics['max_abs']:.3e} max_rel={metrics['max_rel']:.3e} "
            f"exceeds atol={atol:.3e} rtol={rtol:.3e}"
        )


def _make_legacy_semantics_batch(reference_batch, full_params):
    batch = GScodes.build_wrapper_powerlaw_iso_batch(
        batch_size=reference_batch.batch_size,
        nfreq=reference_batch.nfreq,
        nu0_hz=reference_batch.nu0_hz,
        dlog10_nu=reference_batch.dlog10_nu,
        nu_cr_factor=getattr(reference_batch, "nu_cr_factor", 0.0),
        nu_wh_factor=getattr(reference_batch, "nu_wh_factor", 0.0),
    )
    contract.fill_analytical_powerlaw_iso_batch(batch, contract.legacy_reduced_parameter_batch_from_full(full_params))
    return batch


def _select_thresholds(*, reference_backend, jax_device):
    device_platform = normalize_platform_name(jax_device.platform if jax_device is not None else jax.default_backend())
    if reference_backend == "cpu" and device_platform == "cuda":
        return CPU_REFERENCE_CUDA_ENVELOPE_THRESHOLDS
    return STRICT_THRESHOLDS


def validate_case(*, lib_path, xla_lib_path, workload, batch_size, reference_backend, jax_device, thresholds):
    batch = phase1_workloads.build_supported_workload(workload, batch_size)
    full_params = contract.full_parameter_batch_from_wrapper_batch(batch)
    jax_full_params = maybe_device_put(full_params, jax_device)

    wrapper_result = GScodes.run_powerlaw_iso_batch_wrapper(
        lib_path,
        batch,
        backend=reference_backend,
        precision="fp64",
        npoints=16,
        q_on=True,
    )
    jax_rl, jax_freq_hz = mw_approx_batch_rl_jax(
        jax_full_params,
        lib_path=xla_lib_path,
        nfreq=batch.nfreq,
        nu0_hz=batch.nu0_hz,
        dlog10_nu=batch.dlog10_nu,
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
    )
    freq_metrics = _error_metrics(np.asarray(jax_freq_hz), wrapper_result.native_freq_hz)
    rl_metrics = _error_metrics(np.asarray(jax_rl), np.transpose(wrapper_result.rl, (2, 0, 1)))

    jax_tb, jax_target_freq_hz = mw_approx_batch_legacy_spectrum_jax(
        jax_full_params,
        lib_path=xla_lib_path,
        nfreq=batch.nfreq,
        nu0_hz=batch.nu0_hz,
        dlog10_nu=batch.dlog10_nu,
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
    )
    reference_tb = contract.extract_legacy_spectrum_from_batch_result(batch, wrapper_result, spec_in_tb=True)
    tb_metrics = _error_metrics(np.asarray(jax_tb), reference_tb)
    target_freq_metrics = _error_metrics(np.asarray(jax_target_freq_hz), contract.LEGACY_TARGET_FREQ_GHZ * 1.0e9)

    legacy_batch = _make_legacy_semantics_batch(batch, full_params)
    legacy_result = GScodes.run_powerlaw_iso_batch_wrapper(
        lib_path,
        legacy_batch,
        backend=reference_backend,
        precision="fp64",
        npoints=16,
        q_on=True,
    )
    legacy_params = contract.legacy_reduced_parameter_batch_from_full(full_params)
    jax_legacy_params = maybe_device_put(legacy_params, jax_device)
    legacy_tb_jax, _ = mw_approx_batch_legacy_spectrum_jax(
        jax_legacy_params,
        lib_path=xla_lib_path,
        nfreq=batch.nfreq,
        nu0_hz=batch.nu0_hz,
        dlog10_nu=batch.dlog10_nu,
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
    )
    legacy_tb_reference = contract.extract_legacy_spectrum_from_batch_result(legacy_batch, legacy_result, spec_in_tb=True)
    legacy8_tb_metrics = _error_metrics(np.asarray(legacy_tb_jax), legacy_tb_reference)

    _assert_threshold("freq_hz", freq_metrics, rtol=0.0, atol=0.0)
    _assert_threshold("rl", rl_metrics, rtol=thresholds["rl_rtol"], atol=thresholds["rl_atol"])
    _assert_threshold("legacy_tb", tb_metrics, rtol=thresholds["tb_rtol"], atol=thresholds["tb_atol"])
    _assert_threshold("legacy_target_freq_hz", target_freq_metrics, rtol=0.0, atol=0.0)
    _assert_threshold("legacy8_tb", legacy8_tb_metrics, rtol=thresholds["tb_rtol"], atol=thresholds["tb_atol"])

    return {
        "workload": workload,
        "batch_size": batch_size,
        "freq": freq_metrics,
        "rl": rl_metrics,
        "legacy_tb": tb_metrics,
        "legacy8_tb": legacy8_tb_metrics,
        "postprocess_path": wrapper_result.postprocess_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate the staged JAX FFI path across supported workloads and batch sizes.")
    parser.add_argument("--lib", default=str(REPO_ROOT / "source" / "MWTransferArr.so"))
    parser.add_argument("--xla-lib", default=str(REPO_ROOT / "source" / "MWTransferArrXLA.so"))
    parser.add_argument("--reference-backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--jax-device-platform",
        default="default",
        choices=["default", "cpu", "gpu", "cuda"],
        help="Place the JAX validation inputs on the first visible device matching this platform.",
    )
    parser.add_argument("--workloads", nargs="+", default=["real-sweep", "stress-sweep"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    args = parser.parse_args()

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX x64 mode is required for FP64 validation.")

    jax_device = resolve_jax_device(args.jax_device_platform)
    thresholds = _select_thresholds(reference_backend=args.reference_backend, jax_device=jax_device)
    rows = []
    worst_rl = {"max_abs": 0.0, "max_rel": 0.0, "case": None}
    worst_tb = {"max_abs": 0.0, "max_rel": 0.0, "case": None}
    worst_legacy8_tb = {"max_abs": 0.0, "max_rel": 0.0, "case": None}

    for workload in args.workloads:
        for batch_size in args.batch_sizes:
            row = validate_case(
                lib_path=args.lib,
                xla_lib_path=args.xla_lib,
                workload=workload,
                batch_size=batch_size,
                reference_backend=args.reference_backend,
                jax_device=jax_device,
                thresholds=thresholds,
            )
            rows.append(row)
            if row["rl"]["max_rel"] >= worst_rl["max_rel"]:
                worst_rl = {**row["rl"], "case": (workload, batch_size)}
            if row["legacy_tb"]["max_rel"] >= worst_tb["max_rel"]:
                worst_tb = {**row["legacy_tb"], "case": (workload, batch_size)}
            if row["legacy8_tb"]["max_rel"] >= worst_legacy8_tb["max_rel"]:
                worst_legacy8_tb = {**row["legacy8_tb"], "case": (workload, batch_size)}

            print(
                f"validated workload={workload:11s} batch={batch_size:4d}: "
                f"rl_max_abs={row['rl']['max_abs']:.3e} rl_max_rel={row['rl']['max_rel']:.3e} "
                f"tb_max_abs={row['legacy_tb']['max_abs']:.3e} tb_max_rel={row['legacy_tb']['max_rel']:.3e} "
                f"legacy8_tb_max_rel={row['legacy8_tb']['max_rel']:.3e} "
                f"postprocess_path={row['postprocess_path']}"
            )

    print("JAX phase-1 validation summary")
    print(f"reference_backend: {args.reference_backend}")
    print(f"validation_profile: {thresholds['name']}")
    print(f"jax_default_backend: {jax.default_backend()}")
    print(f"jax_input_device: {device_label(jax_device)}")
    print(
        f"worst_rl: workload={worst_rl['case'][0]} batch={worst_rl['case'][1]} "
        f"max_abs={worst_rl['max_abs']:.3e} max_rel={worst_rl['max_rel']:.3e}"
    )
    print(
        f"worst_legacy_tb: workload={worst_tb['case'][0]} batch={worst_tb['case'][1]} "
        f"max_abs={worst_tb['max_abs']:.3e} max_rel={worst_tb['max_rel']:.3e}"
    )
    print(
        f"worst_legacy8_tb: workload={worst_legacy8_tb['case'][0]} batch={worst_legacy8_tb['case'][1]} "
        f"max_abs={worst_legacy8_tb['max_abs']:.3e} max_rel={worst_legacy8_tb['max_rel']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
