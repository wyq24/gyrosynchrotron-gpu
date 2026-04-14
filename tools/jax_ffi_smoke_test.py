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
from jax_ffi.jax_runtime import device_label, maybe_device_put, resolve_jax_device

maybe_reexec_tool_main(
    "jax_ffi_smoke_test",
    script_path=__file__,
    repo_root=REPO_ROOT,
    extra_paths=(TOOLS_DIR, EXAMPLES_DIR),
)

import GScodes
import jax
import phase1_workloads
from jax_ffi import mw_approx_batch_contract as contract
from jax_ffi.mw_approx_batch_jax import mw_approx_batch_legacy_spectrum_jax, mw_approx_batch_rl_jax


def _assert_allclose(name, candidate, reference, *, rtol=1.0e-12, atol=1.0e-12):
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        raise AssertionError(f"{name}: shape mismatch {candidate.shape} vs {reference.shape}")
    if not np.allclose(candidate, reference, rtol=rtol, atol=atol, equal_nan=True):
        abs_err = float(np.max(np.abs(candidate - reference)))
        ref_scale = max(float(np.max(np.abs(reference))), 1.0e-300)
        rel_err = abs_err / ref_scale
        raise AssertionError(f"{name}: max_abs={abs_err:.3e} max_rel={rel_err:.3e} exceeds tolerance")
    abs_err = float(np.max(np.abs(candidate - reference)))
    ref_scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    rel_err = abs_err / ref_scale
    print(f"{name}: ok (max_abs={abs_err:.3e}, max_rel={rel_err:.3e})")


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


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the staged JAX FFI proof against the current FP64 wrapper path.")
    parser.add_argument("--lib", required=True, help="Host-compatible MWTransferArr shared library for the reference wrapper path.")
    parser.add_argument("--xla-lib", required=True, help="Host-compatible MWTransferArrXLA shared library for the JAX FFI path.")
    parser.add_argument("--workload", default="real-sweep", choices=["real-sweep", "stress-sweep"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference-backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--jax-device-platform",
        default="default",
        choices=["default", "cpu", "gpu", "cuda"],
        help="Place the JAX inputs on the first visible device matching this platform before the FFI call.",
    )
    args = parser.parse_args()

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("This smoke test requires JAX x64 mode, but enabling it at runtime did not succeed.")

    batch = phase1_workloads.build_supported_workload(args.workload, args.batch_size)
    full_params = contract.full_parameter_batch_from_wrapper_batch(batch)
    jax_device = resolve_jax_device(args.jax_device_platform)
    jax_full_params = maybe_device_put(full_params, jax_device)

    wrapper_result = GScodes.run_powerlaw_iso_batch_wrapper(
        args.lib,
        batch,
        backend=args.reference_backend,
        precision="fp64",
        npoints=16,
        q_on=True,
    )
    jax_rl, jax_freq_hz = mw_approx_batch_rl_jax(
        jax_full_params,
        lib_path=args.xla_lib,
        nfreq=batch.nfreq,
        nu0_hz=batch.nu0_hz,
        dlog10_nu=batch.dlog10_nu,
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
    )
    _assert_allclose("freq_hz", np.asarray(jax_freq_hz), wrapper_result.native_freq_hz)
    _assert_allclose(
        "rl",
        np.asarray(jax_rl),
        np.transpose(wrapper_result.rl, (2, 0, 1)),
        rtol=5.0e-12,
        atol=1.0e-10,
    )

    jax_tb, jax_target_freq_hz = mw_approx_batch_legacy_spectrum_jax(
        jax_full_params,
        lib_path=args.xla_lib,
        nfreq=batch.nfreq,
        nu0_hz=batch.nu0_hz,
        dlog10_nu=batch.dlog10_nu,
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
    )
    reference_tb = contract.extract_legacy_spectrum_from_batch_result(batch, wrapper_result, spec_in_tb=True)
    _assert_allclose("legacy_tb", np.asarray(jax_tb), reference_tb, rtol=5.0e-12, atol=1.0e-10)
    _assert_allclose("legacy_target_freq_hz", np.asarray(jax_target_freq_hz), contract.LEGACY_TARGET_FREQ_GHZ * 1.0e9)

    legacy_batch = _make_legacy_semantics_batch(batch, full_params)
    legacy_result = GScodes.run_powerlaw_iso_batch_wrapper(
        args.lib,
        legacy_batch,
        backend=args.reference_backend,
        precision="fp64",
        npoints=16,
        q_on=True,
    )
    legacy_params = contract.legacy_reduced_parameter_batch_from_full(full_params)
    jax_legacy_params = maybe_device_put(legacy_params, jax_device)
    legacy_tb_jax, _ = mw_approx_batch_legacy_spectrum_jax(
        jax_legacy_params,
        lib_path=args.xla_lib,
        nfreq=batch.nfreq,
        nu0_hz=batch.nu0_hz,
        dlog10_nu=batch.dlog10_nu,
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
    )
    legacy_tb_reference = contract.extract_legacy_spectrum_from_batch_result(legacy_batch, legacy_result, spec_in_tb=True)
    _assert_allclose("legacy8_tb", np.asarray(legacy_tb_jax), legacy_tb_reference, rtol=5.0e-12, atol=1.0e-10)

    print("postprocess_path:", wrapper_result.postprocess_path)
    print("reference_backend:", args.reference_backend)
    print("jax_default_backend:", jax.default_backend())
    print("jax_input_device:", device_label(jax_device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
