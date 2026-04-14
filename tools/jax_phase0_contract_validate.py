import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = REPO_ROOT / "examples"
MCMC_DIR = REPO_ROOT / "mcmc_example"

for path in (REPO_ROOT, TOOLS_DIR, EXAMPLES_DIR, MCMC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import GScodes
import mcmc_backend_gpu_batched
import phase1_workloads
from jax_ffi import mw_approx_batch_contract as contract


def _max_abs_rel(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    abs_err = np.max(np.abs(candidate - reference))
    ref_scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    rel_err = abs_err / ref_scale
    return abs_err, rel_err


def _assert_allclose(name, candidate, reference, *, rtol=1.0e-12, atol=1.0e-12):
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape:
        raise AssertionError(f"{name}: shape mismatch {candidate.shape} vs {reference.shape}")
    if not np.allclose(candidate, reference, rtol=rtol, atol=atol, equal_nan=True):
        abs_err, rel_err = _max_abs_rel(candidate, reference)
        raise AssertionError(f"{name}: max_abs={abs_err:.3e} max_rel={rel_err:.3e} exceeds tolerance")
    abs_err, rel_err = _max_abs_rel(candidate, reference)
    print(f"{name}: ok (max_abs={abs_err:.3e}, max_rel={rel_err:.3e})")


def _build_blank_batch_like(reference_batch):
    return GScodes.build_wrapper_powerlaw_iso_batch(
        batch_size=reference_batch.batch_size,
        nfreq=reference_batch.nfreq,
        nu0_hz=reference_batch.nu0_hz,
        dlog10_nu=reference_batch.dlog10_nu,
        nu_cr_factor=getattr(reference_batch, "nu_cr_factor", 0.0),
        nu_wh_factor=getattr(reference_batch, "nu_wh_factor", 0.0),
    )


def _validate_parameter_contract(reference_batch):
    full_params = contract.full_parameter_batch_from_wrapper_batch(reference_batch)
    legacy_params = contract.legacy_reduced_parameter_batch_from_full(full_params)
    legacy_expected = full_params.copy()
    legacy_expected[:, 0] = contract.LEGACY_AREA_ASEC2 * (contract.ASEC2CM ** 2)
    legacy_expected[:, 9] = contract.LEGACY_EMAX_MEV

    batch_from_full = _build_blank_batch_like(reference_batch)
    contract.fill_analytical_powerlaw_iso_batch(batch_from_full, full_params)

    batch_from_legacy = _build_blank_batch_like(reference_batch)
    contract.fill_analytical_powerlaw_iso_batch(batch_from_legacy, legacy_params)

    for field_name in (
        "area_cm2",
        "depth_cm",
        "bmag_g",
        "temperature_k",
        "thermal_density_cm3",
        "nonthermal_density_cm3",
        "delta",
        "theta_deg",
        "emin_mev",
        "emax_mev",
    ):
        reference = np.asarray(getattr(reference_batch, field_name), dtype=np.float64)
        _assert_allclose(f"{field_name} from 10D", getattr(batch_from_full, field_name), reference)

    legacy_expected_batch = _build_blank_batch_like(reference_batch)
    contract.fill_analytical_powerlaw_iso_batch(legacy_expected_batch, legacy_expected)
    for field_name in (
        "area_cm2",
        "depth_cm",
        "bmag_g",
        "temperature_k",
        "thermal_density_cm3",
        "nonthermal_density_cm3",
        "delta",
        "theta_deg",
        "emin_mev",
        "emax_mev",
    ):
        _assert_allclose(
            f"{field_name} from legacy 8D",
            getattr(batch_from_legacy, field_name),
            getattr(legacy_expected_batch, field_name),
        )

    print(f"legacy_param_shape: {legacy_params.shape}")
    print(f"full_param_shape: {full_params.shape}")
    return full_params, legacy_params


def _resolve_library_or_skip(explicit_path):
    try:
        return phase1_workloads.default_library_path(explicit_path)
    except Exception as exc:
        print(f"runtime_validation_skipped: {type(exc).__name__}: {exc}")
        return None


def _validate_observable_contract(libname, reference_batch, *, backend, precision, npoints, q_on, spec_in_tb):
    result = GScodes.run_powerlaw_iso_batch_wrapper(
        libname,
        reference_batch,
        backend=backend,
        precision=precision,
        npoints=npoints,
        q_on=q_on,
    )
    candidate = contract.extract_legacy_spectrum_from_batch_result(
        reference_batch,
        result,
        spec_in_tb=spec_in_tb,
    )
    reference = mcmc_backend_gpu_batched.LegacyTbSpectrumExtractor(
        batch=reference_batch,
        spec_in_tb=spec_in_tb,
    )(result)

    _assert_allclose("observable adapter", candidate, reference)
    print(f"runtime_status_unique: {np.unique(np.asarray(result.status, dtype=np.int32)).tolist()}")
    print(f"postprocess_path: {getattr(result, 'postprocess_path', '')}")


def main():
    parser = argparse.ArgumentParser(description="Validate the staged JAX parameter/observable contract against the current supported path.")
    parser.add_argument("--lib", help="Explicit path to MWTransferArr shared library.")
    parser.add_argument("--workload", default="real-sweep", choices=["real-sweep", "stress-sweep"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--precision", default="fp64", choices=["fp64", "fp32"])
    parser.add_argument("--npoints", type=int, default=16)
    parser.add_argument("--q-off", action="store_true", help="Disable q_on for the wrapper call.")
    parser.add_argument("--spec-in-tb", action="store_true", default=True, help="Compare the legacy Tb observable.")
    parser.add_argument("--spec-in-flux", action="store_true", help="Compare flux instead of Tb.")
    parser.add_argument("--require-runtime", action="store_true", help="Exit nonzero if the shared library cannot be validated on this host.")
    args = parser.parse_args()

    spec_in_tb = not args.spec_in_flux

    reference_batch = phase1_workloads.build_supported_workload(args.workload, args.batch_size)
    _validate_parameter_contract(reference_batch)

    libname = _resolve_library_or_skip(args.lib)
    if libname is None:
        return 2 if args.require_runtime else 0

    _validate_observable_contract(
        libname,
        reference_batch,
        backend=args.backend,
        precision=args.precision,
        npoints=args.npoints,
        q_on=not args.q_off,
        spec_in_tb=spec_in_tb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
