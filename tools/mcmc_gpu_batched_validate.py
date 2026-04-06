import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache-codex")

from examples import GScodes
from mcmc_example.mcmc_backend_gpu_batched import (
    CubeSamplingConfig,
    LEGACY_AREA_ASEC2,
    LEGACY_EMAX_MEV,
    LEGACY_TARGET_FREQ_GHZ,
    SamplingConfig,
    WarmStartConfig,
    build_legacy_8d_batched_backend,
    fit_cube_mcmc_resumable_gpu,
)
from mcmc_example.spec_utils import simulate_spectrum_optimized


TEST_PARAMS_8D = np.array([
    [5.0, 2.0, 1.0, 10.0, 5.0, 4.0, 45.0, 10.0],
    [6.5, 1.6, 1.3, 10.2, 5.3, 3.8, 52.0, 14.0],
    [4.5, 2.4, 0.9, 9.8, 4.8, 4.6, 40.0, 8.0],
    [7.0, 1.2, 1.7, 10.4, 5.6, 3.5, 60.0, 18.0],
], dtype=np.float64)

ALL_PARAM_BOUNDS_8D = [
    (1.0, 20.0),
    (0.1, 5.0),
    (0.1, 30.0),
    (8.0, 12.0),
    (3.0, 8.0),
    (2.0, 7.0),
    (10.0, 80.0),
    (5.0, 50.0),
]


def _error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    abs_err = np.abs(cand - ref)
    floor = max(float(np.max(np.abs(ref))) * 1.0e-12, 1.0e-300)
    rel_err = abs_err / np.maximum(np.abs(ref), floor)
    return {
        "max_abs": float(np.max(abs_err)),
        "max_rel": float(np.max(rel_err)),
        "median_rel": float(np.median(rel_err)),
    }


def _flux_sfu_to_tb_numeric(target_freq_hz: np.ndarray, flux_sfu: np.ndarray, area_asec2: float) -> np.ndarray:
    solid_angle_sr = float(area_asec2) * (np.pi / (180.0 * 3600.0)) ** 2
    factor = (2.99792458e10 ** 2) / (2.0 * 1.380649e-16 * np.square(target_freq_hz) * solid_angle_sr)
    return np.asarray(flux_sfu, dtype=np.float64) * 1.0e-19 * factor


def _run_validated_single_call_reference(params_batch: np.ndarray, lib_path: str) -> np.ndarray:
    mwfunc = GScodes.initGET_MW(lib_path)
    target_freq_hz = LEGACY_TARGET_FREQ_GHZ * 1.0e9
    out = []

    for row in np.asarray(params_batch, dtype=np.float64):
        batch = GScodes.build_wrapper_powerlaw_iso_batch(
            batch_size=1,
            area_asec2=LEGACY_AREA_ASEC2,
            depth_asec=row[0],
            bmag_100g=row[1],
            temperature_mk=row[2],
            log_nth=row[3],
            log_nnth=row[4],
            delta=row[5],
            theta_deg=row[6],
            emin_kev=row[7],
            emax_mev=LEGACY_EMAX_MEV,
        )
        lparms, rparms, parms, dummy, rl = GScodes._build_powerlaw_iso_single_inputs(batch, 0)
        lparms[4] = 16
        lparms[5] = 1
        lparms[6] = 0
        lparms[7] = 1
        parms[5, 0] = 6.0
        status = int(mwfunc(lparms, rparms, parms, dummy, dummy, dummy, rl))
        total_flux = rl[5] + rl[6]
        if status != 0 or not np.any(total_flux):
            out.append(np.full(target_freq_hz.size, 1.0e4, dtype=np.float64))
            continue

        interp_flux = np.power(
            10.0,
            np.interp(
                np.log10(target_freq_hz),
                np.log10(rl[0] * 1.0e9),
                np.log10(np.maximum(total_flux, 1.0e-300)),
            ),
        )
        out.append(_flux_sfu_to_tb_numeric(target_freq_hz, interp_flux, LEGACY_AREA_ASEC2))
    return np.stack(out, axis=0)


def validate_spectra(lib_path: str, backend_name: str) -> None:
    backend = build_legacy_8d_batched_backend(
        lib_path=lib_path,
        batch_capacity=TEST_PARAMS_8D.shape[0],
        backend=backend_name,
    )
    legacy = np.stack(
        [simulate_spectrum_optimized(row, libname=lib_path) for row in TEST_PARAMS_8D],
        axis=0,
    )
    approx_reference = _run_validated_single_call_reference(TEST_PARAMS_8D, lib_path)
    batched = backend.simulate_batch(TEST_PARAMS_8D)
    approx_metrics = _error_metrics(batched, approx_reference)
    legacy_metrics = _error_metrics(batched, legacy)

    print("Spectrum validation")
    print(f"backend: {backend_name}")
    print(f"batch_size: {TEST_PARAMS_8D.shape[0]}")
    print(f"freq_count: {batched.shape[1]}")
    print("validated_single_call_reference")
    print(f"  max_abs_error: {approx_metrics['max_abs']:.6e}")
    print(f"  max_rel_error: {approx_metrics['max_rel']:.6e}")
    print(f"  median_rel_error: {approx_metrics['median_rel']:.6e}")
    print("current_legacy_simulator")
    print(f"  max_abs_error: {legacy_metrics['max_abs']:.6e}")
    print(f"  max_rel_error: {legacy_metrics['max_rel']:.6e}")
    print(f"  median_rel_error: {legacy_metrics['median_rel']:.6e}")

    if approx_metrics["max_rel"] > 1.0e-9:
        raise RuntimeError("Validated single-call reference and new batched backend disagree beyond tolerance")


def validate_tiny_cube(lib_path: str, backend_name: str) -> None:
    if importlib.util.find_spec("emcee") is None:
        raise RuntimeError("Cube mode requires the 'emcee' package to be installed in this environment")

    backend = build_legacy_8d_batched_backend(
        lib_path=lib_path,
        batch_capacity=16,
        backend=backend_name,
    )
    cube = backend.simulate_batch(TEST_PARAMS_8D).reshape(2, 2, -1)

    cube_cfg = CubeSamplingConfig(
        sampling=SamplingConfig(
            n_walkers=16,
            n_steps=12,
            burn_in=4,
            thin=2,
            noise_level=0.1,
        ),
        warm_start=WarmStartConfig(
            use_neighbor_samples=True,
            jitter_std_norm=0.03,
            exploration_fraction=0.25,
            max_neighbor_samples=512,
            broad_init_scale=0.25,
        ),
        checkpoint_every=1,
        max_nodes=4,
        save_samples=False,
    )

    with tempfile.TemporaryDirectory(prefix="mcmc_gpu_batched_validate_") as tmpdir:
        result = fit_cube_mcmc_resumable_gpu(
            cube=cube,
            all_param_bounds=ALL_PARAM_BOUNDS_8D,
            vary_indices=list(range(8)),
            fixed_params=[0.0] * 8,
            x_log_bounds=(4.0, 9.0),
            segmentation="pixel",
            block_k=1,
            valid_mask=None,
            out_dir=tmpdir,
            resume_path=str(Path(tmpdir) / "resume.npz"),
            cube_cfg=cube_cfg,
            forward_backend=backend,
            seed=12345,
        )

    print("Tiny cube smoke test")
    print(f"backend: {backend_name}")
    print(f"cube_shape: {cube.shape}")
    print(f"done_nodes: {int(np.count_nonzero(result.done_nodes))}/{result.done_nodes.size}")
    print(f"theta_map_shape: {result.theta_map.shape}")
    print(f"processed_nodes: {result.debug['processed_nodes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the GPU-batched MCMC forward adapter.")
    parser.add_argument("--lib", default="source/MWTransferArr.so", help="Path to MWTransferArr shared library")
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda"], help="Native backend")
    parser.add_argument("--mode", default="all", choices=["spectra", "cube", "all"], help="Validation mode")
    args = parser.parse_args()

    if args.mode in {"spectra", "all"}:
        validate_spectra(args.lib, args.backend)
    if args.mode in {"cube", "all"}:
        validate_tiny_cube(args.lib, args.backend)


if __name__ == "__main__":
    main()
