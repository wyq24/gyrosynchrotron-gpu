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
    "jax_ess_trial",
    script_path=__file__,
    repo_root=REPO_ROOT,
    extra_paths=(TOOLS_DIR, EXAMPLES_DIR),
)

import jax
from jax_ffi.numpyro_ess import (
    ESSRunConfig,
    JAXFfiLegacyObservableForwardModel,
    JAXLogDensityConfig,
    build_logdensity,
    make_initial_ensemble,
    run_ess_mcmc,
)
from mcmc_example.mcmc_backend_gpu_batched import (
    BatchedNormalizedLogProbEvaluator,
    ParameterNormalizer,
    SamplingConfig,
    WarmStartConfig,
    build_legacy_8d_batched_backend,
    run_single_mcmc_gpu_batched,
)


TRUE_PARAMS_8D = np.array([5.0, 2.0, 1.0, 10.0, 5.0, 4.0, 45.0, 10.0], dtype=np.float64)
PARAM_NAMES_8D = [
    "depth_asec",
    "bmag_hundreds_g",
    "temperature_mk",
    "log10_nth",
    "log10_nb",
    "delta",
    "theta_deg",
    "emin_keV",
]
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


def _error_metrics(candidate, reference):
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    abs_err = np.abs(candidate - reference)
    floor = np.maximum(np.abs(reference), 1.0e-300)
    rel_err = abs_err / floor
    return {
        "max_abs": float(np.max(abs_err)),
        "max_rel": float(np.max(rel_err)),
        "median_abs": float(np.median(abs_err)),
        "median_rel": float(np.median(rel_err)),
    }


def _posterior_summary(samples):
    samples = np.asarray(samples, dtype=np.float64)
    return {
        "median": np.median(samples, axis=0),
        "q16": np.quantile(samples, 0.16, axis=0),
        "q84": np.quantile(samples, 0.84, axis=0),
    }


def _build_warm_samples(bounds, center, *, count, seed, scale_fraction):
    rng = np.random.default_rng(seed)
    bounds_arr = np.asarray(bounds, dtype=np.float64)
    span = bounds_arr[:, 1] - bounds_arr[:, 0]
    samples = np.asarray(center, dtype=np.float64)[None, :] + rng.normal(
        0.0,
        scale_fraction * span,
        size=(count, bounds_arr.shape[0]),
    )
    return np.clip(samples, bounds_arr[:, 0], bounds_arr[:, 1])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a toy NumPyro ESS trial against the staged JAX FFI forward path.")
    parser.add_argument("--lib", default=str(REPO_ROOT / "source" / "MWTransferArr.so"))
    parser.add_argument("--xla-lib", default=str(REPO_ROOT / "source" / "MWTransferArrXLA.so"))
    parser.add_argument("--reference-backend", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--jax-device-platform",
        default="gpu",
        choices=["default", "cpu", "gpu", "cuda"],
        help="Device placement for fixed-point likelihood checks and ESS initial chains.",
    )
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--x-log-min", type=float, default=4.0)
    parser.add_argument("--x-log-max", type=float, default=9.0)
    parser.add_argument("--num-chains", type=int, default=16)
    parser.add_argument("--ess-warmup", type=int, default=8)
    parser.add_argument("--ess-samples", type=int, default=16)
    parser.add_argument("--ess-thin", type=int, default=1)
    parser.add_argument("--ess-max-steps", type=int, default=128)
    parser.add_argument("--ess-max-iter", type=int, default=128)
    parser.add_argument("--ess-init-mu", type=float, default=2.0)
    parser.add_argument("--ess-no-tune-mu", action="store_true")
    parser.add_argument("--ess-no-randomize-split", action="store_true")
    parser.add_argument("--emcee-steps", type=int, default=64)
    parser.add_argument("--emcee-burnin", type=int, default=16)
    parser.add_argument("--emcee-thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fixed-point-count", type=int, default=8)
    parser.add_argument("--init-jitter-std-norm", type=float, default=0.05)
    parser.add_argument("--warm-sample-scale-fraction", type=float, default=0.03)
    parser.add_argument("--posterior-span-tol", type=float, default=0.15)
    return parser


def run_trial(
    *,
    lib: str,
    xla_lib: str,
    reference_backend: str,
    jax_device_platform: str,
    noise_level: float,
    x_log_min: float,
    x_log_max: float,
    num_chains: int,
    ess_warmup: int,
    ess_samples: int,
    ess_thin: int,
    ess_max_steps: int,
    ess_max_iter: int,
    ess_init_mu: float,
    ess_no_tune_mu: bool,
    ess_no_randomize_split: bool,
    emcee_steps: int,
    emcee_burnin: int,
    emcee_thin: int,
    seed: int,
    fixed_point_count: int,
    init_jitter_std_norm: float,
    warm_sample_scale_fraction: float,
):
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX x64 mode is required for the ESS trial.")

    jax_device = resolve_jax_device(jax_device_platform)
    x_log_bounds = (float(x_log_min), float(x_log_max))

    reference_backend = build_legacy_8d_batched_backend(
        lib_path=lib,
        batch_capacity=max(num_chains, fixed_point_count, 32),
        backend=reference_backend,
    )
    observation = np.asarray(reference_backend.simulate_batch(TRUE_PARAMS_8D[None, :])[0], dtype=np.float64)

    cfg = JAXLogDensityConfig(
        vary_bounds=ALL_PARAM_BOUNDS_8D,
        vary_indices=list(range(TRUE_PARAMS_8D.size)),
        fixed_params=[0.0] * TRUE_PARAMS_8D.size,
        x_log_bounds=x_log_bounds,
        observation=observation,
        noise_level=noise_level,
    )
    jax_forward = JAXFfiLegacyObservableForwardModel(xla_lib_path=xla_lib)
    logdensity = build_logdensity(cfg, jax_forward)

    normalizer = ParameterNormalizer(ALL_PARAM_BOUNDS_8D, x_log_bounds=x_log_bounds, target_range=(-5.0, 5.0))
    obs_norm = normalizer.normalize_observation(observation)
    reference_logprob = BatchedNormalizedLogProbEvaluator(
        vary_bounds=ALL_PARAM_BOUNDS_8D,
        vary_indices=list(range(TRUE_PARAMS_8D.size)),
        fixed_params=[0.0] * TRUE_PARAMS_8D.size,
        x_log_bounds=x_log_bounds,
        observation_norm=obs_norm,
        noise_level=noise_level,
        forward_backend=reference_backend,
    )

    fixed_point_params = _build_warm_samples(
        ALL_PARAM_BOUNDS_8D,
        TRUE_PARAMS_8D,
        count=fixed_point_count,
        seed=seed,
        scale_fraction=warm_sample_scale_fraction,
    )
    fixed_point_norm = np.asarray(normalizer.normalize_params(fixed_point_params), dtype=np.float64)
    jax_fixed_point_norm = maybe_device_put(fixed_point_norm, jax_device)
    jax_logdensity_values = np.asarray(jax.jit(logdensity)(jax_fixed_point_norm), dtype=np.float64)
    reference_logdensity_values = np.asarray(reference_logprob(fixed_point_norm), dtype=np.float64)
    logdensity_metrics = _error_metrics(jax_logdensity_values, reference_logdensity_values)

    ess_init = make_initial_ensemble(
        cfg,
        TRUE_PARAMS_8D,
        num_chains=num_chains,
        jitter_std_norm=init_jitter_std_norm,
        seed=seed + 1,
    )
    ess_init = maybe_device_put(ess_init, jax_device)
    ess_result = run_ess_mcmc(
        cfg,
        jax_forward,
        run_cfg=ESSRunConfig(
            num_warmup=ess_warmup,
            num_samples=ess_samples,
            num_chains=num_chains,
            thinning=ess_thin,
            progress_bar=False,
            randomize_split=not ess_no_randomize_split,
            max_steps=ess_max_steps,
            max_iter=ess_max_iter,
            init_mu=ess_init_mu,
            tune_mu=not ess_no_tune_mu,
            seed=seed,
        ),
        init_params=ess_init,
    )
    ess_samples = ess_result.samples_denorm.reshape(-1, TRUE_PARAMS_8D.size)

    warm_samples = _build_warm_samples(
        ALL_PARAM_BOUNDS_8D,
        TRUE_PARAMS_8D,
        count=max(256, num_chains * 8),
        seed=seed + 2,
        scale_fraction=warm_sample_scale_fraction,
    )
    emcee_samples = run_single_mcmc_gpu_batched(
        observation,
        vary_bounds=ALL_PARAM_BOUNDS_8D,
        vary_indices=list(range(TRUE_PARAMS_8D.size)),
        fixed_params=[0.0] * TRUE_PARAMS_8D.size,
        x_log_bounds=x_log_bounds,
        sampling_cfg=SamplingConfig(
            n_walkers=num_chains,
            n_steps=emcee_steps,
            burn_in=emcee_burnin,
            thin=emcee_thin,
            noise_level=noise_level,
        ),
        forward_backend=reference_backend,
        seed=seed,
        warm_samples_denorm=warm_samples,
        warm_start_cfg=WarmStartConfig(
            use_neighbor_samples=True,
            jitter_std_norm=init_jitter_std_norm,
            exploration_fraction=0.0,
            max_neighbor_samples=warm_samples.shape[0],
            broad_init_scale=0.25,
        ),
    )

    ess_summary = _posterior_summary(ess_samples)
    emcee_summary = _posterior_summary(emcee_samples)
    spans = np.asarray([hi - lo for lo, hi in ALL_PARAM_BOUNDS_8D], dtype=np.float64)
    median_span_fraction = np.abs(ess_summary["median"] - emcee_summary["median"]) / spans
    q16_span_fraction = np.abs(ess_summary["q16"] - emcee_summary["q16"]) / spans
    q84_span_fraction = np.abs(ess_summary["q84"] - emcee_summary["q84"]) / spans
    worst_median_idx = int(np.argmax(median_span_fraction))
    worst_q16_idx = int(np.argmax(q16_span_fraction))
    worst_q84_idx = int(np.argmax(q84_span_fraction))
    return {
        "seed": int(seed),
        "reference_backend": str(reference_backend.backend if hasattr(reference_backend, "backend") else reference_backend),
        "jax_default_backend": jax.default_backend(),
        "jax_input_device": device_label(jax_device),
        "ess_init_mu": float(ess_init_mu),
        "ess_tune_mu": bool(not ess_no_tune_mu),
        "ess_randomize_split": bool(not ess_no_randomize_split),
        "observation_freq_count": int(observation.size),
        "fixed_point_count": int(fixed_point_norm.shape[0]),
        "logdensity_metrics": logdensity_metrics,
        "ess_samples_shape": tuple(int(v) for v in ess_samples.shape),
        "emcee_samples_shape": tuple(int(v) for v in emcee_samples.shape),
        "max_median_span_fraction": float(np.max(median_span_fraction)),
        "worst_median_param": PARAM_NAMES_8D[worst_median_idx],
        "max_q16_span_fraction": float(np.max(q16_span_fraction)),
        "worst_q16_param": PARAM_NAMES_8D[worst_q16_idx],
        "max_q84_span_fraction": float(np.max(q84_span_fraction)),
        "worst_q84_param": PARAM_NAMES_8D[worst_q84_idx],
        "median_span_fraction": median_span_fraction,
        "q16_span_fraction": q16_span_fraction,
        "q84_span_fraction": q84_span_fraction,
    }


def print_trial_summary(result: dict) -> None:
    print("JAX ESS trial summary")
    print(f"seed: {result['seed']}")
    print(f"reference_backend: {result['reference_backend']}")
    print(f"jax_default_backend: {result['jax_default_backend']}")
    print(f"jax_input_device: {result['jax_input_device']}")
    print(f"ess_init_mu: {result['ess_init_mu']}")
    print(f"ess_tune_mu: {result['ess_tune_mu']}")
    print(f"ess_randomize_split: {result['ess_randomize_split']}")
    print(f"observation_freq_count: {result['observation_freq_count']}")
    print(f"fixed_point_count: {result['fixed_point_count']}")
    print(
        f"logdensity_agreement: max_abs={result['logdensity_metrics']['max_abs']:.3e} "
        f"max_rel={result['logdensity_metrics']['max_rel']:.3e} "
        f"median_abs={result['logdensity_metrics']['median_abs']:.3e} "
        f"median_rel={result['logdensity_metrics']['median_rel']:.3e}"
    )
    print(f"ess_samples_shape: {result['ess_samples_shape']}")
    print(f"emcee_samples_shape: {result['emcee_samples_shape']}")
    print(f"max_median_span_fraction: {result['max_median_span_fraction']:.3e}")
    print(f"worst_median_param: {result['worst_median_param']}")
    print(f"max_q16_span_fraction: {result['max_q16_span_fraction']:.3e}")
    print(f"worst_q16_param: {result['worst_q16_param']}")
    print(f"max_q84_span_fraction: {result['max_q84_span_fraction']:.3e}")
    print(f"worst_q84_param: {result['worst_q84_param']}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_trial(**vars(args))
    print_trial_summary(result)

    if result["logdensity_metrics"]["max_abs"] > 1.0e-8 and result["logdensity_metrics"]["max_rel"] > 5.0e-11:
        raise RuntimeError("JAX logdensity and current exact batched backend disagree beyond tolerance.")

    if result["max_median_span_fraction"] > args.posterior_span_tol:
        raise RuntimeError(
            f"ESS posterior median drift exceeds tolerance: "
            f"max_median_span_fraction={result['max_median_span_fraction']:.3e} "
            f"> {args.posterior_span_tol:.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
