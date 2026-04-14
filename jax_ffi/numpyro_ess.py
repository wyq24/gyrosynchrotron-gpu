"""NumPyro ESS helpers built on top of the staged JAX FFI forward wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .jax_runtime import ensure_jax_x64
from . import mw_approx_batch_contract as contract
from .mw_approx_batch_jax import mw_approx_batch_legacy_spectrum_jax


@dataclass
class JAXLogDensityConfig:
    """Likelihood config for the current normalized fitting workflow.

    `fixed_params` and `vary_indices` may refer either to the legacy reduced 8D
    fitting order or the canonical 10D physical order. The staged JAX forward
    wrapper accepts both conventions explicitly.
    """

    vary_bounds: Sequence[tuple[float, float]]
    vary_indices: Sequence[int]
    fixed_params: Sequence[float]
    x_log_bounds: tuple[float, float]
    observation: np.ndarray
    noise_level: float = 0.1


class JAXForwardModelProtocol(Protocol):
    """Minimal JAX-facing contract for the current sampler migration."""

    def simulate_batch_jax(self, full_params_batch):
        """Return simulated spectra with shape `[B, F]` as JAX arrays."""


@dataclass
class JAXFfiLegacyObservableForwardModel(JAXForwardModelProtocol):
    """Forward model adapter returning the current legacy fitting observable."""

    xla_lib_path: str | None = None
    target_freq_ghz: np.ndarray = field(default_factory=lambda: contract.LEGACY_TARGET_FREQ_GHZ.copy())
    nfreq: int = 100
    nu0_hz: float = 0.8e9
    dlog10_nu: float = 0.02
    npoints: int = 16
    q_on: bool = True
    d_sun_au: float = 1.0
    spec_in_tb: bool = True

    def simulate_batch_jax(self, full_params_batch):
        ensure_jax_x64()
        spectra, _ = mw_approx_batch_legacy_spectrum_jax(
            full_params_batch,
            lib_path=self.xla_lib_path,
            target_freq_ghz=self.target_freq_ghz,
            spec_in_tb=self.spec_in_tb,
            nfreq=self.nfreq,
            nu0_hz=self.nu0_hz,
            dlog10_nu=self.dlog10_nu,
            npoints=self.npoints,
            q_on=self.q_on,
            d_sun_au=self.d_sun_au,
        )
        return spectra


class TODOJAXForwardModel(JAXFfiLegacyObservableForwardModel):
    """Backward-compatible name retained from the original migration skeleton."""


@dataclass
class ESSRunConfig:
    num_warmup: int = 40
    num_samples: int = 80
    num_chains: int = 24
    thinning: int = 1
    progress_bar: bool = False
    randomize_split: bool = True
    max_steps: int = 10_000
    max_iter: int = 10_000
    init_mu: float = 2.0
    tune_mu: bool = True
    seed: int = 0


@dataclass
class ESSRunResult:
    samples_norm: np.ndarray
    samples_denorm: np.ndarray
    extra_fields: dict


@dataclass
class AIESRunConfig:
    num_warmup: int = 40
    num_samples: int = 80
    num_chains: int = 24
    thinning: int = 1
    progress_bar: bool = False
    randomize_split: bool = True
    moves: tuple[tuple[str, float], ...] = (("de", 1.0),)
    seed: int = 0


def _theta_norm_to_unconstrained(theta_norm):
    import jax.numpy as jnp

    theta_norm = jnp.asarray(theta_norm, dtype=jnp.float64)
    scaled = (theta_norm + jnp.float64(5.0)) / jnp.float64(10.0)
    scaled = jnp.clip(scaled, jnp.float64(1.0e-12), jnp.float64(1.0 - 1.0e-12))
    return jnp.log(scaled) - jnp.log1p(-scaled)


def _unconstrained_to_theta_norm(theta_raw):
    import jax
    import jax.numpy as jnp

    theta_raw = jnp.asarray(theta_raw, dtype=jnp.float64)
    return jnp.float64(10.0) * jax.nn.sigmoid(theta_raw) - jnp.float64(5.0)


def build_corrected_ess_kernel(
    *,
    model=None,
    potential_fn=None,
    randomize_split: bool = True,
    max_steps: int = 10_000,
    max_iter: int = 10_000,
    init_mu: float = 1.0,
    tune_mu: bool = True,
):
    """Return an ESS kernel with a corrected complementary-ensemble split.

    NumPyro 0.19.0's `EnsembleSampler.sample()` uses the second half of the
    ensemble as both active and inactive walkers during the second sub-step.
    That breaks the ensemble update. Keep NumPyro's ESS implementation, but fix
    the split used in `sample()`.
    """

    from numpyro.infer import ESS
    from numpyro.infer.ensemble import EnsembleSamplerState
    from numpyro.infer.ensemble_util import batch_ravel_pytree
    import jax

    class _PatchedESS(ESS):
        def sample(self, state, model_args, model_kwargs):
            z, inner_state, rng_key = state
            rng_key, _ = jax.random.split(rng_key)
            z_flat, unravel_fn = batch_ravel_pytree(z)

            if self._randomize_split:
                z_flat = jax.random.permutation(rng_key, z_flat, axis=0)

            split_ind = self._num_chains // 2

            def body_fn(i, z_flat_inner_state):
                z_flat, inner_state = z_flat_inner_state

                active, inactive = jax.lax.cond(
                    i == 0,
                    lambda x: (x[:split_ind], x[split_ind:]),
                    lambda x: (x[split_ind:], x[:split_ind]),
                    z_flat,
                )

                z_updates, inner_state = self.update_active_chains(active, inactive, inner_state)

                z_flat = jax.lax.cond(
                    i == 0,
                    lambda x: x.at[:split_ind].set(z_updates),
                    lambda x: x.at[split_ind:].set(z_updates),
                    z_flat,
                )
                return (z_flat, inner_state)

            z_flat, inner_state = jax.lax.fori_loop(0, 2, body_fn, (z_flat, inner_state))
            return EnsembleSamplerState(unravel_fn(z_flat), inner_state, rng_key)

    return _PatchedESS(
        model=model,
        potential_fn=potential_fn,
        randomize_split=randomize_split,
        max_steps=max_steps,
        max_iter=max_iter,
        init_mu=init_mu,
        tune_mu=tune_mu,
    )


def _normalize_params_jax(params, param_min, param_max):
    import jax.numpy as jnp

    p = jnp.asarray(params, dtype=jnp.float64)
    p = jnp.clip(p, param_min, param_max)
    p01 = (p - param_min) / (param_max - param_min)
    return p01 * jnp.float64(10.0) - jnp.float64(5.0)


def _denormalize_params_jax(params_norm, param_min, param_max):
    import jax.numpy as jnp

    pn = jnp.asarray(params_norm, dtype=jnp.float64)
    p01 = (pn + jnp.float64(5.0)) / jnp.float64(10.0)
    return p01 * (param_max - param_min) + param_min


def _normalize_observation_jax(obs, *, x_log_min, x_log_max):
    import jax.numpy as jnp

    x = jnp.asarray(obs, dtype=jnp.float64)
    x_log = jnp.log10(jnp.clip(x, jnp.float64(1.0e-10), None))
    x_log = jnp.clip(x_log, x_log_min, x_log_max)
    x01 = (x_log - x_log_min) / (x_log_max - x_log_min)
    return x01 * jnp.float64(10.0) - jnp.float64(5.0)


def build_logdensity(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    ensure_jax_x64()

    import jax.numpy as jnp

    vary_idx = jnp.asarray(cfg.vary_indices, dtype=jnp.int32)
    bounds = jnp.asarray(cfg.vary_bounds, dtype=jnp.float64)
    param_min = bounds[:, 0]
    param_max = bounds[:, 1]
    fixed = jnp.asarray(cfg.fixed_params, dtype=jnp.float64)
    obs = jnp.asarray(cfg.observation, dtype=jnp.float64)
    x_log_min = jnp.float64(cfg.x_log_bounds[0])
    x_log_max = jnp.float64(cfg.x_log_bounds[1])
    sigma2 = jnp.float64(float(cfg.noise_level) ** 2)
    obs_norm = _normalize_observation_jax(obs, x_log_min=x_log_min, x_log_max=x_log_max)

    def logdensity(theta_norm):
        theta_norm = jnp.asarray(theta_norm, dtype=jnp.float64)
        finite = jnp.isfinite(theta_norm)
        inside = jnp.all(
            finite & (theta_norm >= jnp.float64(-5.0)) & (theta_norm <= jnp.float64(5.0)),
            axis=-1,
        )
        theta_norm_safe = jnp.nan_to_num(
            theta_norm,
            nan=jnp.float64(0.0),
            posinf=jnp.float64(5.0),
            neginf=jnp.float64(-5.0),
        )
        theta = _denormalize_params_jax(
            jnp.clip(theta_norm_safe, jnp.float64(-5.0), jnp.float64(5.0)),
            param_min,
            param_max,
        )
        full = jnp.broadcast_to(fixed, theta.shape[:-1] + fixed.shape)
        full = full.at[..., vary_idx].set(theta)
        full_flat = full.reshape((-1, full.shape[-1]))
        sim = forward_model.simulate_batch_jax(full_flat)
        sim = sim.reshape(full.shape[:-1] + (sim.shape[-1],))
        sim_norm = _normalize_observation_jax(sim, x_log_min=x_log_min, x_log_max=x_log_max)
        resid = obs_norm - sim_norm
        ll = -jnp.float64(0.5) * jnp.sum((resid ** 2) / sigma2, axis=-1)
        ll -= jnp.float64(0.5) * obs_norm.size * jnp.log(jnp.float64(2.0) * jnp.pi * sigma2)
        return jnp.where(inside, ll, -jnp.inf)

    return logdensity


def build_potential_fn(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    logdensity = build_logdensity(cfg, forward_model)

    def potential_fn(theta_norm):
        return -logdensity(theta_norm)

    return potential_fn


def build_unconstrained_potential_fn(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    ensure_jax_x64()

    import jax
    import jax.numpy as jnp

    logdensity = build_logdensity(cfg, forward_model)

    def potential_fn(theta_raw):
        theta_raw = jnp.asarray(theta_raw, dtype=jnp.float64)
        theta_norm = _unconstrained_to_theta_norm(theta_raw)
        logdet = _theta_raw_logdet(theta_raw)
        return -(logdensity(theta_norm) + logdet)

    return potential_fn


def _theta_raw_logdet(theta_raw):
    ensure_jax_x64()

    import jax
    import jax.numpy as jnp

    theta_raw = jnp.asarray(theta_raw, dtype=jnp.float64)
    log_ten = jnp.log(jnp.float64(10.0))
    return jnp.sum(
        log_ten + jax.nn.log_sigmoid(theta_raw) + jax.nn.log_sigmoid(-theta_raw),
        axis=-1,
    )


def build_numpyro_factor_model(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    ensure_jax_x64()

    import numpyro
    import numpyro.distributions as dist

    logdensity = build_logdensity(cfg, forward_model)
    n_dim = len(cfg.vary_bounds)

    def model():
        theta_norm = numpyro.sample(
            "theta_norm",
            dist.Uniform(-5.0, 5.0).expand((n_dim,)).to_event(1),
        )
        numpyro.factor("logdensity", logdensity(theta_norm))

    return model


def build_dynamic_logdensity(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    ensure_jax_x64()

    import jax.numpy as jnp

    vary_idx = jnp.asarray(cfg.vary_indices, dtype=jnp.int32)
    bounds = jnp.asarray(cfg.vary_bounds, dtype=jnp.float64)
    param_min = bounds[:, 0]
    param_max = bounds[:, 1]
    x_log_min = jnp.float64(cfg.x_log_bounds[0])
    x_log_max = jnp.float64(cfg.x_log_bounds[1])
    sigma2 = jnp.float64(float(cfg.noise_level) ** 2)

    def logdensity(theta_norm, observation, fixed_params):
        theta_norm = jnp.asarray(theta_norm, dtype=jnp.float64)
        obs = jnp.asarray(observation, dtype=jnp.float64).reshape(-1)
        fixed = jnp.asarray(fixed_params, dtype=jnp.float64).reshape(-1)
        finite = jnp.isfinite(theta_norm)
        inside = jnp.all(
            finite & (theta_norm >= jnp.float64(-5.0)) & (theta_norm <= jnp.float64(5.0)),
            axis=-1,
        )
        theta_norm_safe = jnp.nan_to_num(
            theta_norm,
            nan=jnp.float64(0.0),
            posinf=jnp.float64(5.0),
            neginf=jnp.float64(-5.0),
        )
        theta = _denormalize_params_jax(
            jnp.clip(theta_norm_safe, jnp.float64(-5.0), jnp.float64(5.0)),
            param_min,
            param_max,
        )
        full = jnp.broadcast_to(fixed, theta.shape[:-1] + fixed.shape)
        full = full.at[..., vary_idx].set(theta)
        full_flat = full.reshape((-1, full.shape[-1]))
        sim = forward_model.simulate_batch_jax(full_flat)
        sim = sim.reshape(full.shape[:-1] + (sim.shape[-1],))
        obs_norm = _normalize_observation_jax(obs, x_log_min=x_log_min, x_log_max=x_log_max)
        sim_norm = _normalize_observation_jax(sim, x_log_min=x_log_min, x_log_max=x_log_max)
        resid = obs_norm - sim_norm
        ll = -jnp.float64(0.5) * jnp.sum((resid ** 2) / sigma2, axis=-1)
        ll -= jnp.float64(0.5) * obs_norm.size * jnp.log(jnp.float64(2.0) * jnp.pi * sigma2)
        return jnp.where(inside, ll, -jnp.inf)

    return logdensity


def build_dynamic_numpyro_factor_model(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    ensure_jax_x64()

    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    logdensity = build_dynamic_logdensity(cfg, forward_model)
    n_dim = len(cfg.vary_bounds)
    base_dist = dist.Normal(0.0, 1.0)

    def model(observation, fixed_params):
        theta_raw = numpyro.sample(
            "theta_raw",
            base_dist.expand((n_dim,)).to_event(1),
        )
        target_logprob = logdensity(
            _unconstrained_to_theta_norm(theta_raw),
            observation,
            fixed_params,
        ) + _theta_raw_logdet(theta_raw)
        base_logprob = jnp.sum(base_dist.log_prob(theta_raw), axis=-1)
        numpyro.factor("target_logprob", target_logprob - base_logprob)

    return model


def make_initial_ensemble(
    cfg: JAXLogDensityConfig,
    center_params_denorm,
    *,
    num_chains: int,
    jitter_std_norm: float = 0.05,
    seed: int = 0,
):
    ensure_jax_x64()

    import jax
    import jax.numpy as jnp

    bounds = jnp.asarray(cfg.vary_bounds, dtype=jnp.float64)
    center = _normalize_params_jax(jnp.asarray(center_params_denorm, dtype=jnp.float64), bounds[:, 0], bounds[:, 1])
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, (int(num_chains), center.shape[0]), dtype=jnp.float64) * jnp.float64(jitter_std_norm)
    return jnp.clip(center[None, :] + noise, jnp.float64(-4.9), jnp.float64(4.9))


def denormalize_posterior_samples(cfg: JAXLogDensityConfig, samples_norm):
    ensure_jax_x64()

    import jax.numpy as jnp

    bounds = jnp.asarray(cfg.vary_bounds, dtype=jnp.float64)
    return _denormalize_params_jax(jnp.asarray(samples_norm, dtype=jnp.float64), bounds[:, 0], bounds[:, 1])


class ReusableESSMcmcRunner:
    """Keep one NumPyro ESS sampler alive so compiled state can be reused."""

    def __init__(
        self,
        cfg: JAXLogDensityConfig,
        forward_model: JAXForwardModelProtocol,
        *,
        run_cfg: ESSRunConfig | None = None,
    ) -> None:
        ensure_jax_x64()

        self.cfg = cfg
        self.forward_model = forward_model
        self.run_cfg = run_cfg or ESSRunConfig()
        self._n_dim = len(cfg.vary_bounds)
        self._fixed_param_size = int(np.asarray(cfg.fixed_params, dtype=np.float64).size)
        self._observation_size = int(np.asarray(cfg.observation, dtype=np.float64).size)
        self._mcmc = self._build_mcmc()

    def _build_mcmc(self):
        from numpyro.infer import MCMC

        config = self.run_cfg
        if config.num_chains % 2 != 0:
            raise ValueError(f"ESS requires an even number of chains, got {config.num_chains}")
        if config.num_chains < 2 * self._n_dim:
            raise ValueError(f"ESS recommends num_chains >= 2 * D = {2 * self._n_dim}, got {config.num_chains}")

        model = build_dynamic_numpyro_factor_model(self.cfg, self.forward_model)
        kernel = build_corrected_ess_kernel(
            model=model,
            randomize_split=config.randomize_split,
            max_steps=config.max_steps,
            max_iter=config.max_iter,
            init_mu=config.init_mu,
            tune_mu=config.tune_mu,
        )
        return MCMC(
            kernel,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
            thinning=config.thinning,
            chain_method="vectorized",
            progress_bar=config.progress_bar,
            jit_model_args=True,
        )

    def run(
        self,
        *,
        observation,
        fixed_params,
        init_params=None,
        seed: int | None = None,
    ) -> ESSRunResult:
        ensure_jax_x64()

        import jax
        import jax.numpy as jnp

        obs = jnp.asarray(observation, dtype=jnp.float64).reshape(-1)
        fixed = jnp.asarray(fixed_params, dtype=jnp.float64).reshape(-1)
        if int(obs.size) != self._observation_size:
            raise ValueError(
                f"Observation size mismatch: got {int(obs.size)}, expected {self._observation_size}"
            )
        if int(fixed.size) != self._fixed_param_size:
            raise ValueError(
                f"Fixed-parameter size mismatch: got {int(fixed.size)}, expected {self._fixed_param_size}"
            )

        unconstrained_init = None
        if init_params is not None:
            unconstrained_init = {"theta_raw": _theta_norm_to_unconstrained(init_params)}

        run_seed = self.run_cfg.seed if seed is None else int(seed)
        self._mcmc.run(
            jax.random.PRNGKey(run_seed),
            obs,
            fixed,
            init_params=unconstrained_init,
        )

        samples_raw = self._mcmc.get_samples(group_by_chain=True)["theta_raw"]
        samples_norm = np.asarray(_unconstrained_to_theta_norm(samples_raw), dtype=np.float64)
        samples_denorm = np.asarray(denormalize_posterior_samples(self.cfg, samples_norm), dtype=np.float64)
        return ESSRunResult(
            samples_norm=samples_norm,
            samples_denorm=samples_denorm,
            extra_fields={
                key: np.asarray(value)
                for key, value in self._mcmc.get_extra_fields(group_by_chain=True).items()
            },
        )


def _build_aies_moves(moves_spec: Sequence[tuple[str, float]]):
    from numpyro.infer import AIES

    built = {}
    for name, weight in moves_spec:
        key = name.strip().lower()
        if key == "de":
            built[AIES.DEMove()] = float(weight)
        elif key == "stretch":
            built[AIES.StretchMove()] = float(weight)
        else:
            raise ValueError(f"Unknown AIES move: {name}")
    return built


class ReusableAIESMcmcRunner:
    """Keep one NumPyro AIES sampler alive so compiled state can be reused."""

    def __init__(
        self,
        cfg: JAXLogDensityConfig,
        forward_model: JAXForwardModelProtocol,
        *,
        run_cfg: AIESRunConfig | None = None,
    ) -> None:
        ensure_jax_x64()

        self.cfg = cfg
        self.forward_model = forward_model
        self.run_cfg = run_cfg or AIESRunConfig()
        self._n_dim = len(cfg.vary_bounds)
        self._fixed_param_size = int(np.asarray(cfg.fixed_params, dtype=np.float64).size)
        self._observation_size = int(np.asarray(cfg.observation, dtype=np.float64).size)
        self._mcmc = self._build_mcmc()

    def _build_mcmc(self):
        from numpyro.infer import AIES, MCMC

        config = self.run_cfg
        if config.num_chains % 2 != 0:
            raise ValueError(f"AIES requires an even number of chains, got {config.num_chains}")
        if config.num_chains < 2 * self._n_dim:
            raise ValueError(f"AIES recommends num_chains >= 2 * D = {2 * self._n_dim}, got {config.num_chains}")

        model = build_dynamic_numpyro_factor_model(self.cfg, self.forward_model)
        kernel = AIES(
            model=model,
            randomize_split=config.randomize_split,
            moves=_build_aies_moves(config.moves),
        )
        return MCMC(
            kernel,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
            thinning=config.thinning,
            chain_method="vectorized",
            progress_bar=config.progress_bar,
            jit_model_args=True,
        )

    def run(
        self,
        *,
        observation,
        fixed_params,
        init_params=None,
        seed: int | None = None,
    ) -> ESSRunResult:
        ensure_jax_x64()

        import jax
        import jax.numpy as jnp

        obs = jnp.asarray(observation, dtype=jnp.float64).reshape(-1)
        fixed = jnp.asarray(fixed_params, dtype=jnp.float64).reshape(-1)
        if int(obs.size) != self._observation_size:
            raise ValueError(
                f"Observation size mismatch: got {int(obs.size)}, expected {self._observation_size}"
            )
        if int(fixed.size) != self._fixed_param_size:
            raise ValueError(
                f"Fixed-parameter size mismatch: got {int(fixed.size)}, expected {self._fixed_param_size}"
            )

        unconstrained_init = None
        if init_params is not None:
            unconstrained_init = {"theta_raw": _theta_norm_to_unconstrained(init_params)}

        run_seed = self.run_cfg.seed if seed is None else int(seed)
        self._mcmc.run(
            jax.random.PRNGKey(run_seed),
            obs,
            fixed,
            init_params=unconstrained_init,
        )

        samples_raw = self._mcmc.get_samples(group_by_chain=True)["theta_raw"]
        samples_norm = np.asarray(_unconstrained_to_theta_norm(samples_raw), dtype=np.float64)
        samples_denorm = np.asarray(denormalize_posterior_samples(self.cfg, samples_norm), dtype=np.float64)
        return ESSRunResult(
            samples_norm=samples_norm,
            samples_denorm=samples_denorm,
            extra_fields={
                key: np.asarray(value)
                for key, value in self._mcmc.get_extra_fields(group_by_chain=True).items()
            },
        )


def run_ess_mcmc(
    cfg: JAXLogDensityConfig,
    forward_model: JAXForwardModelProtocol,
    *,
    run_cfg: ESSRunConfig | None = None,
    init_params=None,
):
    ensure_jax_x64()

    import numpy as np
    from numpyro.infer import MCMC

    config = run_cfg or ESSRunConfig()
    n_dim = len(cfg.vary_bounds)
    if config.num_chains % 2 != 0:
        raise ValueError(f"ESS requires an even number of chains, got {config.num_chains}")
    if config.num_chains < 2 * n_dim:
        raise ValueError(f"ESS recommends num_chains >= 2 * D = {2 * n_dim}, got {config.num_chains}")

    potential_fn = build_unconstrained_potential_fn(cfg, forward_model)
    kernel = build_corrected_ess_kernel(
        potential_fn=potential_fn,
        randomize_split=config.randomize_split,
        max_steps=config.max_steps,
        max_iter=config.max_iter,
        init_mu=config.init_mu,
        tune_mu=config.tune_mu,
    )

    if init_params is None:
        unconstrained_init = None
    else:
        unconstrained_init = _theta_norm_to_unconstrained(init_params)

    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        thinning=config.thinning,
        chain_method="vectorized",
        progress_bar=config.progress_bar,
    )
    import jax

    mcmc.run(jax.random.PRNGKey(config.seed), init_params=unconstrained_init)
    samples_raw = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=np.float64)
    samples_norm = np.asarray(_unconstrained_to_theta_norm(samples_raw), dtype=np.float64)
    samples_denorm = np.asarray(denormalize_posterior_samples(cfg, samples_norm), dtype=np.float64)
    return ESSRunResult(
        samples_norm=samples_norm,
        samples_denorm=samples_denorm,
        extra_fields={key: np.asarray(value) for key, value in mcmc.get_extra_fields(group_by_chain=True).items()},
    )


def run_aies_mcmc(
    cfg: JAXLogDensityConfig,
    forward_model: JAXForwardModelProtocol,
    *,
    run_cfg: AIESRunConfig | None = None,
    init_params=None,
):
    ensure_jax_x64()

    runner = ReusableAIESMcmcRunner(cfg, forward_model, run_cfg=run_cfg)
    return runner.run(
        observation=np.asarray(cfg.observation, dtype=np.float64),
        fixed_params=np.asarray(cfg.fixed_params, dtype=np.float64),
        init_params=init_params,
        seed=(run_cfg.seed if run_cfg is not None else None),
    )
