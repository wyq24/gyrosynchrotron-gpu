"""Long-term migration skeleton for a JAX/NumPyro ensemble sampler.

Use this only after the forward-model call is JAX-compatible, either by:
1. rewriting the forward model in JAX, or
2. exposing the CUDA backend through a JAX FFI/custom call layer.

This file is *not* the immediate production solution for the current external
shared-library call path. It exists to freeze the interface boundary so that the
sampler migration later is cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class JAXLogDensityConfig:
    vary_bounds: Sequence[tuple[float, float]]
    vary_indices: Sequence[int]
    fixed_params: Sequence[float]
    x_log_bounds: tuple[float, float]
    observation: np.ndarray
    noise_level: float = 0.1


class JAXForwardModelProtocol:
    """Target interface for the future JAX-native forward model.

    Contract:
      * input:  [B, P] full physical parameter batch
      * output: [B, F] spectra as JAX arrays
    """

    def simulate_batch_jax(self, full_params_batch):  # pragma: no cover - placeholder
        raise NotImplementedError


class TODOJAXForwardModel(JAXForwardModelProtocol):
    def simulate_batch_jax(self, full_params_batch):
        raise NotImplementedError(
            "Implement a JAX-compatible forward call here. Preferred options: \n"
            "1. pure JAX rewrite of the validated narrow FP64 path, or\n"
            "2. JAX FFI/custom-call wrapper around the existing CUDA backend."
        )


# The coding agent can use this structure once the forward model is JAX-ready:
EXAMPLE = r'''
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, ESS


def build_logdensity(cfg: JAXLogDensityConfig, forward_model: JAXForwardModelProtocol):
    vary_idx = jnp.asarray(cfg.vary_indices)
    fixed = jnp.asarray(cfg.fixed_params, dtype=jnp.float64)
    obs = jnp.asarray(cfg.observation, dtype=jnp.float64)
    sigma2 = jnp.float64(cfg.noise_level ** 2)
    pb = jnp.asarray(cfg.vary_bounds, dtype=jnp.float64)
    pmin = pb[:, 0]
    pmax = pb[:, 1]
    xlog_min, xlog_max = map(jnp.float64, cfg.x_log_bounds)

    def normalize_obs(x):
        x = jnp.clip(x, 1e-10, None)
        x = jnp.log10(x)
        x = jnp.clip(x, xlog_min, xlog_max)
        x01 = (x - xlog_min) / (xlog_max - xlog_min)
        return x01 * 10.0 - 5.0

    obs_norm = normalize_obs(obs)

    def logdensity(theta_norm):
        inside = jnp.all((theta_norm >= -5.0) & (theta_norm <= 5.0))
        theta01 = (theta_norm + 5.0) / 10.0
        theta = theta01 * (pmax - pmin) + pmin
        full = fixed.at[vary_idx].set(theta)
        sim = forward_model.simulate_batch_jax(full[None, :])[0]
        sim_norm = normalize_obs(sim)
        resid = obs_norm - sim_norm
        ll = -0.5 * jnp.sum((resid ** 2) / sigma2)
        ll -= 0.5 * obs_norm.size * jnp.log(2.0 * jnp.pi * sigma2)
        return jnp.where(inside, ll, -jnp.inf)

    return logdensity


cfg = ...
forward_model = TODOJAXForwardModel()
logdensity = build_logdensity(cfg, forward_model)

kernel = ESS(lambda: None)  # Replace with the correct NumPyro model/kernal pattern.
# More realistically, define a NumPyro model using numpyro.factor(...) around logdensity.
'''
