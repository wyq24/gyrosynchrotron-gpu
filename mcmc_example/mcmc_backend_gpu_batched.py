"""GPU-batched exact MCMC backend for solar radio spectral fitting.

This module is designed as a drop-in *framework* for replacing the slow parts of
an emcee-based pixel/block spectral fitter while preserving the user's existing
GPU forward-model investment.

Key ideas:
1. Use one persistent GPU forward-model backend object.
2. Evaluate a *batch* of walkers at once via emcee(vectorize=True).
3. Remove multiprocessing for GPU execution.
4. Reuse neighboring-pixel posterior samples to warm-start new pixels.
5. Keep the forward model behind a narrow adapter so later migration to
   NumPyro / BlackJAX / JAX-FFI changes the sampler layer, not the whole code.

The validated wrapper adapter is now wired for the current legacy fitting
workflow, while still leaving `fill_batch_fn` / `extract_spectrum_fn` overridable
for future narrow-scope variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

try:
    from .assemble import node_theta_to_pixel_map
    from .segmentation import Segmentation, make_block_segmentation, make_pixel_segmentation
except ImportError:
    from assemble import node_theta_to_pixel_map
    from segmentation import Segmentation, make_block_segmentation, make_pixel_segmentation


ArrayLike = np.ndarray
ASEC2CM = 0.725e8
ARCSEC2_TO_SR = (np.pi / (180.0 * 3600.0)) ** 2
SPEED_OF_LIGHT_CGS = 2.99792458e10
K_BOLTZMANN_CGS = 1.380649e-16

LEGACY_AREA_ASEC2 = 1.625625
LEGACY_EMAX_MEV = 10.0
LEGACY_REDUCED_PARAM_DIM = 8
FULL_PHYSICAL_PARAM_DIM = 10
LEGACY_TARGET_FREQ_GHZ = np.array([
    1.2558008e+09, 1.5798831e+09, 2.5500792e+09, 2.8738688e+09,
    3.1975603e+09, 3.5210079e+09, 3.8442601e+09, 4.1673172e+09,
    4.4923274e+09, 4.8173373e+09, 5.1423473e+09, 5.4673577e+09,
    5.7923676e+09, 6.1173780e+09, 6.4423880e+09, 6.7673979e+09,
    7.0924083e+09, 7.4174182e+09, 7.7424282e+09, 8.0674386e+09,
    8.3924485e+09, 8.7174584e+09, 9.0424689e+09, 9.3674793e+09,
    9.6924887e+09, 1.0017499e+10, 1.0342510e+10, 1.0667519e+10,
    1.0992529e+10, 1.1317540e+10, 1.1642549e+10, 1.1967560e+10,
    1.2292570e+10, 1.2617580e+10, 1.2942590e+10, 1.3267600e+10,
    1.3592611e+10, 1.3917620e+10, 1.4242631e+10, 1.4567641e+10,
    1.4892650e+10, 1.5217661e+10, 1.5542671e+10, 1.5867681e+10,
    1.6192691e+10, 1.6517702e+10, 1.6842711e+10, 1.7167721e+10,
    1.7492732e+10, 1.7817741e+10,
], dtype=np.float64) / 1.0e9


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------


class ParameterNormalizer:
    """Normalize parameters and observations to the same bounded working space."""

    def __init__(
        self,
        param_bounds: Sequence[tuple[float, float]],
        x_log_bounds: tuple[float, float] | None = None,
        target_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        pb = np.asarray(param_bounds, dtype=np.float64)
        if pb.ndim != 2 or pb.shape[1] != 2:
            raise ValueError(f"param_bounds must be [D,2], got {pb.shape}")
        self.param_bounds = pb
        self.param_min = pb[:, 0]
        self.param_max = pb[:, 1]
        self.target_min, self.target_max = map(float, target_range)
        if x_log_bounds is None:
            self.x_log_min, self.x_log_max = 4.0, 9.0
        else:
            self.x_log_min, self.x_log_max = map(float, x_log_bounds)

    def normalize_params(self, params: ArrayLike) -> np.ndarray:
        p = np.asarray(params, dtype=np.float64)
        p = np.clip(p, self.param_min, self.param_max)
        p01 = (p - self.param_min) / (self.param_max - self.param_min)
        return p01 * (self.target_max - self.target_min) + self.target_min

    def denormalize_params(self, params_norm: ArrayLike) -> np.ndarray:
        pn = np.asarray(params_norm, dtype=np.float64)
        p01 = (pn - self.target_min) / (self.target_max - self.target_min)
        return p01 * (self.param_max - self.param_min) + self.param_min

    def normalize_observation(self, obs: ArrayLike) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        x_log = np.log10(np.clip(x, 1e-10, None))
        x_log = np.clip(x_log, self.x_log_min, self.x_log_max)
        x01 = (x_log - self.x_log_min) / (self.x_log_max - self.x_log_min)
        return x01 * (self.target_max - self.target_min) + self.target_min

    def normalize_observation_batch(self, obs_batch: ArrayLike) -> np.ndarray:
        xb = np.asarray(obs_batch, dtype=np.float64)
        if xb.ndim != 2:
            raise ValueError(f"obs_batch must be [B,F], got {xb.shape}")
        return np.vstack([self.normalize_observation(row) for row in xb])


# -----------------------------------------------------------------------------
# Forward-model adapter boundary
# -----------------------------------------------------------------------------


class ForwardModelBackend(Protocol):
    """Minimal contract needed by the exact sampler."""

    def simulate_batch(self, full_params_batch: np.ndarray) -> np.ndarray:
        """Return simulated spectra with shape [B, F]."""


@dataclass
class LegacySingleCallBackend:
    """Fallback adapter around an existing single-call simulator.

    Useful for testing the sampler logic before wiring in the validated batch
    wrapper. This is *not* the recommended production path.
    """

    simulator_fn: Callable[[np.ndarray], np.ndarray]

    def simulate_batch(self, full_params_batch: np.ndarray) -> np.ndarray:
        full_params_batch = np.asarray(full_params_batch, dtype=np.float64)
        if full_params_batch.ndim != 2:
            raise ValueError(f"full_params_batch must be [B,P], got {full_params_batch.shape}")
        out = [np.asarray(self.simulator_fn(row), dtype=np.float64).reshape(-1) for row in full_params_batch]
        return np.stack(out, axis=0)


# These prompt strings are meant to be copied directly into a coding agent if
# the batch wrapper integration still needs to be completed.
_FILL_BATCH_TODO = """
Implement fill_batch_fn(batch, full_params_batch) for your repo.

Requirements:
1. Open examples/GScodes.py and inspect the object returned by
   build_wrapper_powerlaw_iso_batch(batch_size=...).
2. Map each row of full_params_batch to the analytical PLW + ISO, Nz=1 wrapper
   fields expected by the validated narrow batch workflow.
3. full_params_batch uses the physical parameter order:
   [area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth,
    delta, theta, Emin_keV, Emax_MeV]
4. Fill only the currently supported validated path; do not broaden scope.
5. Preserve FP64 and the normal wrapper call path.
6. Batch object should be reused in-place across calls when possible.
""".strip()

_EXTRACT_SPECTRUM_TODO = """
Implement extract_spectrum_fn(result) for your repo.

Requirements:
1. Inspect the return object from GScodes.run_powerlaw_iso_batch_wrapper(...).
2. Return a numpy array with shape [B, F] containing the model spectrum in the
   same observable definition used by your likelihood (e.g. brightness
   temperature or flux; match the observation exactly).
3. If the wrapper returns polarized components, combine them consistently with
   the observation model.
4. Preserve the validated FP64 wrapper fast path.
""".strip()


def default_fill_batch_fn(batch: object, full_params_batch: np.ndarray) -> None:
    raise NotImplementedError(_FILL_BATCH_TODO)



def default_extract_spectrum_fn(result: object) -> np.ndarray:
    raise NotImplementedError(_EXTRACT_SPECTRUM_TODO)


def _expand_legacy_or_full_parameter_batch(full_params_batch: np.ndarray) -> np.ndarray:
    params = np.asarray(full_params_batch, dtype=np.float64)
    if params.ndim != 2:
        raise ValueError(f"full_params_batch must be [B,P], got {params.shape}")

    if params.shape[1] == LEGACY_REDUCED_PARAM_DIM:
        expanded = np.empty((params.shape[0], FULL_PHYSICAL_PARAM_DIM), dtype=np.float64)
        expanded[:, 0] = LEGACY_AREA_ASEC2
        expanded[:, 1:9] = params
        expanded[:, 9] = LEGACY_EMAX_MEV
        return expanded
    if params.shape[1] == FULL_PHYSICAL_PARAM_DIM:
        return params
    raise ValueError(
        "Expected full_params_batch to use either the legacy reduced 8D fitting order "
        "[depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV] "
        "or the full 10D order "
        "[area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV, Emax_MeV]."
    )


def fill_batch_from_legacy_or_full_params(batch: object, full_params_batch: np.ndarray) -> None:
    """Fill an `AnalyticalPowerLawIsoBatch` in-place from legacy 8D or full 10D parameters."""

    expanded = _expand_legacy_or_full_parameter_batch(full_params_batch)
    if expanded.shape[0] > batch.batch_size:
        raise ValueError(f"batch can hold {batch.batch_size} rows, got {expanded.shape[0]}")

    area_asec2 = expanded[:, 0]
    depth_asec = expanded[:, 1]
    bmag_100g = expanded[:, 2]
    temperature_mk = expanded[:, 3]
    log_nth = expanded[:, 4]
    log_nnth = expanded[:, 5]
    delta = expanded[:, 6]
    theta_deg = expanded[:, 7]
    emin_kev = expanded[:, 8]
    emax_mev = expanded[:, 9]

    if np.any(area_asec2 <= 0.0):
        raise ValueError("area_asec2 must be positive")
    if np.any(depth_asec <= 0.0):
        raise ValueError("depth_asec must be positive")
    if np.any(bmag_100g <= 0.0):
        raise ValueError("Bx100G must be positive")
    if np.any(temperature_mk <= 0.0):
        raise ValueError("T_MK must be positive")
    if np.any(emin_kev <= 0.0):
        raise ValueError("Emin_keV must be positive")
    if np.any(emax_mev <= (emin_kev / 1.0e3)):
        raise ValueError("Emax_MeV must be greater than Emin_keV / 1000 for every spectrum")

    active = expanded.shape[0]
    batch._active_batch_size = active

    np.copyto(batch.area_cm2[:active], area_asec2 * (ASEC2CM ** 2))
    np.copyto(batch.depth_cm[:active], depth_asec * ASEC2CM)
    np.copyto(batch.bmag_g[:active], bmag_100g * 100.0)
    np.copyto(batch.temperature_k[:active], temperature_mk * 1.0e6)
    np.copyto(batch.thermal_density_cm3[:active], np.power(10.0, log_nth))
    np.copyto(batch.nonthermal_density_cm3[:active], np.power(10.0, log_nnth))
    np.copyto(batch.delta[:active], delta)
    np.copyto(batch.theta_deg[:active], theta_deg)
    np.copyto(batch.emin_mev[:active], emin_kev / 1.0e3)
    np.copyto(batch.emax_mev[:active], emax_mev)


def _interpolate_positive_spectrum(native_freq_hz: np.ndarray, flux_native: np.ndarray, target_freq_hz: np.ndarray) -> np.ndarray:
    return np.power(
        10.0,
        np.interp(
            np.log10(target_freq_hz),
            np.log10(native_freq_hz),
            np.log10(np.maximum(flux_native, 1.0e-300)),
        ),
    )


def _flux_sfu_to_tb(target_freq_hz: np.ndarray, flux_sfu: np.ndarray, area_asec2: float) -> np.ndarray:
    solid_angle_sr = float(area_asec2) * ARCSEC2_TO_SR
    factor = (SPEED_OF_LIGHT_CGS ** 2) / (2.0 * K_BOLTZMANN_CGS * np.square(target_freq_hz) * solid_angle_sr)
    return np.asarray(flux_sfu, dtype=np.float64) * 1.0e-19 * factor


@dataclass
class LegacyTbSpectrumExtractor:
    """Convert the validated wrapper output to the legacy brightness-temperature observable."""

    batch: object
    target_freq_ghz: np.ndarray = field(default_factory=lambda: LEGACY_TARGET_FREQ_GHZ.copy())
    spec_in_tb: bool = True

    def __post_init__(self) -> None:
        self.target_freq_hz = np.asarray(self.target_freq_ghz, dtype=np.float64).reshape(-1) * 1.0e9
        if self.target_freq_hz.ndim != 1 or self.target_freq_hz.size == 0:
            raise ValueError("target_freq_ghz must be a non-empty 1D array")

    def __call__(self, result: object) -> np.ndarray:
        rl = np.asarray(result.rl, dtype=np.float64)
        status = np.asarray(result.status, dtype=np.int32).reshape(-1)
        native_freq_hz = np.asarray(result.native_freq_hz, dtype=np.float64).reshape(-1)
        if rl.ndim != 3 or rl.shape[0] != 7:
            raise ValueError(f"Expected result.rl to have shape [7,nfreq,batch], got {rl.shape}")

        batch_size = rl.shape[2]
        spectra = np.empty((batch_size, self.target_freq_hz.size), dtype=np.float64)
        total_flux = rl[5] + rl[6]
        area_asec2 = np.asarray(self.batch.area_cm2[:batch_size], dtype=np.float64) / (ASEC2CM ** 2)

        for index in range(batch_size):
            if status[index] != 0 or not np.any(total_flux[:, index]):
                spectra[index, :] = 1.0e4 if self.spec_in_tb else 1.0e-11
                continue

            interp_flux = _interpolate_positive_spectrum(
                native_freq_hz=native_freq_hz,
                flux_native=total_flux[:, index],
                target_freq_hz=self.target_freq_hz,
            )
            if self.spec_in_tb:
                spectra[index, :] = _flux_sfu_to_tb(self.target_freq_hz, interp_flux, area_asec2[index])
            else:
                spectra[index, :] = interp_flux
        return spectra


@dataclass
class ValidatedWrapperBackend:
    """Adapter for the validated GPU batch wrapper documented in CallingEfficiently.

    The public Python-level calling convention is intentionally preserved.
    Only two repo-specific hooks are required:
      * fill_batch_fn(batch, full_params_batch)
      * extract_spectrum_fn(result) -> np.ndarray[B, F]
    """

    lib_path: str = "source/MWTransferArr.so"
    batch_capacity: int = 128
    backend: str = "cuda"
    precision: str = "fp64"
    npoints: int = 16
    q_on: bool = True
    d_sun_au: float = 1.0
    nu_cr_factor: float = 0.0
    nu_wh_factor: float = 0.0
    target_freq_ghz: np.ndarray = field(default_factory=lambda: LEGACY_TARGET_FREQ_GHZ.copy())
    fill_batch_fn: Callable[[object, np.ndarray], None] = default_fill_batch_fn
    extract_spectrum_fn: Callable[[object], np.ndarray] = default_extract_spectrum_fn
    _gscodes: object = field(init=False, repr=False)
    _batch: object = field(init=False, repr=False)
    _batch_func_rl: object = field(init=False, repr=False)
    _status: np.ndarray = field(init=False, repr=False)
    _native_freq_hz: np.ndarray = field(init=False, repr=False)
    _rl: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.precision.lower() != "fp64":
            raise ValueError("ValidatedWrapperBackend is intended for FP64 exact fitting only.")
        if self.npoints != 16 or not self.q_on:
            raise ValueError("Validated fast path expects npoints=16 and q_on=True.")
        if self.d_sun_au != 1.0 or self.nu_cr_factor != 0.0 or self.nu_wh_factor != 0.0:
            raise ValueError("Validated fast path expects d_sun_au=1.0 and both correction factors = 0.0.")

        # Import lazily so this file can still be imported in environments where
        # the gyrosynchrotron repo is not fully present yet.
        from examples import GScodes

        self._gscodes = GScodes
        self.lib_path = GScodes.resolve_library_path(self.lib_path, prefer_source=True, require_local_source=True)
        if self.backend.lower() == "cuda" and not GScodes.cuda_available(self.lib_path):
            raise RuntimeError(
                f"CUDA backend requested for {self.lib_path}, but pyMW_Approx_Batch_CudaAvailable returned 0. "
                "Build with CUDA=1 on a machine with accessible NVIDIA runtime support."
            )
        self._batch = GScodes.build_wrapper_powerlaw_iso_batch(batch_size=self.batch_capacity)
        if self.fill_batch_fn is default_fill_batch_fn:
            self.fill_batch_fn = fill_batch_from_legacy_or_full_params
        if self.extract_spectrum_fn is default_extract_spectrum_fn:
            self.extract_spectrum_fn = LegacyTbSpectrumExtractor(
                batch=self._batch,
                target_freq_ghz=np.asarray(self.target_freq_ghz, dtype=np.float64),
                spec_in_tb=True,
            )
        self._batch_func_rl = GScodes.initMW_Approx_Batch_RL(self.lib_path)
        self._status = np.zeros(self.batch_capacity, dtype=np.int32)
        self._native_freq_hz = np.zeros(self._batch.nfreq, dtype=np.float64)
        self._rl = np.zeros((7, self._batch.nfreq, self.batch_capacity), dtype=np.float64, order="F")

    def simulate_batch(self, full_params_batch: np.ndarray) -> np.ndarray:
        fp = np.asarray(full_params_batch, dtype=np.float64)
        if fp.ndim != 2:
            raise ValueError(f"full_params_batch must be [B,P], got {fp.shape}")
        if fp.shape[0] > self.batch_capacity:
            raise ValueError(
                f"batch size {fp.shape[0]} exceeds adapter capacity {self.batch_capacity}; "
                "increase batch_capacity or chunk the call"
            )

        self.fill_batch_fn(self._batch, fp)
        batch_size = fp.shape[0]
        res = int(
            self._batch_func_rl(
                self._gscodes._native_backend_code(self.backend.lower()),
                self._gscodes._native_precision_code(self.precision),
                batch_size,
                self._batch.nfreq,
                int(self.npoints),
                int(bool(self.q_on)),
                float(self._batch.nu0_hz),
                float(self._batch.dlog10_nu),
                self._batch.area_cm2,
                self._batch.depth_cm,
                self._batch.bmag_g,
                self._batch.temperature_k,
                self._batch.thermal_density_cm3,
                self._batch.nonthermal_density_cm3,
                self._batch.delta,
                self._batch.theta_deg,
                self._batch.emin_mev,
                self._batch.emax_mev,
                float(self.d_sun_au),
                self._status,
                self._native_freq_hz,
                self._rl,
            )
        )
        if res != 0:
            raise RuntimeError(
                f"pyMW_Approx_Batch_RL failed with status {res} "
                f"({self._gscodes.approx_batch_error_name(res)})"
            )
        result = self._gscodes.BatchRunResult(
            status=self._status[:batch_size].copy(),
            rl=self._rl[:, :, :batch_size],
            native_freq_hz=self._native_freq_hz.copy(),
            postprocess_path="fused-native-rl-fast-path",
            postprocess_reason="",
        )
        spectra = np.asarray(self.extract_spectrum_fn(result), dtype=np.float64)
        if spectra.ndim != 2:
            raise ValueError(f"extract_spectrum_fn must return [B,F], got {spectra.shape}")
        if spectra.shape[0] != fp.shape[0]:
            raise ValueError(
                f"extract_spectrum_fn returned batch {spectra.shape[0]} for input batch {fp.shape[0]}"
            )
        return spectra


def build_legacy_8d_batched_backend(
    *,
    lib_path: str = "source/MWTransferArr.so",
    batch_capacity: int = 128,
    backend: str = "cuda",
    target_freq_ghz: np.ndarray | None = None,
) -> ValidatedWrapperBackend:
    """Build the validated FP64 wrapper backend for the current legacy 8D fitting workflow."""

    return ValidatedWrapperBackend(
        lib_path=lib_path,
        batch_capacity=batch_capacity,
        backend=backend,
        precision="fp64",
        npoints=16,
        q_on=True,
        d_sun_au=1.0,
        nu_cr_factor=0.0,
        nu_wh_factor=0.0,
        target_freq_ghz=LEGACY_TARGET_FREQ_GHZ.copy() if target_freq_ghz is None else np.asarray(target_freq_ghz, dtype=np.float64),
    )


def simulate_legacy_8d_spectrum_gpu_batched(
    params: np.ndarray,
    *,
    forward_backend: ForwardModelBackend | None = None,
    lib_path: str = "source/MWTransferArr.so",
    backend: str = "cuda",
) -> np.ndarray:
    """Simulate one legacy 8D spectrum through the validated batched wrapper path."""

    params_arr = np.asarray(params, dtype=np.float64).reshape(1, -1)
    local_backend = forward_backend
    if local_backend is None:
        local_backend = build_legacy_8d_batched_backend(
            lib_path=lib_path,
            batch_capacity=max(1, params_arr.shape[0]),
            backend=backend,
        )
    return np.asarray(local_backend.simulate_batch(params_arr)[0], dtype=np.float64)


# -----------------------------------------------------------------------------
# Exact batched log-probability evaluator
# -----------------------------------------------------------------------------


@dataclass
class BatchedNormalizedLogProbEvaluator:
    vary_bounds: Sequence[tuple[float, float]]
    vary_indices: Sequence[int]
    fixed_params: Sequence[float]
    x_log_bounds: tuple[float, float]
    observation_norm: np.ndarray
    noise_level: float
    forward_backend: ForwardModelBackend

    def __post_init__(self) -> None:
        self.normalizer = ParameterNormalizer(
            self.vary_bounds,
            x_log_bounds=self.x_log_bounds,
            target_range=(-5.0, 5.0),
        )
        self.vary_indices_arr = np.asarray(self.vary_indices, dtype=np.int64)
        self.fixed_params_arr = np.asarray(self.fixed_params, dtype=np.float64).reshape(-1)
        self.obs_norm_arr = np.asarray(self.observation_norm, dtype=np.float64).reshape(-1)
        self.sigma2 = float(self.noise_level) ** 2

    def _assemble_full_params_batch(self, params_denorm_batch: np.ndarray) -> np.ndarray:
        batch = np.repeat(self.fixed_params_arr[None, :], params_denorm_batch.shape[0], axis=0)
        batch[:, self.vary_indices_arr] = params_denorm_batch
        return batch

    def __call__(self, params_norm_batch: np.ndarray) -> np.ndarray:
        pb = np.asarray(params_norm_batch, dtype=np.float64)
        if pb.ndim == 1:
            pb = pb[None, :]
        if pb.ndim != 2:
            raise ValueError(f"params_norm_batch must be [B,D], got {pb.shape}")

        out = np.full((pb.shape[0],), -np.inf, dtype=np.float64)
        valid = np.all((pb >= -5.0) & (pb <= 5.0), axis=1)
        if not np.any(valid):
            return out

        params_denorm = np.asarray(self.normalizer.denormalize_params(pb[valid]), dtype=np.float64)
        full_params_batch = self._assemble_full_params_batch(params_denorm)
        sim_batch = np.asarray(self.forward_backend.simulate_batch(full_params_batch), dtype=np.float64)
        if sim_batch.ndim != 2:
            raise ValueError(f"forward_backend must return [B,F], got {sim_batch.shape}")
        sim_norm_batch = self.normalizer.normalize_observation_batch(sim_batch)

        residuals = self.obs_norm_arr[None, :] - sim_norm_batch
        ll = -0.5 * np.sum((residuals ** 2) / self.sigma2, axis=1)
        ll -= 0.5 * self.obs_norm_arr.size * np.log(2.0 * np.pi * self.sigma2)
        out[valid] = ll
        return out


# -----------------------------------------------------------------------------
# Sampler config / outputs
# -----------------------------------------------------------------------------


@dataclass
class WarmStartConfig:
    use_neighbor_samples: bool = True
    jitter_std_norm: float = 0.05
    exploration_fraction: float = 0.25
    max_neighbor_samples: int = 4096
    broad_init_scale: float = 0.35


@dataclass
class MCMCFitResult:
    theta_map: np.ndarray
    q16_map: np.ndarray
    q84_map: np.ndarray
    node_thetas: np.ndarray
    q16_nodes: np.ndarray
    q84_nodes: np.ndarray
    seg: Segmentation
    done_nodes: np.ndarray
    debug: dict


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _extract_node_spectra(
    cube: np.ndarray,
    seg: Segmentation,
    valid_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    h, w, _ = cube.shape
    spectra_list: list[np.ndarray] = []
    mask_flat = None
    if valid_mask is not None:
        if valid_mask.shape != (h, w):
            raise ValueError(f"valid_mask must be [H,W]={h,w}, got {valid_mask.shape}")
        mask_flat = valid_mask.ravel()

    for pixels in seg.nodes:
        px = pixels
        if valid_mask is not None:
            keep = mask_flat[px]
            px = px[keep]
        if px.size == 0:
            spectra_list.append(np.zeros((0, cube.shape[2]), dtype=np.float64))
            continue
        ys, xs = np.unravel_index(px, (h, w))
        spectra = np.asarray(cube[ys, xs, :], dtype=np.float64)
        finite = np.all(np.isfinite(spectra), axis=1)
        spectra_list.append(spectra[finite])
    return spectra_list



def _save_resume(
    resume_path: Path,
    node_thetas: np.ndarray,
    q16_nodes: np.ndarray,
    q84_nodes: np.ndarray,
    done_nodes: np.ndarray,
) -> None:
    np.savez_compressed(
        resume_path,
        node_thetas=node_thetas,
        q16_nodes=q16_nodes,
        q84_nodes=q84_nodes,
        done_nodes=done_nodes,
    )



def _load_resume(
    resume_path: Path,
    n_nodes: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not resume_path.exists():
        node_thetas = np.full((n_nodes, d), np.nan, dtype=np.float64)
        q16_nodes = np.full((n_nodes, d), np.nan, dtype=np.float64)
        q84_nodes = np.full((n_nodes, d), np.nan, dtype=np.float64)
        done_nodes = np.zeros((n_nodes,), dtype=bool)
        return node_thetas, q16_nodes, q84_nodes, done_nodes

    data = np.load(resume_path, allow_pickle=False)
    node_thetas = np.asarray(data["node_thetas"], dtype=np.float64)
    q16_nodes = np.asarray(data["q16_nodes"], dtype=np.float64)
    q84_nodes = np.asarray(data["q84_nodes"], dtype=np.float64)
    done_nodes = np.asarray(data["done_nodes"], dtype=bool)
    if node_thetas.shape != (n_nodes, d):
        raise ValueError(
            f"Resume shape mismatch: node_thetas {node_thetas.shape} vs expected {(n_nodes, d)}"
        )
    return node_thetas, q16_nodes, q84_nodes, done_nodes



def _pixel_neighbors(node_index: int, width: int, height: int) -> list[int]:
    y, x = divmod(node_index, width)
    out: list[int] = []
    for yy, xx in ((y, x - 1), (y - 1, x), (y - 1, x - 1), (y - 1, x + 1)):
        if 0 <= yy < height and 0 <= xx < width:
            out.append(yy * width + xx)
    return out



def _gather_neighbor_samples(
    node_index: int,
    segmentation: str,
    image_hw: tuple[int, int],
    sample_bank: dict[int, np.ndarray],
    max_neighbor_samples: int,
) -> np.ndarray | None:
    if not sample_bank:
        return None

    if segmentation == "pixel":
        h, w = image_hw
        keys = _pixel_neighbors(node_index, w, h)
    else:
        keys = [node_index - 1] if node_index > 0 else []

    chunks = [sample_bank[k] for k in keys if k in sample_bank and sample_bank[k].size > 0]
    if not chunks:
        return None
    cat = np.concatenate(chunks, axis=0)
    if cat.shape[0] > max_neighbor_samples:
        sel = np.linspace(0, cat.shape[0] - 1, max_neighbor_samples, dtype=int)
        cat = cat[sel]
    return cat



def _initialize_walkers(
    rng: np.random.Generator,
    n_walkers: int,
    n_dim: int,
    warm_samples_denorm: np.ndarray | None,
    normalizer: ParameterNormalizer,
    cfg: WarmStartConfig,
) -> np.ndarray:
    """Create walker positions in normalized space.

    Strategy:
      * majority: draw from neighboring posterior samples + small jitter
      * minority: broad exploration around the center of normalized space
    """
    n_explore = max(1, int(round(cfg.exploration_fraction * n_walkers)))
    n_reuse = n_walkers - n_explore

    chunks: list[np.ndarray] = []

    if warm_samples_denorm is not None and warm_samples_denorm.ndim == 2 and warm_samples_denorm.shape[0] > 0:
        idx = rng.integers(0, warm_samples_denorm.shape[0], size=n_reuse)
        reuse_denorm = np.asarray(warm_samples_denorm[idx], dtype=np.float64)
        reuse_norm = normalizer.normalize_params(reuse_denorm)
        reuse_norm += rng.normal(0.0, cfg.jitter_std_norm, size=reuse_norm.shape)
        chunks.append(reuse_norm)
    else:
        n_explore = n_walkers

    explore = rng.normal(loc=0.0, scale=cfg.broad_init_scale, size=(n_explore, n_dim))
    chunks.append(explore)

    p0 = np.concatenate(chunks, axis=0)
    if p0.shape[0] > n_walkers:
        p0 = p0[:n_walkers]
    elif p0.shape[0] < n_walkers:
        extra = rng.normal(loc=0.0, scale=cfg.broad_init_scale, size=(n_walkers - p0.shape[0], n_dim))
        p0 = np.concatenate([p0, extra], axis=0)
    p0 = np.clip(p0, -4.9, 4.9)
    return p0


# -----------------------------------------------------------------------------
# Exact batched emcee driver
# -----------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    n_walkers: int = 32
    n_steps: int = 1000
    burn_in: int = 300
    thin: int = 10
    noise_level: float = 0.1
    moves: tuple[tuple[str, float], ...] = (("stretch", 0.8), ("desnooker", 0.2))



def _build_emcee_moves(moves_spec: tuple[tuple[str, float], ...]):
    import emcee

    built = []
    for name, weight in moves_spec:
        key = name.strip().lower()
        if key == "stretch":
            built.append((emcee.moves.StretchMove(), float(weight)))
        elif key == "desnooker":
            built.append((emcee.moves.DESnookerMove(), float(weight)))
        elif key == "walk":
            built.append((emcee.moves.WalkMove(), float(weight)))
        else:
            raise ValueError(f"Unknown emcee move: {name}")
    return built



def run_single_mcmc_gpu_batched(
    spectrum: np.ndarray,
    vary_bounds: Sequence[tuple[float, float]],
    vary_indices: Sequence[int],
    fixed_params: Sequence[float],
    x_log_bounds: tuple[float, float],
    sampling_cfg: SamplingConfig,
    forward_backend: ForwardModelBackend,
    seed: int,
    warm_samples_denorm: np.ndarray | None = None,
    warm_start_cfg: WarmStartConfig | None = None,
) -> np.ndarray:
    """Exact posterior sampling for one spectrum using batched GPU likelihood calls."""
    import emcee

    n_dim = len(vary_bounds)
    if sampling_cfg.n_walkers < 2 * n_dim:
        raise ValueError(f"n_walkers must be >= 2*D ({2*n_dim}), got {sampling_cfg.n_walkers}")

    warm_cfg = warm_start_cfg or WarmStartConfig()
    rng = np.random.default_rng(seed)
    normalizer = ParameterNormalizer(vary_bounds, x_log_bounds=x_log_bounds, target_range=(-5.0, 5.0))
    obs_norm = normalizer.normalize_observation(spectrum)

    log_prob = BatchedNormalizedLogProbEvaluator(
        vary_bounds=vary_bounds,
        vary_indices=vary_indices,
        fixed_params=fixed_params,
        x_log_bounds=x_log_bounds,
        observation_norm=obs_norm,
        noise_level=sampling_cfg.noise_level,
        forward_backend=forward_backend,
    )

    p0 = _initialize_walkers(
        rng=rng,
        n_walkers=sampling_cfg.n_walkers,
        n_dim=n_dim,
        warm_samples_denorm=warm_samples_denorm,
        normalizer=normalizer,
        cfg=warm_cfg,
    )

    sampler = emcee.EnsembleSampler(
        sampling_cfg.n_walkers,
        n_dim,
        log_prob,
        vectorize=True,
        moves=_build_emcee_moves(sampling_cfg.moves),
    )
    sampler.run_mcmc(p0, sampling_cfg.n_steps, progress=False)

    flat_norm = sampler.get_chain(discard=sampling_cfg.burn_in, thin=sampling_cfg.thin, flat=True)
    if flat_norm.shape[0] == 0:
        raise RuntimeError("Empty MCMC chain after burn-in/thin. Increase n_steps or reduce burn_in.")
    return np.asarray(normalizer.denormalize_params(flat_norm), dtype=np.float64)


# -----------------------------------------------------------------------------
# Cube / node fitting with resume support and neighbor reuse
# -----------------------------------------------------------------------------


@dataclass
class CubeSamplingConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    checkpoint_every: int = 10
    max_nodes: int | None = None
    save_samples: bool = True



def fit_cube_mcmc_resumable_gpu(
    cube: np.ndarray,
    all_param_bounds: list[tuple[float, float]],
    vary_indices: list[int],
    fixed_params: list[float],
    x_log_bounds: tuple[float, float],
    segmentation: str,
    block_k: int,
    valid_mask: np.ndarray | None,
    out_dir: str,
    resume_path: str,
    cube_cfg: CubeSamplingConfig,
    forward_backend: ForwardModelBackend,
    seed: int,
) -> MCMCFitResult:
    cube_arr = np.asarray(cube, dtype=np.float64)
    if cube_arr.ndim != 3:
        raise ValueError(f"cube must be [H,W,F], got {cube_arr.shape}")

    h, w, _ = cube_arr.shape
    if segmentation == "pixel":
        seg = make_pixel_segmentation(h, w)
    elif segmentation == "block":
        seg = make_block_segmentation(h, w, block_k)
    else:
        raise ValueError(f"Unknown segmentation: {segmentation}")

    vary_bounds = [all_param_bounds[i] for i in vary_indices]
    d = len(vary_bounds)

    node_spectra = _extract_node_spectra(cube_arr, seg, valid_mask=valid_mask)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resume = Path(resume_path)
    samples_dir = out_root / "mcmc_posteriors"
    if cube_cfg.save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    node_thetas, q16_nodes, q84_nodes, done_nodes = _load_resume(resume, seg.n_nodes, d)

    processed = 0
    sampled = 0
    sample_bank: dict[int, np.ndarray] = {}

    for n in range(seg.n_nodes):
        if done_nodes[n]:
            continue
        if cube_cfg.max_nodes is not None and sampled >= cube_cfg.max_nodes:
            break

        spectra = node_spectra[n]
        if spectra.shape[0] == 0:
            done_nodes[n] = True
            processed += 1
            if processed % cube_cfg.checkpoint_every == 0:
                _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)
            continue

        node_spec = np.nanmedian(spectra, axis=0)
        warm_samples = None
        if cube_cfg.warm_start.use_neighbor_samples:
            warm_samples = _gather_neighbor_samples(
                node_index=n,
                segmentation=segmentation,
                image_hw=(h, w),
                sample_bank=sample_bank,
                max_neighbor_samples=cube_cfg.warm_start.max_neighbor_samples,
            )

        samples = run_single_mcmc_gpu_batched(
            spectrum=node_spec,
            vary_bounds=vary_bounds,
            vary_indices=vary_indices,
            fixed_params=fixed_params,
            x_log_bounds=x_log_bounds,
            sampling_cfg=cube_cfg.sampling,
            forward_backend=forward_backend,
            seed=seed + n,
            warm_samples_denorm=warm_samples,
            warm_start_cfg=cube_cfg.warm_start,
        )

        node_thetas[n] = np.median(samples, axis=0)
        q16_nodes[n] = np.quantile(samples, 0.16, axis=0)
        q84_nodes[n] = np.quantile(samples, 0.84, axis=0)
        sample_bank[n] = samples

        if cube_cfg.save_samples:
            np.save(samples_dir / f"node_{n:06d}.npy", samples)

        done_nodes[n] = True
        sampled += 1
        processed += 1
        if processed % cube_cfg.checkpoint_every == 0:
            _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)

    _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)

    theta_map = node_theta_to_pixel_map(seg, node_thetas)
    q16_map = node_theta_to_pixel_map(seg, q16_nodes)
    q84_map = node_theta_to_pixel_map(seg, q84_nodes)

    return MCMCFitResult(
        theta_map=theta_map,
        q16_map=q16_map,
        q84_map=q84_map,
        node_thetas=node_thetas,
        q16_nodes=q16_nodes,
        q84_nodes=q84_nodes,
        seg=seg,
        done_nodes=done_nodes,
        debug={
            "resume_path": str(resume),
            "processed_nodes": int(processed),
            "sampled_nodes": int(sampled),
            "save_samples": bool(cube_cfg.save_samples),
            "sampling": {
                "n_walkers": int(cube_cfg.sampling.n_walkers),
                "n_steps": int(cube_cfg.sampling.n_steps),
                "burn_in": int(cube_cfg.sampling.burn_in),
                "thin": int(cube_cfg.sampling.thin),
                "noise_level": float(cube_cfg.sampling.noise_level),
                "moves": list(cube_cfg.sampling.moves),
            },
            "warm_start": {
                "use_neighbor_samples": bool(cube_cfg.warm_start.use_neighbor_samples),
                "jitter_std_norm": float(cube_cfg.warm_start.jitter_std_norm),
                "exploration_fraction": float(cube_cfg.warm_start.exploration_fraction),
                "max_neighbor_samples": int(cube_cfg.warm_start.max_neighbor_samples),
                "broad_init_scale": float(cube_cfg.warm_start.broad_init_scale),
            },
        },
    )
