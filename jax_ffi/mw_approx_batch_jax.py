"""JAX wrapper around the staged XLA FFI proof for the validated batch path."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from . import mw_approx_batch_contract as contract


_RL_TARGET_NAME = "mw_approx_batch_rl"
_PLATFORM_TARGETS = {
    "cpu": ("cpu", "MWApproxBatchRlCpu"),
    "cuda": ("CUDA", "MWApproxBatchRlCuda"),
}
_REGISTERED_TARGETS = {}
_LOADED_LIBRARIES = {}


def _check_jax_x64() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "mw_approx_batch_rl_jax requires FP64. "
            "Call jax.config.update('jax_enable_x64', True) before invoking this wrapper."
        )


def _default_xla_library_path(lib_path=None) -> Path:
    if lib_path is not None:
        return Path(lib_path).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "source" / "MWTransferArrXLA.so"


def _row_major_layout(ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim))


def _normalize_platform_name(platform: str | None) -> str | None:
    if platform is None:
        return None
    if platform == "gpu":
        return "cuda"
    return platform


def _array_platform(array) -> str | None:
    devices_fn = getattr(array, "devices", None)
    if not callable(devices_fn):
        sharding = getattr(array, "sharding", None)
        device_set = getattr(sharding, "device_set", None)
        if device_set is None:
            return None
        try:
            devices = tuple(device_set)
        except Exception:
            return None
    else:
        try:
            devices = tuple(devices_fn())
        except Exception:
            # Tracers do not expose a concrete device placement during lowering.
            return None
    if len(devices) != 1:
        return None
    return _normalize_platform_name(devices[0].platform)


def _register_targets(lib_path=None) -> tuple[Path, frozenset[str]]:
    path = _default_xla_library_path(lib_path)
    if not path.exists():
        raise FileNotFoundError(f"JAX FFI library was not found at {path}")

    library = _LOADED_LIBRARIES.get(str(path))
    if library is None:
        library = ctypes.CDLL(str(path))
        _LOADED_LIBRARIES[str(path)] = library

    available_platforms = []
    for logical_platform, (registration_platform, symbol_name) in _PLATFORM_TARGETS.items():
        if not hasattr(library, symbol_name):
            continue

        registration_key = (_RL_TARGET_NAME, logical_platform)
        registered = _REGISTERED_TARGETS.get(registration_key)
        if registered is not None:
            if registered != str(path):
                raise RuntimeError(
                    f"FFI target {_RL_TARGET_NAME!r} for platform {logical_platform!r} "
                    f"is already registered from {registered}, cannot also register {path}"
                )
        else:
            jax.ffi.register_ffi_target(
                _RL_TARGET_NAME,
                jax.ffi.pycapsule(getattr(library, symbol_name)),
                platform=registration_platform,
            )
            _REGISTERED_TARGETS[registration_key] = str(path)
        available_platforms.append(logical_platform)

    if not available_platforms:
        raise RuntimeError(
            f"{path} does not export any supported JAX FFI targets: "
            f"{sorted(symbol_name for _, symbol_name in _PLATFORM_TARGETS.values())}"
        )

    return path, frozenset(available_platforms)


def expand_legacy_8d_params_jax(params):
    """Expand legacy reduced 8D fit parameters to the canonical 10D physical order."""

    params = jnp.asarray(params, dtype=jnp.float64)
    if params.shape[-1] != contract.LEGACY_REDUCED_PARAM_DIM:
        raise ValueError(
            f"Legacy 8D parameters must have last dimension {contract.LEGACY_REDUCED_PARAM_DIM}, "
            f"got {params.shape}"
        )

    legacy_area_cm2 = jnp.float64(contract.LEGACY_AREA_ASEC2 * (contract.ASEC2CM ** 2))
    legacy_emax_mev = jnp.float64(contract.LEGACY_EMAX_MEV)
    return jnp.concatenate(
        [
            jnp.full(params.shape[:-1] + (1,), legacy_area_cm2, dtype=jnp.float64),
            (params[..., 0:1] * jnp.float64(contract.ASEC2CM)),
            (params[..., 1:2] * jnp.float64(100.0)),
            (params[..., 2:3] * jnp.float64(1.0e6)),
            jnp.power(jnp.float64(10.0), params[..., 3:4]),
            jnp.power(jnp.float64(10.0), params[..., 4:5]),
            params[..., 5:6],
            params[..., 6:7],
            (params[..., 7:8] / jnp.float64(1.0e3)),
            jnp.full(params.shape[:-1] + (1,), legacy_emax_mev, dtype=jnp.float64),
        ],
        axis=-1,
    )


def canonicalize_params_jax(params):
    """Accept either legacy 8D or canonical 10D params and return canonical 10D."""

    params = jnp.asarray(params, dtype=jnp.float64)
    if params.shape[-1] == contract.LEGACY_REDUCED_PARAM_DIM:
        return expand_legacy_8d_params_jax(params)
    if params.shape[-1] == contract.FULL_PHYSICAL_PARAM_DIM:
        return params
    raise ValueError(
        "Parameters must use either the legacy 8D order or the canonical 10D physical order; "
        f"got shape {params.shape}"
    )


def mw_approx_batch_rl_jax(
    params,
    *,
    lib_path=None,
    nfreq: int = 100,
    nu0_hz: float = 0.8e9,
    dlog10_nu: float = 0.02,
    npoints: int = 16,
    q_on: bool = True,
    d_sun_au: float = 1.0,
):
    """Call the registered CPU/CUDA XLA FFI target and return RL as `[..., 7, nfreq]`."""

    _check_jax_x64()
    path, available_platforms = _register_targets(lib_path)

    params = canonicalize_params_jax(params)
    if params.ndim < 1:
        raise ValueError(f"params must have shape [..., 10], got {params.shape}")
    if int(nfreq) <= 0:
        raise ValueError("nfreq must be positive")
    if int(npoints) <= 0:
        raise ValueError("npoints must be positive")
    if float(d_sun_au) <= 0.0:
        raise ValueError("d_sun_au must be positive")

    params = jnp.asarray(params, dtype=jnp.float64)
    params_platform = _array_platform(params)
    if params_platform == "cuda" and "cuda" not in available_platforms:
        raise RuntimeError(
            f"Parameters are placed on a CUDA device, but {path} only registered "
            f"platforms {sorted(available_platforms)}. Rebuild MWTransferArrXLA with CUDA=1."
        )

    rl_storage_shape = params.shape[:-1] + (int(nfreq), 7)
    call = jax.ffi.ffi_call(
        _RL_TARGET_NAME,
        (
            jax.ShapeDtypeStruct(rl_storage_shape, jnp.float64),
            jax.ShapeDtypeStruct((int(nfreq),), jnp.float64),
        ),
        # The custom call already consumes explicit packed batches. Sequential
        # vmap avoids rank-mismatching the shared frequency output when a
        # higher-level likelihood vectorizes over independent evaluations.
        vmap_method="sequential",
        input_layouts=(_row_major_layout(params.ndim),),
        output_layouts=(
            _row_major_layout(len(rl_storage_shape)),
            (0,),
        ),
    )

    rl_storage, freq_hz = call(
        params,
        npoints=np.int32(npoints),
        q_on=np.int32(bool(q_on)),
        nu0_hz=np.float64(nu0_hz),
        dlog10_nu=np.float64(dlog10_nu),
        d_sun_au=np.float64(d_sun_au),
    )
    return jnp.swapaxes(rl_storage, -1, -2), freq_hz


def mw_approx_batch_legacy_spectrum_jax(
    params,
    *,
    lib_path=None,
    target_freq_ghz=contract.LEGACY_TARGET_FREQ_GHZ,
    spec_in_tb: bool = True,
    nfreq: int = 100,
    nu0_hz: float = 0.8e9,
    dlog10_nu: float = 0.02,
    npoints: int = 16,
    q_on: bool = True,
    d_sun_au: float = 1.0,
):
    """Return the current fitting observable as `[..., F_target]`."""

    canonical_params = canonicalize_params_jax(params)
    rl, freq_hz = mw_approx_batch_rl_jax(
        canonical_params,
        lib_path=lib_path,
        nfreq=nfreq,
        nu0_hz=nu0_hz,
        dlog10_nu=dlog10_nu,
        npoints=npoints,
        q_on=q_on,
        d_sun_au=d_sun_au,
    )

    target_freq_hz = jnp.asarray(target_freq_ghz, dtype=jnp.float64).reshape(-1) * jnp.float64(1.0e9)
    total_flux = rl[..., 5, :] + rl[..., 6, :]
    flat_flux = total_flux.reshape((-1, total_flux.shape[-1]))

    def _interp_row(row):
        return jnp.power(
            jnp.float64(10.0),
            jnp.interp(
                jnp.log10(target_freq_hz),
                jnp.log10(freq_hz),
                jnp.log10(jnp.maximum(row, jnp.float64(1.0e-300))),
            ),
        )

    interp_flux = jax.vmap(_interp_row)(flat_flux)
    out_shape = total_flux.shape[:-1] + (target_freq_hz.size,)
    interp_flux = interp_flux.reshape(out_shape)

    if not spec_in_tb:
        return interp_flux, target_freq_hz

    flat_area_asec2 = (canonical_params[..., 0] / jnp.float64(contract.ASEC2CM ** 2)).reshape((-1, 1))
    factor = (
        jnp.float64(contract.SPEED_OF_LIGHT_CGS ** 2)
        / (
            jnp.float64(2.0 * contract.K_BOLTZMANN_CGS)
            * jnp.square(target_freq_hz)[None, :]
            * flat_area_asec2
            * jnp.float64(contract.ARCSEC2_TO_SR)
        )
    )
    tb = interp_flux.reshape((-1, target_freq_hz.size)) * jnp.float64(1.0e-19) * factor
    return tb.reshape(out_shape), target_freq_hz
