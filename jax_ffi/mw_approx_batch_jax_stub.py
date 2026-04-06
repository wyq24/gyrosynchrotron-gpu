"""Stub/sketch for the future JAX FFI wrapper around the validated batch path.

This file is a non-functional interface sketch.  It exists to:
  1. Freeze the proposed JAX-facing function signature before engineering begins.
  2. Show exactly where the ctypes → XLA FFI boundary replacement happens.
  3. Serve as the starting point for M1 (see doc/JAXVectorizedForwardFeasibility.md).

DO NOT import this in production code.
DO NOT modify any existing files based on this stub.
The ctypes path (GScodes.py / MWTransferArr.so) is the authoritative validated path.

Parameter order in params[:, i]
--------------------------------
  0: area_cm2              — source area in cm^2
  1: depth_cm              — source depth in cm
  2: bmag_g                — magnetic field in Gauss
  3: temperature_k         — thermal temperature in Kelvin
  4: thermal_density_cm3   — thermal electron density in cm^-3
  5: nonthermal_density_cm3 — nonthermal electron density in cm^-3
  6: delta                 — power-law spectral index
  7: theta_deg             — viewing angle in degrees
  8: emin_mev              — minimum energy in MeV
  9: emax_mev              — maximum energy in MeV

Output rl[:, :, :] layout
--------------------------
  rl[b, r, f]  where  b ∈ [0, B)  r ∈ [0, 7)  f ∈ [0, F)
  (C-contiguous; transposed from the F-contiguous [7, F, B] in the ctypes path)
  The 7 radio components follow the same ordering as pyMW_Approx_Batch_RL.

FP64 note
---------
  Requires jax.config.update("jax_enable_x64", True) before import.
  FP32 is not supported (consistent with policy in OptFP32Salvage.md).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Placeholder: replace with real XLA FFI registration once MWTransferArrXLA.so
# is built.  See doc/JAXVectorizedForwardFeasibility.md § M1.
# ---------------------------------------------------------------------------

_FFI_REGISTERED = False  # set True after plugin load in M1


def _check_jax_x64() -> None:
    """Raise informatively if x64 mode is not enabled."""
    try:
        import jax
        import jax.numpy as jnp
        if jnp.float64 is not jnp.float32 and not jax.config.jax_enable_x64:
            raise RuntimeError(
                "mw_approx_batch_rl_jax requires FP64. "
                "Call jax.config.update('jax_enable_x64', True) before import."
            )
    except ImportError as exc:
        raise ImportError("JAX is required for this module.") from exc


def mw_approx_batch_rl_jax(
    params,          # [B, 10] float64 JAX array, physical units per column above
    *,
    nfreq: int = 100,
    nu0_hz: float = 0.8e9,
    dlog10_nu: float = 0.02,
    npoints: int = 16,
    q_on: bool = True,
    d_sun_au: float = 1.0,
):
    """JAX-native forward call for the validated narrow GS batch path.

    NOT yet implemented — this is a signature-only stub for M1.

    Returns
    -------
    rl : jax.Array, shape [B, 7, nfreq], dtype float64
    freq_hz : jax.Array, shape [nfreq], dtype float64
    """
    _check_jax_x64()
    if not _FFI_REGISTERED:
        raise NotImplementedError(
            "The XLA FFI plugin has not been built yet.\n"
            "See doc/JAXVectorizedForwardFeasibility.md for the M1 implementation plan.\n"
            "The validated ctypes path (examples/GScodes.py) remains the production interface."
        )
    # --- M1 implementation outline (not code) ---
    # 1. Load MWTransferArrXLA.so via jax.extend.ffi.load_plugin(plugin_path)
    # 2. Call the registered custom-call target via jax.extend.ffi.ffi_call(
    #        "mw_approx_batch_rl",
    #        result_shape_dtypes=(
    #            jax.ShapeDtypeStruct((B, 7, nfreq), jnp.float64),   # rl
    #            jax.ShapeDtypeStruct((nfreq,), jnp.float64),        # freq_hz
    #        ),
    #        params,
    #        nfreq=nfreq, nu0_hz=nu0_hz, dlog10_nu=dlog10_nu,
    #        npoints=npoints, q_on=int(q_on), d_sun_au=d_sun_au,
    #    )
    # 3. Return (rl, freq_hz)
    raise NotImplementedError  # remove once M1 is complete
