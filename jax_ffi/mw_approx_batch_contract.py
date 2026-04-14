"""Contract helpers for the staged JAX wrapper around the validated batch path.

These helpers intentionally mirror the currently supported narrow workflow:
  - analytical PLW + ISO
  - Nz=1
  - approximate GS only
  - FP64 correctness path

The goal here is to freeze the parameter and observable contract before the
XLA FFI layer exists. This module does not depend on JAX so it can be validated
against the current supported-path oracle on any machine that can load the
shared library.
"""

from __future__ import annotations

import numpy as np


ASEC2CM = 0.725e8
ARCSEC2_TO_SR = (np.pi / (180.0 * 3600.0)) ** 2
SPEED_OF_LIGHT_CGS = 2.99792458e10
K_BOLTZMANN_CGS = 1.380649e-16

LEGACY_AREA_ASEC2 = 1.625625
LEGACY_EMAX_MEV = 10.0
LEGACY_REDUCED_PARAM_DIM = 8
FULL_PHYSICAL_PARAM_DIM = 10

LEGACY_TARGET_FREQ_GHZ = np.array(
    [
        1.2558008e09,
        1.5798831e09,
        2.5500792e09,
        2.8738688e09,
        3.1975603e09,
        3.5210079e09,
        3.8442601e09,
        4.1673172e09,
        4.4923274e09,
        4.8173373e09,
        5.1423473e09,
        5.4673577e09,
        5.7923676e09,
        6.1173780e09,
        6.4423880e09,
        6.7673979e09,
        7.0924083e09,
        7.4174182e09,
        7.7424282e09,
        8.0674386e09,
        8.3924485e09,
        8.7174584e09,
        9.0424689e09,
        9.3674793e09,
        9.6924887e09,
        1.0017499e10,
        1.0342510e10,
        1.0667519e10,
        1.0992529e10,
        1.1317540e10,
        1.1642549e10,
        1.1967560e10,
        1.2292570e10,
        1.2617580e10,
        1.2942590e10,
        1.3267600e10,
        1.3592611e10,
        1.3917620e10,
        1.4242631e10,
        1.4567641e10,
        1.4892650e10,
        1.5217661e10,
        1.5542671e10,
        1.5867681e10,
        1.6192691e10,
        1.6517702e10,
        1.6842711e10,
        1.7167721e10,
        1.7492732e10,
        1.7817741e10,
    ],
    dtype=np.float64,
) / 1.0e9


def _as_float64_2d(values, *, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    return array


def expand_legacy_or_full_parameter_batch(params):
    """Expand either the legacy 8D fit order or the canonical 10D order to 10D.

    Canonical 10D physical order:
      [area_cm2, depth_cm, bmag_g, temperature_k, thermal_density_cm3,
       nonthermal_density_cm3, delta, theta_deg, emin_mev, emax_mev]

    Legacy reduced 8D order:
      [depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV]
    """

    values = _as_float64_2d(params, name="params")

    if values.shape[1] == LEGACY_REDUCED_PARAM_DIM:
        expanded = np.empty((values.shape[0], FULL_PHYSICAL_PARAM_DIM), dtype=np.float64)
        expanded[:, 0] = LEGACY_AREA_ASEC2 * (ASEC2CM ** 2)
        expanded[:, 1] = values[:, 0] * ASEC2CM
        expanded[:, 2] = values[:, 1] * 100.0
        expanded[:, 3] = values[:, 2] * 1.0e6
        expanded[:, 4] = np.power(10.0, values[:, 3])
        expanded[:, 5] = np.power(10.0, values[:, 4])
        expanded[:, 6] = values[:, 5]
        expanded[:, 7] = values[:, 6]
        expanded[:, 8] = values[:, 7] / 1.0e3
        expanded[:, 9] = LEGACY_EMAX_MEV
        return expanded

    if values.shape[1] == FULL_PHYSICAL_PARAM_DIM:
        return values.copy()

    raise ValueError(
        "Expected params to use either the legacy 8D order "
        "[depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV] "
        "or the canonical 10D physical order "
        "[area_cm2, depth_cm, bmag_g, temperature_k, thermal_density_cm3, "
        "nonthermal_density_cm3, delta, theta_deg, emin_mev, emax_mev]."
    )


def validate_supported_parameter_batch(params):
    """Validate the canonical or legacy parameter batch for the supported path."""

    full = expand_legacy_or_full_parameter_batch(params)

    if np.any(full[:, 0] <= 0.0):
        raise ValueError("area_cm2 must be positive")
    if np.any(full[:, 1] <= 0.0):
        raise ValueError("depth_cm must be positive")
    if np.any(full[:, 2] <= 0.0):
        raise ValueError("bmag_g must be positive")
    if np.any(full[:, 3] <= 0.0):
        raise ValueError("temperature_k must be positive")
    if np.any(full[:, 4] <= 0.0):
        raise ValueError("thermal_density_cm3 must be positive")
    if np.any(full[:, 5] <= 0.0):
        raise ValueError("nonthermal_density_cm3 must be positive")
    if np.any(full[:, 8] <= 0.0):
        raise ValueError("emin_mev must be positive")
    if np.any(full[:, 9] <= full[:, 8]):
        raise ValueError("emax_mev must be greater than emin_mev")

    return full


def full_parameter_batch_from_wrapper_batch(batch, *, active_batch_size=None):
    """Extract the canonical 10D physical parameter batch from a wrapper batch."""

    active = getattr(batch, "_active_batch_size", batch.batch_size)
    if active_batch_size is not None:
        active = int(active_batch_size)
    if active < 0 or active > int(batch.batch_size):
        raise ValueError(f"active_batch_size must be in [0, {batch.batch_size}], got {active}")

    return np.column_stack(
        [
            np.asarray(batch.area_cm2[:active], dtype=np.float64),
            np.asarray(batch.depth_cm[:active], dtype=np.float64),
            np.asarray(batch.bmag_g[:active], dtype=np.float64),
            np.asarray(batch.temperature_k[:active], dtype=np.float64),
            np.asarray(batch.thermal_density_cm3[:active], dtype=np.float64),
            np.asarray(batch.nonthermal_density_cm3[:active], dtype=np.float64),
            np.asarray(batch.delta[:active], dtype=np.float64),
            np.asarray(batch.theta_deg[:active], dtype=np.float64),
            np.asarray(batch.emin_mev[:active], dtype=np.float64),
            np.asarray(batch.emax_mev[:active], dtype=np.float64),
        ]
    )


def legacy_reduced_parameter_batch_from_full(full_params):
    """Convert canonical 10D physical parameters to the legacy reduced 8D fit order."""

    full = validate_supported_parameter_batch(full_params)

    return np.column_stack(
        [
            full[:, 1] / ASEC2CM,
            full[:, 2] / 100.0,
            full[:, 3] / 1.0e6,
            np.log10(full[:, 4]),
            np.log10(full[:, 5]),
            full[:, 6],
            full[:, 7],
            full[:, 8] * 1.0e3,
        ]
    )


def legacy_reduced_parameter_batch_from_wrapper_batch(batch, *, active_batch_size=None):
    """Extract the legacy reduced 8D fit order from a wrapper batch."""

    return legacy_reduced_parameter_batch_from_full(
        full_parameter_batch_from_wrapper_batch(batch, active_batch_size=active_batch_size)
    )


def fill_analytical_powerlaw_iso_batch(batch, params):
    """Fill an `AnalyticalPowerLawIsoBatch`-style object in place from 8D or 10D."""

    full = validate_supported_parameter_batch(params)
    if full.shape[0] > int(batch.batch_size):
        raise ValueError(f"batch can hold {batch.batch_size} rows, got {full.shape[0]}")

    active = full.shape[0]
    batch._active_batch_size = active

    np.copyto(batch.area_cm2[:active], full[:, 0])
    np.copyto(batch.depth_cm[:active], full[:, 1])
    np.copyto(batch.bmag_g[:active], full[:, 2])
    np.copyto(batch.temperature_k[:active], full[:, 3])
    np.copyto(batch.thermal_density_cm3[:active], full[:, 4])
    np.copyto(batch.nonthermal_density_cm3[:active], full[:, 5])
    np.copyto(batch.delta[:active], full[:, 6])
    np.copyto(batch.theta_deg[:active], full[:, 7])
    np.copyto(batch.emin_mev[:active], full[:, 8])
    np.copyto(batch.emax_mev[:active], full[:, 9])

    return batch


def interpolate_positive_spectrum(native_freq_hz, flux_native, target_freq_hz):
    """Interpolate a strictly positive spectrum in log-log space."""

    native_freq_hz = np.asarray(native_freq_hz, dtype=np.float64).reshape(-1)
    flux_native = np.asarray(flux_native, dtype=np.float64).reshape(-1)
    target_freq_hz = np.asarray(target_freq_hz, dtype=np.float64).reshape(-1)
    if native_freq_hz.shape != flux_native.shape:
        raise ValueError("native_freq_hz and flux_native must have the same shape")

    return np.power(
        10.0,
        np.interp(
            np.log10(target_freq_hz),
            np.log10(native_freq_hz),
            np.log10(np.maximum(flux_native, 1.0e-300)),
        ),
    )


def flux_sfu_to_tb(target_freq_hz, flux_sfu, area_asec2):
    """Convert flux density in sfu to brightness temperature."""

    target_freq_hz = np.asarray(target_freq_hz, dtype=np.float64).reshape(-1)
    flux_sfu = np.asarray(flux_sfu, dtype=np.float64)
    area_asec2 = np.asarray(area_asec2, dtype=np.float64)

    if flux_sfu.ndim == 1:
        if area_asec2.ndim != 0:
            raise ValueError("area_asec2 must be scalar when flux_sfu is 1D")
        factor = (SPEED_OF_LIGHT_CGS ** 2) / (
            2.0 * K_BOLTZMANN_CGS * np.square(target_freq_hz) * float(area_asec2) * ARCSEC2_TO_SR
        )
        return flux_sfu * 1.0e-19 * factor

    if flux_sfu.ndim == 2:
        if flux_sfu.shape[1] != target_freq_hz.size:
            raise ValueError(
                f"flux_sfu must have shape [B,{target_freq_hz.size}] for 2D input, got {flux_sfu.shape}"
            )
        if area_asec2.ndim == 0:
            area_asec2 = np.full((flux_sfu.shape[0],), float(area_asec2), dtype=np.float64)
        if area_asec2.shape != (flux_sfu.shape[0],):
            raise ValueError(
                f"area_asec2 must have shape [{flux_sfu.shape[0]}] for 2D flux input, got {area_asec2.shape}"
            )
        factor = (SPEED_OF_LIGHT_CGS ** 2) / (
            2.0
            * K_BOLTZMANN_CGS
            * np.square(target_freq_hz[None, :])
            * area_asec2[:, None]
            * ARCSEC2_TO_SR
        )
        return flux_sfu * 1.0e-19 * factor

    raise ValueError(f"flux_sfu must be 1D or 2D, got shape {flux_sfu.shape}")


def total_flux_sfu_from_result(result):
    """Return total intensity `I_L + I_R` from a batch wrapper result as [B, F]."""

    rl = np.asarray(result.rl, dtype=np.float64)
    if rl.ndim != 3 or rl.shape[0] != 7:
        raise ValueError(f"Expected result.rl to have shape [7,nfreq,batch], got {rl.shape}")
    return np.asarray(rl[5] + rl[6], dtype=np.float64).T


def extract_legacy_spectrum_from_result(
    result,
    *,
    area_cm2,
    target_freq_ghz=LEGACY_TARGET_FREQ_GHZ,
    spec_in_tb=True,
):
    """Convert wrapper RL output to the legacy fitting observable `[B, F]`.

    This mirrors the current supported fitting definition:
      - total intensity `I_L + I_R`
      - interpolation onto the legacy target frequency grid
      - optional conversion to brightness temperature
    """

    rl = np.asarray(result.rl, dtype=np.float64)
    status = np.asarray(result.status, dtype=np.int32).reshape(-1)
    native_freq_hz = np.asarray(result.native_freq_hz, dtype=np.float64).reshape(-1)
    target_freq_hz = np.asarray(target_freq_ghz, dtype=np.float64).reshape(-1) * 1.0e9
    area_cm2 = np.asarray(area_cm2, dtype=np.float64).reshape(-1)

    if rl.ndim != 3 or rl.shape[0] != 7:
        raise ValueError(f"Expected result.rl to have shape [7,nfreq,batch], got {rl.shape}")

    batch_size = rl.shape[2]
    if area_cm2.shape[0] < batch_size:
        raise ValueError(f"area_cm2 must provide at least {batch_size} entries, got {area_cm2.shape[0]}")

    spectra = np.empty((batch_size, target_freq_hz.size), dtype=np.float64)
    total_flux = rl[5] + rl[6]
    area_asec2 = area_cm2[:batch_size] / (ASEC2CM ** 2)

    for index in range(batch_size):
        if status[index] != 0 or not np.any(total_flux[:, index]):
            spectra[index, :] = 1.0e4 if spec_in_tb else 1.0e-11
            continue

        interp_flux = interpolate_positive_spectrum(
            native_freq_hz=native_freq_hz,
            flux_native=total_flux[:, index],
            target_freq_hz=target_freq_hz,
        )
        if spec_in_tb:
            spectra[index, :] = flux_sfu_to_tb(target_freq_hz, interp_flux, area_asec2[index])
        else:
            spectra[index, :] = interp_flux

    return spectra


def extract_legacy_spectrum_from_batch_result(
    batch,
    result,
    *,
    target_freq_ghz=LEGACY_TARGET_FREQ_GHZ,
    spec_in_tb=True,
):
    """Convert wrapper RL output to the legacy observable using a wrapper batch."""

    return extract_legacy_spectrum_from_result(
        result,
        area_cm2=np.asarray(batch.area_cm2, dtype=np.float64),
        target_freq_ghz=target_freq_ghz,
        spec_in_tb=spec_in_tb,
    )
