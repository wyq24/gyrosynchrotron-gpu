from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import GScodes


def default_library_path(libname=None):
    return GScodes.resolve_library_path(libname, prefer_source=True, require_local_source=True)


def _cycled_values(values, batch_size, stride, offset=0):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")
    indices = (offset + stride * np.arange(batch_size, dtype=np.int64)) % values.size
    return values[indices]


def build_supported_workload(name, batch_size):
    workload = str(name).lower()
    if workload == "real-sweep":
        return build_real_wrapper_sweep(batch_size)
    if workload == "stress-sweep":
        return build_supported_stress_sweep(batch_size)
    raise ValueError(f"Unsupported workload {name!r}")


def build_real_wrapper_workload(batch_size=1, **overrides):
    defaults = {
        "area_asec2": 4.0,
        "depth_asec": 5.0,
        "bmag_100g": 2.0,
        "temperature_mk": 1.0,
        "log_nth": 10.0,
        "log_nnth": 5.0,
        "delta": 4.0,
        "theta_deg": 45.0,
        "emin_kev": 10.0,
        "emax_mev": 10.0,
        "nfreq": 100,
        "nu0_hz": 0.8e9,
        "dlog10_nu": 0.02,
        "nu_cr_factor": 0.0,
        "nu_wh_factor": 0.0,
    }
    defaults.update(overrides)
    return GScodes.build_wrapper_powerlaw_iso_batch(batch_size=batch_size, **defaults)


def build_real_wrapper_sweep(batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    frac = np.linspace(0.0, 1.0, batch_size, dtype=np.float64)
    return GScodes.build_wrapper_powerlaw_iso_batch(
        batch_size=batch_size,
        area_asec2=8.0 + 24.0 * frac,
        depth_asec=3.0 + 12.0 * frac,
        bmag_100g=1.5 + 3.5 * frac,
        temperature_mk=0.5 + 9.5 * frac,
        log_nth=9.2 + 1.0 * frac,
        log_nnth=4.5 + 2.0 * frac,
        delta=2.5 + 3.0 * frac,
        theta_deg=25.0 + 55.0 * frac,
        emin_kev=8.0 + 22.0 * frac,
        emax_mev=5.0 + 15.0 * frac,
        nfreq=100,
        nu0_hz=0.8e9,
        dlog10_nu=0.02,
        nu_cr_factor=0.0,
        nu_wh_factor=0.0,
    )


def build_supported_stress_sweep(batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return GScodes.build_wrapper_powerlaw_iso_batch(
        batch_size=batch_size,
        area_asec2=_cycled_values([2.0, 4.0, 8.0, 16.0, 32.0, 48.0], batch_size, 5),
        depth_asec=_cycled_values([1.0, 2.5, 5.0, 10.0, 20.0], batch_size, 3, offset=1),
        bmag_100g=_cycled_values([0.8, 1.2, 2.0, 3.5, 5.0, 7.5], batch_size, 5, offset=2),
        temperature_mk=_cycled_values([0.2, 0.5, 1.0, 3.0, 10.0, 20.0], batch_size, 7, offset=1),
        log_nth=_cycled_values([8.5, 9.0, 9.5, 10.0, 10.5, 11.0], batch_size, 5, offset=3),
        log_nnth=_cycled_values([3.5, 4.0, 4.5, 5.5, 6.5, 7.0], batch_size, 7, offset=2),
        delta=_cycled_values([2.1, 2.5, 3.0, 4.0, 5.0, 6.0], batch_size, 5, offset=4),
        theta_deg=_cycled_values([10.0, 25.0, 45.0, 65.0, 80.0, 87.0], batch_size, 7, offset=5),
        emin_kev=_cycled_values([5.0, 8.0, 10.0, 20.0, 40.0, 80.0], batch_size, 5, offset=1),
        emax_mev=_cycled_values([1.0, 2.0, 5.0, 10.0, 20.0, 40.0], batch_size, 7, offset=4),
        nfreq=100,
        nu0_hz=0.8e9,
        dlog10_nu=0.02,
        nu_cr_factor=0.0,
        nu_wh_factor=0.0,
    )


def build_repo_array_baseline():
    nsteps = 30
    nfreq = 100
    ne = 15
    nmu = 15

    lparms = np.zeros(11, dtype=np.int32)
    lparms[0] = nsteps
    lparms[1] = nfreq
    lparms[2] = ne
    lparms[3] = nmu

    rparms = np.zeros(5, dtype=np.float64)
    rparms[0] = 1e20
    rparms[1] = 1e9
    rparms[2] = 0.02
    rparms[3] = 12.0
    rparms[4] = 12.0

    parm_local = np.zeros(24, dtype=np.float64)
    parm_local[0] = 1e10 / nsteps
    parm_local[1] = 3e7
    parm_local[2] = 3e9
    parm_local[3] = 180.0

    parms = np.zeros((24, nsteps), dtype=np.float64, order="F")
    for idx in range(nsteps):
        parms[:, idx] = parm_local
        parms[4, idx] = 50.0 + 30.0 * idx / (nsteps - 1)

    e_arr = np.logspace(np.log10(0.1), np.log10(10.0), ne, dtype=np.float64)
    mu_arr = np.linspace(-1.0, 1.0, nmu, dtype=np.float64)
    f_arr = np.zeros((ne, nmu, nsteps), dtype=np.float64, order="F")

    dmu_c = 0.2
    delta = 4.0
    n_b = 1e6
    mu_c = np.cos(np.pi / 2)
    a0 = n_b / (2.0 * np.pi) * (delta - 1.0) / (0.1 ** (1.0 - delta) - 10.0 ** (1.0 - delta))
    b0 = 0.5 / (mu_c + dmu_c * np.sqrt(np.pi) / 2.0)

    f0 = np.zeros((ne, nmu), dtype=np.float64)
    for i in range(ne):
        for j in range(nmu):
            amu = abs(mu_arr[j])
            if amu < mu_c:
                f0[i, j] = a0 * b0 * e_arr[i] ** (-delta)
            else:
                f0[i, j] = a0 * b0 * e_arr[i] ** (-delta) * np.exp(-((amu - mu_c) / dmu_c) ** 2)
    for idx in range(nsteps):
        f_arr[:, :, idx] = f0

    rl = np.zeros((7, nfreq), dtype=np.float64, order="F")
    native_freq_hz = GScodes.frequency_grid_hz(nfreq, rparms[1], rparms[2])
    nu_b = 2.799249110e6 * parm_local[3]
    above_fraction = float(np.mean(native_freq_hz > (rparms[3] * nu_b)))

    return {
        "lparms": lparms,
        "rparms": rparms,
        "parms": parms,
        "e_arr": e_arr,
        "mu_arr": mu_arr,
        "f_arr": f_arr,
        "rl": rl,
        "nfreq": nfreq,
        "nsteps": nsteps,
        "ne": ne,
        "nmu": nmu,
        "above_nu_cr_fraction": above_fraction,
    }
