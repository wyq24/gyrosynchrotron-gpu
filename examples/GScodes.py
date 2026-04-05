import ctypes
from dataclasses import dataclass
from pathlib import Path
import platform
import time

import numpy as np
from numpy.ctypeslib import ndpointer

MW_BATCH_BACKEND_CPU = 0
MW_BATCH_BACKEND_CUDA = 1

MW_BATCH_PRECISION_FP64 = 0
MW_BATCH_PRECISION_FP32 = 1

_MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}

_APPROX_BATCH_ERRORS = {
    0: "MW_BATCH_OK",
    1000: "MW_BATCH_ERR_INVALID_ARGUMENT",
    1001: "MW_BATCH_ERR_UNSUPPORTED_BACKEND",
    1002: "MW_BATCH_ERR_CUDA_UNAVAILABLE",
    1003: "MW_BATCH_ERR_CUDA_RUNTIME",
}


class _MWApproxBatchTimingStruct(ctypes.Structure):
    _fields_ = [
        ("total_seconds", ctypes.c_double),
        ("setup_seconds", ctypes.c_double),
        ("h2d_seconds", ctypes.c_double),
        ("device_alloc_seconds", ctypes.c_double),
        ("device_zero_seconds", ctypes.c_double),
        ("backend_compute_seconds", ctypes.c_double),
        ("sync_seconds", ctypes.c_double),
        ("d2h_seconds", ctypes.c_double),
        ("cleanup_seconds", ctypes.c_double),
    ]

def initGET_MW(libname):
    _intp=ndpointer(dtype=ctypes.c_int32, flags='F')
    _doublep=ndpointer(dtype=ctypes.c_double, flags='F')
    
    libc_mw=ctypes.CDLL(libname)
    mwfunc=libc_mw.pyGET_MW
    mwfunc.argtypes=[_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype=ctypes.c_int

    return mwfunc


def initMW_Approx_Batch(libname):
    _int1 = ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    _double1 = ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    _double2f = ndpointer(dtype=np.float64, ndim=2, flags="F_CONTIGUOUS")

    libc_mw = ctypes.CDLL(libname)
    mwfunc = libc_mw.pyMW_Approx_Batch
    mwfunc.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        _double1,
        _double1,
        _double1,
        _double1,
        _double1,
        _double1,
        _double1,
        _double1,
        _double1,
        _double1,
        _int1,
        _double1,
        _double2f,
        _double2f,
        _double2f,
        _double2f,
    ]
    mwfunc.restype = ctypes.c_int
    return mwfunc


def initMW_Approx_Batch_CudaAvailable(libname):
    libc_mw = ctypes.CDLL(libname)
    func = libc_mw.pyMW_Approx_Batch_CudaAvailable
    func.argtypes = []
    func.restype = ctypes.c_int
    return func


def initMW_Approx_Batch_TimingReset(libname):
    libc_mw = ctypes.CDLL(libname)
    func = libc_mw.pyMW_Approx_Batch_TimingReset
    func.argtypes = []
    func.restype = None
    return func


def initMW_Approx_Batch_TimingGet(libname):
    libc_mw = ctypes.CDLL(libname)
    func = libc_mw.pyMW_Approx_Batch_TimingGet
    func.argtypes = [ctypes.POINTER(_MWApproxBatchTimingStruct)]
    func.restype = None
    return func


def _detect_binary_format(path):
    path = Path(path)
    try:
        with path.open("rb") as handle:
            header = handle.read(4)
    except OSError:
        return "unknown"

    if header.startswith(b"\x7fELF"):
        return "elf"
    if header.startswith(b"MZ"):
        return "pe"
    if header in _MACHO_MAGICS:
        return "macho"
    return "unknown"


def _expected_binary_format():
    system = platform.system()
    if system == "Linux":
        return "elf"
    if system == "Darwin":
        return "macho"
    if system == "Windows":
        return "pe"
    return None


def inspect_shared_library(path):
    candidate = Path(path).expanduser().resolve()
    binary_format = _detect_binary_format(candidate) if candidate.exists() else "missing"
    expected = _expected_binary_format()
    compatible = candidate.exists() and (expected is None or binary_format == expected)
    return {
        "path": str(candidate),
        "exists": candidate.exists(),
        "binary_format": binary_format,
        "expected_binary_format": expected,
        "compatible": compatible,
    }


def _validated_library_path(candidate, *, context):
    info = inspect_shared_library(candidate)
    if not info["exists"]:
        raise FileNotFoundError(f"{context}: shared library was not found at {info['path']}")
    if not info["compatible"]:
        expected = info["expected_binary_format"] or "the current host platform"
        raise RuntimeError(
            f"{context}: shared library at {info['path']} has binary format {info['binary_format']!r}, "
            f"but this host expects {expected!r}. Rebuild MWTransferArr.so locally before running."
        )
    return info["path"]


def resolve_library_path(libname=None, prefer_source=True, require_local_source=False):
    if libname:
        return _validated_library_path(Path(libname), context="Explicit MWTransferArr path")

    repo_root = Path(__file__).resolve().parents[1]
    source_candidate = repo_root / "source" / "MWTransferArr.so"

    if require_local_source:
        return _validated_library_path(
            source_candidate,
            context="Local source build required for this workflow",
        )

    candidates = []
    if prefer_source:
        candidates.append(source_candidate)
    candidates.extend(
        [
            repo_root / "binaries" / "MWTransferArr.so",
            repo_root / "binaries" / "MWTransferArr64.dll",
            repo_root / "binaries" / "MWTransferArr32.dll",
        ]
    )
    if not prefer_source:
        candidates.append(source_candidate)

    mismatches = []

    for candidate in candidates:
        if not candidate.exists():
            continue
        info = inspect_shared_library(candidate)
        if info["compatible"]:
            return info["path"]
        mismatches.append(info)

    if mismatches:
        details = ", ".join(f"{item['path']} ({item['binary_format']})" for item in mismatches)
        raise RuntimeError(
            "Found MWTransferArr shared libraries, but none are compatible with this host platform: "
            f"{details}"
        )
    raise FileNotFoundError("Could not locate a MWTransferArr shared library")


def approx_batch_error_name(code):
    return _APPROX_BATCH_ERRORS.get(int(code), f"MW_BATCH_ERR_UNKNOWN_{int(code)}")


def frequency_grid_hz(nfreq, nu0_hz, dlog10_nu):
    return np.asfortranarray(nu0_hz * np.power(10.0, dlog10_nu * np.arange(nfreq, dtype=np.float64)))


def _as_batch_array(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _broadcast_or_check(values, batch_size, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(batch_size, float(arr), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array")
    if arr.size != batch_size:
        raise ValueError(f"{name} must have size {batch_size}, got {arr.size}")
    return arr.astype(np.float64, copy=False)


@dataclass
class AnalyticalPowerLawIsoBatch:
    area_cm2: np.ndarray
    depth_cm: np.ndarray
    bmag_g: np.ndarray
    temperature_k: np.ndarray
    thermal_density_cm3: np.ndarray
    nonthermal_density_cm3: np.ndarray
    delta: np.ndarray
    theta_deg: np.ndarray
    emin_mev: np.ndarray
    emax_mev: np.ndarray
    nfreq: int = 100
    nu0_hz: float = 0.8e9
    dlog10_nu: float = 0.02
    nu_cr_factor: float = 0.0
    nu_wh_factor: float = 0.0

    def __post_init__(self):
        names = [
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
        ]
        arrays = []
        for name in names:
            arr = _as_batch_array(getattr(self, name), name)
            setattr(self, name, arr)
            arrays.append(arr)
        batch_size = arrays[0].size
        for name, arr in zip(names, arrays):
            if arr.size != batch_size:
                raise ValueError(f"{name} must have size {batch_size}, got {arr.size}")
        if self.nfreq <= 0:
            raise ValueError("nfreq must be positive")
        if self.nu0_hz <= 0.0:
            raise ValueError("nu0_hz must be positive")
        if np.any(self.emin_mev <= 0.0) or np.any(self.emax_mev <= 0.0):
            raise ValueError("Energy cutoffs must be positive")
        if np.any(self.emax_mev <= self.emin_mev):
            raise ValueError("emax_mev must be greater than emin_mev for every spectrum")

    @property
    def batch_size(self):
        return self.area_cm2.size

    @property
    def native_freq_hz(self):
        return frequency_grid_hz(self.nfreq, self.nu0_hz, self.dlog10_nu)

    def fraction_above_nu_cr(self):
        nu_b = 2.799249110e6 * self.bmag_g
        native_freq = self.native_freq_hz[:, None]
        nu_cr = self.nu_cr_factor * nu_b[None, :]
        return np.mean(native_freq > nu_cr, axis=0)

    def fraction_above_nu_wh(self):
        nu_b = 2.799249110e6 * self.bmag_g
        native_freq = self.native_freq_hz[:, None]
        nu_wh = self.nu_wh_factor * nu_b[None, :]
        return np.mean(native_freq > nu_wh, axis=0)


@dataclass
class BatchRunResult:
    status: np.ndarray
    rl: np.ndarray
    native_freq_hz: np.ndarray

    @property
    def total_intensity(self):
        return self.rl[5] + self.rl[6]


@dataclass
class LocalBatchKernelResult:
    status: np.ndarray
    native_freq_hz: np.ndarray
    jx: np.ndarray
    kx: np.ndarray
    jo: np.ndarray
    ko: np.ndarray
    backend: str
    precision: str

    @property
    def output_threads(self):
        return self.native_freq_hz.size * self.status.size * 2


@dataclass
class NativeBatchTiming:
    total_seconds: float
    setup_seconds: float
    h2d_seconds: float
    device_alloc_seconds: float
    device_zero_seconds: float
    backend_compute_seconds: float
    sync_seconds: float
    d2h_seconds: float
    cleanup_seconds: float

    @property
    def internal_overhead_seconds(self):
        accounted = (
            self.setup_seconds
            + self.h2d_seconds
            + self.device_alloc_seconds
            + self.device_zero_seconds
            + self.sync_seconds
            + self.d2h_seconds
            + self.cleanup_seconds
        )
        return max(self.total_seconds - accounted, 0.0)


@dataclass
class NativeBatchProfile:
    binding_seconds: float
    packing_seconds: float
    native_call_seconds: float
    call_boundary_seconds: float
    postprocess_seconds: float
    total_seconds: float
    native_timing: NativeBatchTiming


def build_wrapper_powerlaw_iso_batch(
    batch_size=1,
    *,
    area_asec2=4.0,
    depth_asec=5.0,
    bmag_100g=2.0,
    temperature_mk=1.0,
    log_nth=10.0,
    log_nnth=5.0,
    delta=4.0,
    theta_deg=45.0,
    emin_kev=10.0,
    emax_mev=10.0,
    nfreq=100,
    nu0_hz=0.8e9,
    dlog10_nu=0.02,
    nu_cr_factor=0.0,
    nu_wh_factor=0.0,
):
    asec2cm = 0.725e8
    return AnalyticalPowerLawIsoBatch(
        area_cm2=_broadcast_or_check(area_asec2, batch_size, "area_asec2") * (asec2cm ** 2),
        depth_cm=_broadcast_or_check(depth_asec, batch_size, "depth_asec") * asec2cm,
        bmag_g=_broadcast_or_check(bmag_100g, batch_size, "bmag_100g") * 100.0,
        temperature_k=_broadcast_or_check(temperature_mk, batch_size, "temperature_mk") * 1.0e6,
        thermal_density_cm3=np.power(10.0, _broadcast_or_check(log_nth, batch_size, "log_nth")),
        nonthermal_density_cm3=np.power(10.0, _broadcast_or_check(log_nnth, batch_size, "log_nnth")),
        delta=_broadcast_or_check(delta, batch_size, "delta"),
        theta_deg=_broadcast_or_check(theta_deg, batch_size, "theta_deg"),
        emin_mev=_broadcast_or_check(emin_kev, batch_size, "emin_kev") / 1.0e3,
        emax_mev=_broadcast_or_check(emax_mev, batch_size, "emax_mev"),
        nfreq=nfreq,
        nu0_hz=nu0_hz,
        dlog10_nu=dlog10_nu,
        nu_cr_factor=nu_cr_factor,
        nu_wh_factor=nu_wh_factor,
    )


def _build_powerlaw_iso_single_inputs(batch, index):
    lparms = np.zeros(11, dtype=np.int32)
    lparms[0] = 1
    lparms[1] = batch.nfreq

    rparms = np.zeros(5, dtype=np.float64)
    rparms[0] = batch.area_cm2[index]
    rparms[1] = batch.nu0_hz
    rparms[2] = batch.dlog10_nu
    rparms[3] = batch.nu_cr_factor
    rparms[4] = batch.nu_wh_factor

    parms = np.zeros((24, 1), dtype=np.float64, order="F")
    parms[0, 0] = batch.depth_cm[index]
    parms[1, 0] = batch.temperature_k[index]
    parms[2, 0] = batch.thermal_density_cm3[index]
    parms[3, 0] = batch.bmag_g[index]
    parms[4, 0] = batch.theta_deg[index]
    parms[6, 0] = 3.0
    parms[7, 0] = batch.nonthermal_density_cm3[index]
    parms[9, 0] = batch.emin_mev[index]
    parms[10, 0] = batch.emax_mev[index]
    parms[12, 0] = batch.delta[index]
    parms[14, 0] = 0.0
    parms[15, 0] = 90.0
    parms[16, 0] = 0.2

    dummy = np.zeros(1, dtype=np.float64)
    rl = np.zeros((7, batch.nfreq), dtype=np.float64, order="F")

    return lparms, rparms, parms, dummy, rl


def _native_backend_code(backend):
    name = str(backend).lower()
    if name == "cpu":
        return MW_BATCH_BACKEND_CPU
    if name == "cuda":
        return MW_BATCH_BACKEND_CUDA
    raise ValueError(f"Unsupported backend {backend!r}")


def _native_precision_code(precision):
    name = str(precision).lower()
    if name == "fp64":
        return MW_BATCH_PRECISION_FP64
    if name == "fp32":
        return MW_BATCH_PRECISION_FP32
    raise ValueError(f"Unsupported precision {precision!r}")


def cuda_available(libname=None):
    libname = resolve_library_path(libname)
    return bool(initMW_Approx_Batch_CudaAvailable(libname)())


def _batch_array_view(values):
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def get_last_batch_timing(libname=None):
    libname = resolve_library_path(libname)
    timing = _MWApproxBatchTimingStruct()
    initMW_Approx_Batch_TimingGet(libname)(ctypes.byref(timing))
    return NativeBatchTiming(
        total_seconds=float(timing.total_seconds),
        setup_seconds=float(timing.setup_seconds),
        h2d_seconds=float(timing.h2d_seconds),
        device_alloc_seconds=float(timing.device_alloc_seconds),
        device_zero_seconds=float(timing.device_zero_seconds),
        backend_compute_seconds=float(timing.backend_compute_seconds),
        sync_seconds=float(timing.sync_seconds),
        d2h_seconds=float(timing.d2h_seconds),
        cleanup_seconds=float(timing.cleanup_seconds),
    )


def run_powerlaw_iso_batch_native(libname, batch, *, backend="cpu", precision="fp64", npoints=16, q_on=True):
    if npoints <= 0:
        raise ValueError("npoints must be positive for the approximate batch backend")

    libname = resolve_library_path(libname)
    backend_name = str(backend).lower()
    if backend_name == "cuda" and not cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend requested for {libname}, but pyMW_Approx_Batch_CudaAvailable returned 0. "
            "Build with CUDA=1 on a machine with accessible NVIDIA runtime support."
        )
    batch_func = initMW_Approx_Batch(libname)

    status = np.zeros(batch.batch_size, dtype=np.int32)
    native_freq_hz = np.zeros(batch.nfreq, dtype=np.float64)
    jx = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    kx = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    jo = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    ko = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")

    res = int(
        batch_func(
            _native_backend_code(backend_name),
            _native_precision_code(precision),
            batch.batch_size,
            batch.nfreq,
            int(npoints),
            int(bool(q_on)),
            float(batch.nu0_hz),
            float(batch.dlog10_nu),
            _batch_array_view(batch.area_cm2),
            _batch_array_view(batch.depth_cm),
            _batch_array_view(batch.bmag_g),
            _batch_array_view(batch.temperature_k),
            _batch_array_view(batch.thermal_density_cm3),
            _batch_array_view(batch.nonthermal_density_cm3),
            _batch_array_view(batch.delta),
            _batch_array_view(batch.theta_deg),
            _batch_array_view(batch.emin_mev),
            _batch_array_view(batch.emax_mev),
            status,
            native_freq_hz,
            jx,
            kx,
            jo,
            ko,
        )
    )
    if res != 0:
        raise RuntimeError(f"pyMW_Approx_Batch failed with status {res} ({approx_batch_error_name(res)})")

    return LocalBatchKernelResult(
        status=status,
        native_freq_hz=native_freq_hz,
        jx=jx,
        kx=kx,
        jo=jo,
        ko=ko,
        backend=backend_name,
        precision=str(precision).lower(),
    )


def run_powerlaw_iso_batch_native_profiled(libname, batch, *, backend="cpu", precision="fp64", npoints=16, q_on=True, include_postprocess=False):
    total_start = time.perf_counter()
    bind_start = total_start

    if npoints <= 0:
        raise ValueError("npoints must be positive for the approximate batch backend")

    libname = resolve_library_path(libname)
    backend_name = str(backend).lower()
    if backend_name == "cuda" and not cuda_available(libname):
        raise RuntimeError(
            f"CUDA backend requested for {libname}, but pyMW_Approx_Batch_CudaAvailable returned 0. "
            "Build with CUDA=1 on a machine with accessible NVIDIA runtime support."
        )
    batch_func = initMW_Approx_Batch(libname)
    timing_reset = initMW_Approx_Batch_TimingReset(libname)
    binding_seconds = time.perf_counter() - bind_start

    pack_start = time.perf_counter()
    status = np.zeros(batch.batch_size, dtype=np.int32)
    native_freq_hz = np.zeros(batch.nfreq, dtype=np.float64)
    jx = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    kx = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    jo = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    ko = np.zeros((batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    area_cm2 = _batch_array_view(batch.area_cm2)
    depth_cm = _batch_array_view(batch.depth_cm)
    bmag_g = _batch_array_view(batch.bmag_g)
    temperature_k = _batch_array_view(batch.temperature_k)
    thermal_density_cm3 = _batch_array_view(batch.thermal_density_cm3)
    nonthermal_density_cm3 = _batch_array_view(batch.nonthermal_density_cm3)
    delta_arr = _batch_array_view(batch.delta)
    theta_deg = _batch_array_view(batch.theta_deg)
    emin_mev = _batch_array_view(batch.emin_mev)
    emax_mev = _batch_array_view(batch.emax_mev)
    packing_seconds = time.perf_counter() - pack_start

    timing_reset()
    native_call_start = time.perf_counter()
    res = int(
        batch_func(
            _native_backend_code(backend_name),
            _native_precision_code(precision),
            batch.batch_size,
            batch.nfreq,
            int(npoints),
            int(bool(q_on)),
            float(batch.nu0_hz),
            float(batch.dlog10_nu),
            area_cm2,
            depth_cm,
            bmag_g,
            temperature_k,
            thermal_density_cm3,
            nonthermal_density_cm3,
            delta_arr,
            theta_deg,
            emin_mev,
            emax_mev,
            status,
            native_freq_hz,
            jx,
            kx,
            jo,
            ko,
        )
    )
    native_call_seconds = time.perf_counter() - native_call_start
    if res != 0:
        raise RuntimeError(f"pyMW_Approx_Batch failed with status {res} ({approx_batch_error_name(res)})")

    kernel_result = LocalBatchKernelResult(
        status=status,
        native_freq_hz=native_freq_hz,
        jx=jx,
        kx=kx,
        jo=jo,
        ko=ko,
        backend=backend_name,
        precision=str(precision).lower(),
    )
    native_timing = get_last_batch_timing(libname)

    postprocess_seconds = 0.0
    wrapper_result = None
    if include_postprocess:
        postprocess_start = time.perf_counter()
        wrapper_result = local_jk_to_single_voxel_rl(batch, kernel_result)
        postprocess_seconds = time.perf_counter() - postprocess_start

    total_seconds = time.perf_counter() - total_start
    profile = NativeBatchProfile(
        binding_seconds=binding_seconds,
        packing_seconds=packing_seconds,
        native_call_seconds=native_call_seconds,
        call_boundary_seconds=max(native_call_seconds - native_timing.total_seconds, 0.0),
        postprocess_seconds=postprocess_seconds,
        total_seconds=total_seconds,
        native_timing=native_timing,
    )
    return kernel_result, wrapper_result, profile


def local_jk_to_single_voxel_rl(batch, kernel_result, *, d_sun_au=1.0):
    if kernel_result.jx.shape != (batch.nfreq, batch.batch_size):
        raise ValueError("kernel_result arrays do not match the batch shape")

    sang = batch.area_cm2 / ((d_sun_au * 1.495978707e13) ** 2 * 1.0e-19)
    rl = np.zeros((7, batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    rl[0, :, :] = kernel_result.native_freq_hz[:, None] / 1.0e9

    theta_rad = np.deg2rad(batch.theta_deg)
    for idx in range(batch.batch_size):
        tau_o = -kernel_result.ko[:, idx] * batch.depth_cm[idx]
        tau_x = -kernel_result.kx[:, idx] * batch.depth_cm[idx]
        e_o = np.where(tau_o < 700.0, np.exp(tau_o), 0.0)
        e_x = np.where(tau_x < 700.0, np.exp(tau_x), 0.0)

        io = kernel_result.jo[:, idx] * batch.depth_cm[idx]
        ix = kernel_result.jx[:, idx] * batch.depth_cm[idx]
        valid_o = (kernel_result.ko[:, idx] != 0.0) & (tau_o <= 700.0)
        valid_x = (kernel_result.kx[:, idx] != 0.0) & (tau_x <= 700.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            delta_o = np.where((1.0 - e_o) != 0.0, 1.0 - e_o, -tau_o)
            delta_x = np.where((1.0 - e_x) != 0.0, 1.0 - e_x, -tau_x)
            io[valid_o] = kernel_result.jo[valid_o, idx] / kernel_result.ko[valid_o, idx] * delta_o[valid_o]
            ix[valid_x] = kernel_result.jx[valid_x, idx] / kernel_result.kx[valid_x, idx] * delta_x[valid_x]

        if theta_rad[idx] > (np.pi / 2.0):
            rl[5, :, idx] = ix * sang[idx]
            rl[6, :, idx] = io * sang[idx]
        else:
            rl[5, :, idx] = io * sang[idx]
            rl[6, :, idx] = ix * sang[idx]

    return BatchRunResult(status=kernel_result.status.copy(), rl=rl, native_freq_hz=kernel_result.native_freq_hz.copy())


def run_powerlaw_iso_batch_cpu(libname, batch):
    mwfunc = initGET_MW(resolve_library_path(libname))
    rl = np.zeros((7, batch.nfreq, batch.batch_size), dtype=np.float64, order="F")
    status = np.zeros(batch.batch_size, dtype=np.int32)

    for idx in range(batch.batch_size):
        lparms, rparms, parms, dummy, out = _build_powerlaw_iso_single_inputs(batch, idx)
        status[idx] = int(mwfunc(lparms, rparms, parms, dummy, dummy, dummy, out))
        rl[:, :, idx] = out

    return BatchRunResult(status=status, rl=rl, native_freq_hz=batch.native_freq_hz)


def interpolate_total_intensity(native_freq_hz, rl, target_freq_hz):
    native_freq_hz = np.asarray(native_freq_hz, dtype=np.float64)
    target_freq_hz = np.asarray(target_freq_hz, dtype=np.float64)
    if rl.ndim != 3:
        raise ValueError("rl must have shape (7, nfreq, batch)")

    total_intensity = rl[5] + rl[6]
    out = np.zeros((target_freq_hz.size, total_intensity.shape[1]), dtype=np.float64)
    log_native_freq = np.log10(native_freq_hz)
    log_target_freq = np.log10(target_freq_hz)
    for idx in range(total_intensity.shape[1]):
        flux = np.maximum(total_intensity[:, idx], 1.0e-300)
        out[:, idx] = np.power(
            10.0,
            np.interp(log_target_freq, log_native_freq, np.log10(flux)),
        )
    return out

def initGET_MW_SLICE(libname):
    _intp=ndpointer(dtype=ctypes.c_int32, flags='F')
    _doublep=ndpointer(dtype=ctypes.c_double, flags='F')
    
    libc_mw=ctypes.CDLL(libname)
    mwfunc=libc_mw.pyGET_MW_SLICE
    mwfunc.argtypes=[_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype=ctypes.c_int

    return mwfunc
