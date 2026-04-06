import platform
import os
import ctypes
from pathlib import Path
from numpy.ctypeslib import ndpointer
import numpy as np
import time
from scipy import interpolate
try:
    from astropy import constants as const
    import astropy.units as u
except ImportError:
    const = None
    u = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import torch
except ImportError:
    torch = None


ASEC2CM = 0.725e8
LEGACY_AREA_ASEC2 = 1.625625
LEGACY_EMAX_MEV = 10.0
DEFAULT_MCMC_FREQGHZ = np.array([
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


def _resolve_default_library_path(libname=None):
    if libname:
        return libname

    repo_root = Path(__file__).resolve().parents[1]
    source_candidate = repo_root / "source" / "MWTransferArr.so"
    if source_candidate.exists():
        return str(source_candidate.resolve())
    return select_binary()


def _expand_legacy_or_full_params(params_array):
    params = np.asarray(params_array, dtype=np.float64).reshape(-1)
    if params.size == 8:
        full = np.empty((10,), dtype=np.float64)
        full[0] = LEGACY_AREA_ASEC2
        full[1:9] = params
        full[9] = LEGACY_EMAX_MEV
        return full
    if params.size == 10:
        return params
    raise ValueError(
        "simulate_spectrum_optimized expects either the legacy 8D reduced parameter order "
        "[depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV] "
        "or the full 10D order "
        "[area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV, Emax_MeV]."
    )


def select_binary():
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        cur_lib_file = '../binaries/MWTransferArr.so'
        if platform.machine() == 'arm64':
            cur_lib_file = '../binaries/MWTransferArr_arm64.so'
        libname = os.path.join(os.path.dirname(os.path.realpath(__file__)), cur_lib_file)
    if platform.system() == 'Windows':
        libname = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../binaries/MWTransferArr64.dll')
    return libname

def initGET_MW(libname, load_GRFF = False):
    """
    Python wrapper for fast gyrosynchrotron codes.
    Identical to GScodes.py in https://github.com/kuznetsov-radio/gyrosynchrotron
    This is for the single thread version
    @param libname: path for locating compiled shared library
    @return: An executable for calling the GS codes in single thread

    The pyGET_MW in GRFF that calculate gyroresonance and free-free emission : https://github.com/kuznetsov-radio/GRFF
    has the same calling manner. So, this function can be used to call the function as well when GRFF lib is provided.
    For the single thread version
    """
    _intp = ndpointer(dtype=ctypes.c_int32, flags='F')
    _doublep = ndpointer(dtype=ctypes.c_double, flags='F')

    libc_mw = ctypes.CDLL(libname)
    if not load_GRFF:
        mwfunc = libc_mw.pyGET_MW
    else:
        mwfunc = libc_mw.PyGET_MW
    mwfunc.argtypes = [_intp, _doublep, _doublep, _doublep, _doublep, _doublep, _doublep]
    mwfunc.restype = ctypes.c_int

    return mwfunc


def simulate_spectrum_optimized(params_array, freqghz=None, spec_in_tb=True, debug=False, libname=None):
    """
    Optimized version of simulate_spectrum that takes an array of parameters instead of parameter object

    Parameters:
    -----------
    params_array : numpy.ndarray or list
        Either the current legacy reduced fitting order:
        [depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV]
        or the full physical order:
        [area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV, Emax_MeV]
    freqghz : numpy.ndarray
        Frequencies in GHz
    spec_in_tb : bool
        If True, return brightness temperature in K, otherwise flux density in sfu
    debug : bool
        If True, print debug information
    libname : str or None
        Shared library path. Defaults to the local repo build when present.

    Returns:
    --------
    mtb : numpy.ndarray
        Simulated brightness temperature spectrum in K (if spec_in_tb=True) or flux in sfu (if spec_in_tb=False)
    elapsed_time : float
        Time taken to simulate the spectrum in seconds
    """
    # Start the timer
    start_time = time.time()

    # Load library
    libname = _resolve_default_library_path(libname)
    GET_MW = initGET_MW(libname)  # load the library

    # Default frequencies if none provided
    if freqghz is None:
        freqghz = DEFAULT_MCMC_FREQGHZ

    area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV, Emax_MeV = (
        _expand_legacy_or_full_params(params_array)
    )


    # Convert parameters to physical units
    src_area = float(area_asec2)  # source area in arcsec^2
    src_area_cm2 = src_area * ASEC2CM ** 2.  # source area in cm^2
    depth_cm = float(depth_asec) * ASEC2CM  # total source depth in cm
    Bmag = float(Bx100G) * 100.  # magnetic field strength in G
    Tth = float(T_MK) * 1e6  # thermal temperature in K
    nth = 10. ** float(log_nth)  # thermal density
    nrl = 10. ** float(log_nnth)  # total nonthermal density above E_min
    Emin = float(Emin_keV) / 1e3  # low energy cutoff of nonthermal electrons in MeV
    Emax = float(Emax_MeV)  # high energy cutoff of nonthermal electrons in MeV

    if debug:
        # debug against previous codes
        print('depth, Bmag, Tth, nth/1e10, lognrl, delta, theta, Emin, Emax: '
              '{0:.1f}, {1:.1f}, {2:.1f}, {3:.1f}, {4:.1f}, '
              '{5:.1f}, {6:.1f}, {7:.2f}, {8:.1f}'.format(depth_cm / 0.725e8, Bmag, Tth / 1e6, nth / 1e10,
                                                          np.log10(nrl), delta, theta, Emin, Emax))

    Nf = 100  # number of frequencies
    NSteps = 1  # number of nodes along the line-of-sight

    Lparms = np.zeros(11, dtype='int32')  # array of dimensions etc.
    Lparms[0] = NSteps
    Lparms[1] = Nf

    Rparms = np.zeros(5, dtype='double')  # array of global floating-point parameters
    Rparms[0] = src_area_cm2  # Area, cm^2
    Rparms[1] = 0.8e9  # starting frequency to calculate spectrum, Hz
    Rparms[2] = 0.02  # logarithmic step in frequency
    Rparms[3] = 0  # f^C
    Rparms[4] = 0  # f^WH

    # Optimize by pre-allocating arrays outside of the loop
    ParmLocal = np.zeros(24, dtype='double')  # array of voxel parameters - for a single voxel
    ParmLocal[0] = depth_cm / NSteps  # voxel depth, cm
    ParmLocal[1] = Tth  # T_0, K
    ParmLocal[2] = nth  # n_0 - thermal electron density, cm^{-3}
    ParmLocal[3] = Bmag  # B - magnetic field, G
    ParmLocal[6] = 3  # distribution over energy (PLW is chosen, 3)
    ParmLocal[7] = nrl  # n_b - nonthermal electron density, cm^{-3}
    ParmLocal[9] = Emin  # E_min, MeV
    ParmLocal[10] = Emax  # E_max, MeV
    ParmLocal[12] = delta  # \delta_1
    ParmLocal[14] = 0  # distribution over pitch-angle (isotropic is chosen)
    ParmLocal[15] = 90  # loss-cone boundary, degrees
    ParmLocal[16] = 0.2  # \Delta\mu

    Parms = np.zeros((24, NSteps), dtype='double', order='F')  # 2D array of input parameters - for multiple voxels
    for i in range(NSteps):
        Parms[:, i] = ParmLocal  # most of the parameters are the same in all voxels
        Parms[4, i] = theta

    RL = np.zeros((7, Nf), dtype='double', order='F')  # input/output array
    dummy = np.array(0, dtype='double')

    # calculating the emission for array distribution (array -> on)
    res = GET_MW(Lparms, Rparms, Parms, dummy, dummy, dummy, RL)

    # retrieving the results
    f = RL[0]
    I_L = RL[5]
    I_R = RL[6]
    all_zeros = not RL.any()

    if not all_zeros:
        flux_model = I_L + I_R
        flux_model = np.nan_to_num(flux_model) + 1e-11

        # Optimize interpolation by working in log space directly
        logf = np.log10(f)
        logflux_model = np.log10(flux_model)
        logfreqghz = np.log10(freqghz)

        # Use linear interpolation which is faster than higher order methods
        interpfunc = interpolate.interp1d(logf, logflux_model, kind='linear')
        logmflux = interpfunc(logfreqghz)
        mflux = 10. ** logmflux

        if spec_in_tb:
            tb = sfu2tb(np.array(freqghz) * 1.e9, mflux, area=src_area)
            mtb = tb.value if hasattr(tb, "value") else np.asarray(tb, dtype=np.float64)
        else:
            mtb = mflux
    else:
        # Handle error case
        if spec_in_tb:
            mtb = np.ones_like(freqghz) * 1e4
        else:
            mtb = np.ones_like(freqghz) * 1e-11
    # if add_noise:
    #     mtb = mtb * (1 + 0.05 * np.random.randn(*mtb.shape))

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    #return mtb, elapsed_time
    return mtb



def sfu2tb(frequency, flux, area=None, size=None, square=True, reverse=False, verbose=False):
    """
        frequency: single element or array, in Hz
        flux: single element or array of flux, in sfu; if reverse, it is brightness temperature in K
        area: area in arcsec^2
        size: Two-dimensional width of the radio source, [major, minor], in arcsec.
              Ignored if both area and size are provided
        reverse: if True, convert brightness temperature in K to flux in sfu integrated uniformly withing the size
    """

    if const is None or u is None:
        if area is None and size is None:
            raise ValueError("area or size is required when astropy is unavailable")

        frequency = np.asarray(frequency, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)

        if area is None:
            if not isinstance(size, list):
                size = [size]
            if len(size) == 1:
                a = b = float(size[0]) / 2.0
            elif len(size) == 2:
                a = float(size[0]) / 2.0
                b = float(size[1]) / 2.0
            else:
                raise ValueError("size needs to have 1 or 2 elements")
            area = 4.0 * a * b if square else np.pi * a * b

        sr = float(area) * (np.pi / (180.0 * 3600.0)) ** 2
        factor = (2.99792458e10 ** 2) / (2.0 * 1.380649e-16 * np.square(frequency) * sr)
        if reverse:
            return flux / (factor * 1.0e-19)
        return flux * 1.0e-19 * factor

    c = const.c.cgs
    k_B = const.k_B.cgs
    sfu = u.jansky * 1e4

    if (not 'area' in vars()) and (not 'size' in vars()):
        print('Neither area nor size is provided. Abort...')

    if not hasattr(frequency, 'unit'):
        # assume frequency is in Hz
        frequency = frequency * u.Hz

    if not hasattr(flux, 'unit'):
        # assume flux is in sfu
        if reverse:
            flux = flux * u.K
        else:
            flux = flux * sfu

    if area is not None:
        if not hasattr(area, 'unit'):
            # assume area is in arcsec^2
            area = area * u.arcsec ** 2

    if size is not None and (area is None):
        if type(size) != list:
            size = [size]

        if len(size) > 2:
            print('size needs to have 1 or 2 elements.')
        elif len(size) < 2:
            if verbose:
                print('Only one element is provided for source size. Assume symmetric source')
            if not hasattr(size[0], 'unit'):
                # assume size in arcsec
                size[0] = size[0] * u.arcsec
            # define half size
            a = b = size[0] / 2.
        else:
            if not hasattr(size[0], 'unit'):
                # assume size in arcsec
                size[0] = size[0] * u.arcsec
            if not hasattr(size[1], 'unit'):
                # assume size in arcsec
                size[1] = size[1] * u.arcsec
            # define half size
            a = size[0] / 2.
            b = size[1] / 2.
        if square:
            if verbose:
                print('Assume square-shaped source.')
            area = 4. * a * b
        else:
            if verbose:
                print('Assume elliptical-shaped source.')
            area = np.pi * a * b

    sr = area.to(u.radian ** 2)
    factor = c ** 2. / (2. * k_B * frequency ** 2. * sr)

    if reverse:
        # returned value is flux in sfu
        if verbose:
            print('converting input brightness temperature in K to flux density in sfu.')
        return (flux / factor).to(sfu, equivalencies=u.dimensionless_angles())
    else:
        # returned value is brightness temperature in K
        if verbose:
            print('converting input flux density in sfu to brightness temperature in K.')
        return (flux * factor).to(u.K, equivalencies=u.dimensionless_angles())


# Function to compare sparse vs. dense frequency approaches
def compare_sparse_vs_dense():
    """
    Compare performance between using sparse frequencies directly vs. dense frequencies with interpolation

    This is a simulation since we don't have the ability to modify the library here.
    """
    print("Simulating performance comparison between sparse and dense frequency approaches")

    # Create a test parameter set
    test_params = [20.0, 10.0, 2.0, 10.0, 10.0, 7.0, 5.0, 45.0, 20.0, 100.0]

    # Define sparse frequencies (what you might observe)
    sparse_freqghz = np.logspace(np.log10(1), np.log10(18), 32)

    # Define dense frequencies (what the library uses internally)
    dense_freqghz = np.logspace(np.log10(0.8), np.log10(20), 100)

    # Simulate timing for current approach (dense calculation + interpolation)
    print("\nCurrent approach (dense calculation + interpolation):")
    start = time.time()
    for _ in range(100):
        # Simulate the dense calculation time
        calc_time = 0.01  # assume 10ms for calculation with fixed frequencies

        # Simulate interpolation to sparse frequencies
        # This would normally use the actual interpolation code
        interp_time = 0.001  # assume 1ms for interpolation

        # Total time
        total_time = calc_time + interp_time
        time.sleep(total_time / 100)  # Scale down for demonstration

    dense_time = time.time() - start
    print(f"Estimated time: {dense_time:.6f} seconds for 100 iterations")

    # Simulate timing for hypothetical sparse approach (direct calculation at sparse points)
    print("\nHypothetical sparse approach (direct calculation at sparse points):")
    start = time.time()
    for _ in range(100):
        # Simulate the calculation time for sparse frequencies
        # This would be faster because fewer points, but overhead might be similar
        calc_time = 0.01 * (len(sparse_freqghz) / len(dense_freqghz))

        # No interpolation needed
        time.sleep(calc_time / 100)  # Scale down for demonstration

    sparse_time = time.time() - start
    print(f"Estimated time: {sparse_time:.6f} seconds for 100 iterations")

    # Calculate potential speedup
    potential_speedup = dense_time / sparse_time
    print(f"\nPotential speedup factor: {potential_speedup:.2f}x")

    # Analysis and recommendation
    print("\nAnalysis of potential library modification:")
    print("- The main computational cost is likely in the physics calculations, not the frequency sampling")
    print("- Overhead of initialization and setup might dominate for sparse frequencies")
    print("- The interpolation cost is minimal compared to the physics calculation")

    if potential_speedup > 1.5:
        print("\nRecommendation: Consider modifying the library to accept sparse frequencies,")
        print("especially if you'll be running millions of simulations for SBI training.")
    else:
        print("\nRecommendation: The potential speedup may not justify the effort of modifying")
        print("the library. Focus on other optimizations first, like parallel processing.")

    return potential_speedup


# Function to perform benchmarking of the simulation function
def benchmark_spectrum_simulation(n_samples=100):
    """
    Benchmark the spectrum simulation function by running it multiple times

    Parameters:
    -----------
    n_samples : int
        Number of simulations to run

    Returns:
    --------
    avg_time : float
        Average time per simulation in seconds
    """
    # Create random parameter sets for benchmarking
    np.random.seed(42)  # For reproducibility

    # Parameter ranges
    param_ranges = [
        #(2.0, 100.0),  # area_asec2
        (1.0, 30.0),  # depth_asec
        (0.5, 10.0),  # Bx100G
        (1.0, 50.0),  # T_MK
        (8.0, 12.0),  # log_nth
        (5.0, 9.0),  # log_nnth
        (3.0, 9.0),  # delta
        (10.0, 80.0),  # theta
        (10.0, 100.0),  # Emin_keV
        #(10.0, 1000.0)  # Emax_MeV
    ]

    # Generate random parameters within these ranges
    param_sets = []
    for _ in range(n_samples):
        params = [np.random.uniform(low, high) for low, high in param_ranges]
        param_sets.append(params)

    # Define frequencies
    freqghz = np.logspace(np.log10(1), np.log10(18), 32)

    # Run benchmarks
    total_time = 0
    times = []

    print(f"Running {n_samples} simulations for benchmarking...")
    for i, params in enumerate(param_sets):
        _, elapsed_time = simulate_spectrum_optimized(params, freqghz)
        total_time += elapsed_time
        times.append(elapsed_time)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_samples} simulations")

    avg_time = total_time / n_samples
    min_time = min(times)
    max_time = max(times)

    print(f"\nBenchmark Results:")
    print(f"Average time per simulation: {avg_time:.6f} seconds")
    print(f"Minimum time: {min_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Total time for {n_samples} simulations: {total_time:.6f} seconds")

    return avg_time, times


def piecewise_distribution_sampling(
        param_bounds=(0.0, 10.0),
        ranges=[(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)],
        densities=[3, 1, 2],
        n_samples=10000,
        sample_mode='relative'
):
    """
    Generate samples with different densities in different ranges.

    Parameters:
    -----------
    param_bounds : tuple
        Overall (min, max) for the parameter
    ranges : list of tuples
        List of (min, max) ranges within param_bounds
    densities : list of float
        Relative densities for each range (higher means more samples)
        or absolute counts if sample_mode='absolute'
    n_samples : int
        Total number of samples to generate (used only in 'relative' mode)
    sample_mode : str
        'relative': densities are relative and total samples = n_samples
        'absolute': densities are absolute sample counts per range

    Returns:
    --------
    samples : numpy.ndarray
        Samples for the parameter
    """
    if sample_mode == 'relative':
        # Normalize densities to sum to 1
        densities = np.array(densities)
        densities = densities / densities.sum()

        # Calculate number of samples for each range
        samples_per_range = (densities * n_samples).astype(int)

        # Adjust if sum doesn't match n_samples due to integer rounding
        remainder = n_samples - samples_per_range.sum()
        if remainder > 0:
            # Add remainder to the range with highest density
            max_density_idx = np.argmax(densities)
            samples_per_range[max_density_idx] += remainder

    elif sample_mode == 'absolute':
        # Use densities directly as sample counts
        samples_per_range = np.array(densities, dtype=int)
        # Update n_samples for reporting
        n_samples = np.sum(samples_per_range)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}. Use 'relative' or 'absolute'.")

    # Generate samples for each range
    all_samples = []
    for i, ((low, high), n) in enumerate(zip(ranges, samples_per_range)):
        if n > 0:  # Only generate if we need samples for this range
            range_samples = np.random.uniform(low, high, n)
            all_samples.append(range_samples)

    # Combine and shuffle
    combined_samples = np.concatenate(all_samples)
    np.random.shuffle(combined_samples)

    print(f"Generated {len(combined_samples)} samples across {len(ranges)} ranges.")
    return combined_samples


# Example usage and visualization
def verify_precise(inp_sample = None):
    # Example: Parameter with range [0, 10] with three sub-ranges
    # Range [0, 2] has 3x the density of range [2, 5]
    # Range [5, 10] has 2x the density of range [2, 5]
    if inp_sample is None:
        inp_sample = piecewise_distribution_sampling(
        param_bounds=(0.0, 10.0),
        ranges=[(0.0, 2.0), (2.0, 5.0), (5.0, 10.0)],
        densities=[3, 1, 2],
        n_samples=10000
        )

    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(inp_sample, bins=50, alpha=0.7)
    plt.title('Piecewise Non-Uniform Distribution')
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')

    # Add vertical lines at range boundaries
    for boundary in [2.0, 5.0]:
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

    # Annotate the ranges with their relative densities
    plt.text(1.0, plt.ylim()[1] * 0.9, "Density: 3x",
             horizontalalignment='center', verticalalignment='center')
    plt.text(3.5, plt.ylim()[1] * 0.9, "Density: 1x",
             horizontalalignment='center', verticalalignment='center')
    plt.text(7.5, plt.ylim()[1] * 0.9, "Density: 2x",
             horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.show()

    # Modify the sampling_spec for this parameter
    # Example of how to add this to the original code:
    # sampling_spec_modified = {
    #     0: {'method': 'piecewise',
    #         'ranges': [(1.0, 10.0), (10.0, 30.0), (30.0, 50.0)],
    #         'densities': [5, 1, 3]},  # More samples in [1,10] and [30,50]
    #     1: {'method': 'grid', 'n_points': 20},
    #     # ... rest of parameters unchanged
    # }




def tmp_testing(test_params):
    #benchmark_spectrum_simulation()
    #verify_precise()
    #test_params = np.array([10.0, 3.6, 10.0, 10.0, 6.0, 4.0, 75, 20.0])
    obs=simulate_spectrum_optimized(test_params,spec_in_tb=True)
    freqghz = np.array([1.2558008e+09, 1.5798831e+09, 2.5500792e+09, 2.8738688e+09,
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
                        1.7492732e+10, 1.7817741e+10], dtype=np.float32) / 1e9  # Convert to GHz
    plt.loglog(freqghz, obs)
    #plt.ylim([1.e4,4.e8])
    plt.show()
    print(obs)


def main():
    #benchmark_spectrum_simulation()
    #verify_precise()
    test_params = np.array([10.0, 3.6, 10.0, 10.0, 6.0, 4.0, 75, 20.0])
    obs=simulate_spectrum_optimized(test_params,spec_in_tb=True)
    freqghz = np.array([1.2558008e+09, 1.5798831e+09, 2.5500792e+09, 2.8738688e+09,
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
                        1.7492732e+10, 1.7817741e+10], dtype=np.float32) / 1e9  # Convert to GHz
    plt.loglog(freqghz, obs)
    #plt.ylim([1.e4,4.e8])
    plt.show()
    print(obs)

if __name__ == '__main__':
    main()
