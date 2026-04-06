# GPU-Batched MCMC Backend

## Scope

This backend is only for the current validated narrow path:

- Python only
- analytical `PLW + ISO`
- `Nz = 1`
- approximate GS only
- batched workflow first
- `FP64` is the production correctness path

It is additive. It does not replace the old legacy MCMC files.

## Main file

- `mcmc_example/mcmc_backend_gpu_batched.py`

## What this backend does

The new backend keeps exact `emcee` sampling, but changes how the forward model is called:

- one persistent forward backend object
- one batched likelihood call per walker batch via `emcee(vectorize=True)`
- no multiprocessing for GPU likelihood evaluation
- wrapper batch object reused in place across calls
- shared library loaded once instead of inside every spectrum simulation

This is why it is much faster than the old path.

## Old path vs new path

### Old legacy path

Main legacy files:

- `mcmc_example/mcmc_backend.py`
- `mcmc_example/npec_helpers.py`
- `mcmc_example/spec_utils.py`

Behavior:

- likelihood uses `simulator_8d` / `simulate_spectrum_optimized`
- forward model is evaluated one spectrum at a time
- shared library loading and wrapper overhead happen repeatedly
- GPU is not used in the efficient validated batch-wrapper manner

### New batched path

Main new file:

- `mcmc_example/mcmc_backend_gpu_batched.py`

Behavior:

- uses the validated wrapper batch path around `MWTransferArr.so`
- uses `emcee(vectorize=True)`
- evaluates all walkers in one batch per likelihood call
- reuses backend state and batch arrays
- GPU path is designed to be used without multiprocessing

## Important scientific difference from the old path

The old legacy simulator and the new validated approximate batch path are not numerically identical models.

The new backend is tied to the validated narrow wrapper path. That is intentional.

From the current repo wiring:

- the new backend matches the validated approximate single-call reference on the supported path
- the old `simulate_spectrum_optimized` path is different enough that its spectra should not be treated as the validation reference for the new backend

So when comparing results:

- compare new backend against the validated wrapper/native reference
- do not assume old legacy spectra are the correctness target

## Parameter order

### Legacy reduced 8D fitting order

This is the current fitting order preserved by the new backend:

```python
[depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV]
```

### Full 10D physical order

The backend also accepts the explicit full physical order:

```python
[area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV, Emax_MeV]
```

### Preserved legacy semantics

When the current 8D fitting order is used, the backend preserves the old public fitting meaning by fixing:

- `area_asec2 = 1.625625`
- `Emax_MeV = 10.0`

This means the backend does **not** silently broaden the current fitting definition.

## Observable returned by the backend

The new batched backend returns the same observable used by the current likelihood:

- total intensity `I_L + I_R`
- interpolated onto the legacy 50-channel MCMC frequency grid
- converted to brightness temperature `Tb`

This is important.

The likelihood is not using raw native RL arrays directly. It is using the `Tb` spectrum reconstructed from the validated wrapper result.

## Production recommendations

Use these settings for production-correct runs:

- `backend="cuda"`
- `precision="fp64"`
- `npoints=16`
- `q_on=True`
- `d_sun_au=1.0`

Do not use `FP32` as the correctness path.

Current repo evidence still shows that `FP32` can have non-negligible worst-case errors relative to CPU/FP64 even though it is much faster.

## Core entry points

### 1. Build a persistent backend

```python
from mcmc_example.mcmc_backend_gpu_batched import build_legacy_8d_batched_backend

backend = build_legacy_8d_batched_backend(
    lib_path="source/MWTransferArr.so",
    batch_capacity=512,
    backend="cuda",
)
```

Notes:

- `batch_capacity` must be at least the maximum walker batch you will evaluate in one call
- for `emcee(vectorize=True)`, this usually means `batch_capacity >= n_walkers`
- if you exceed capacity, the backend raises an error instead of silently reallocating

### 2. Simulate one spectrum through the new path

```python
from mcmc_example.mcmc_backend_gpu_batched import simulate_legacy_8d_spectrum_gpu_batched

params = [5.0, 2.0, 1.0, 10.0, 5.0, 4.0, 45.0, 10.0]
spectrum_tb = simulate_legacy_8d_spectrum_gpu_batched(
    params,
    lib_path="source/MWTransferArr.so",
    backend="cuda",
)
```

This is useful for smoke tests and for checking the exact observable returned by the new path.

### 3. Run one exact MCMC fit on one spectrum

```python
from mcmc_example.mcmc_backend_gpu_batched import (
    SamplingConfig,
    build_legacy_8d_batched_backend,
    run_single_mcmc_gpu_batched,
)

backend = build_legacy_8d_batched_backend(
    lib_path="source/MWTransferArr.so",
    batch_capacity=512,
    backend="cuda",
)

sampling_cfg = SamplingConfig(
    n_walkers=512,
    n_steps=200,
    burn_in=50,
    thin=5,
    noise_level=0.10,
)

samples = run_single_mcmc_gpu_batched(
    spectrum=observed_tb,
    vary_bounds=all_param_bounds,
    vary_indices=list(range(8)),
    fixed_params=[0.0] * 8,
    x_log_bounds=(4.0, 9.0),
    sampling_cfg=sampling_cfg,
    forward_backend=backend,
    seed=12345,
)
```

Returned shape:

- `samples.shape == (n_saved_samples, n_dim)`

### 4. Run resumable cube fitting

```python
from mcmc_example.mcmc_backend_gpu_batched import (
    CubeSamplingConfig,
    SamplingConfig,
    WarmStartConfig,
    build_legacy_8d_batched_backend,
    fit_cube_mcmc_resumable_gpu,
)

backend = build_legacy_8d_batched_backend(
    lib_path="source/MWTransferArr.so",
    batch_capacity=512,
    backend="cuda",
)

cube_cfg = CubeSamplingConfig(
    sampling=SamplingConfig(
        n_walkers=512,
        n_steps=200,
        burn_in=50,
        thin=5,
        noise_level=0.10,
    ),
    warm_start=WarmStartConfig(
        use_neighbor_samples=True,
        jitter_std_norm=0.05,
        exploration_fraction=0.25,
        max_neighbor_samples=4096,
        broad_init_scale=0.35,
    ),
    checkpoint_every=10,
    max_nodes=None,
    save_samples=True,
)

result = fit_cube_mcmc_resumable_gpu(
    cube=cube,
    all_param_bounds=all_param_bounds,
    vary_indices=list(range(8)),
    fixed_params=[0.0] * 8,
    x_log_bounds=(4.0, 9.0),
    segmentation="pixel",
    block_k=1,
    valid_mask=None,
    out_dir="out",
    resume_path="out/resume.npz",
    cube_cfg=cube_cfg,
    forward_backend=backend,
    seed=12345,
)
```

Key outputs:

- `result.theta_map`
- `result.q16_map`
- `result.q84_map`
- `result.node_thetas`
- `result.done_nodes`
- `result.debug`

## What to notice compared with the previous method

### 1. Do not call `simulate_spectrum_optimized` inside the new likelihood

Old method:

- repeatedly calls `simulate_spectrum_optimized`
- old single-call wrapper style
- inefficient for GPU MCMC

New method:

- pass a persistent `forward_backend` into the sampler
- let the sampler call `forward_backend.simulate_batch(...)`

### 2. Do not use multiprocessing with the GPU backend

Old CPU-style logic may use multiprocessing pools.

For the GPU backend, do not do that.

The new path is designed for:

- one process
- one persistent backend
- one batched walker evaluation per log-probability call

### 3. Choose `batch_capacity` from walker count

If you use:

- `n_walkers = 512`

then use at least:

- `batch_capacity = 512`

Otherwise the backend will reject the call.

### 4. The new path is exact MCMC, not an approximate posterior shortcut

Warm starts only change initialization.

They do **not** change the target posterior.

### 5. CPU and CUDA are both supported through the same new API

You can switch:

- `backend="cpu"`
- `backend="cuda"`

without changing the sampler call pattern.

That makes CPU-vs-CUDA benchmarking straightforward.

### 6. FP32 is faster but still not the production path

Current repo evidence shows:

- `FP32` wrapper can be much faster than `FP64`
- but worst-case spectral errors relative to CPU/FP64 are still too large for correctness-sensitive fitting

Use `FP64` for production fits.

## Validation commands

One-spectrum validation:

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex XDG_CACHE_HOME=/tmp/.cache-codex /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_validate.py --mode spectra --backend cuda
```

Tiny-cube smoke test:

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex XDG_CACHE_HOME=/tmp/.cache-codex /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_validate.py --mode cube --backend cuda
```

Forward-model timing benchmark:

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex XDG_CACHE_HOME=/tmp/.cache-codex /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_benchmark.py --backend cuda --batch-size 32 --repeats 10
```

## Practical benchmark interpretation

Current repo evidence now supports three separate claims:

1. The new batched CUDA forward backend is much faster than the old single-call forward path.
2. The new batched MCMC architecture itself is faster than the legacy MCMC architecture even on CPU.
3. The full end-to-end new CUDA MCMC path is dramatically faster than the old legacy MCMC path.

So when timing results are discussed, separate:

- backend speedup
- architecture speedup
- full end-to-end MCMC speedup

## Bottom line

If you are running current production MCMC in this repo, the recommended path is:

- `mcmc_backend_gpu_batched.py`
- persistent backend from `build_legacy_8d_batched_backend(...)`
- `backend="cuda"`
- `precision="fp64"`
- no multiprocessing
- walker count matched by backend batch capacity

That is the path aligned with the current validated GPU wrapper workflow.
