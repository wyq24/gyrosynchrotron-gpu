# GPU-batched exact MCMC framework

This package-level backend is the current additive MCMC path for the validated narrow workflow:

- analytical `PLW + ISO`
- `Nz = 1`
- approximate GS only
- exact `emcee` sampling
- GPU/CPU through the validated wrapper batch interface
- `FP64` as the production correctness path

For the full usage guide, differences from the legacy path, and calling examples, see:

- `doc/public/GpuBatchedMCMC.md`

## Quick start

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

## Most important differences from the old method

- Do not call `simulate_spectrum_optimized` inside the new likelihood.
- Do not use multiprocessing with the GPU backend.
- Set `batch_capacity >= n_walkers`.
- The current 8D fitting meaning is preserved by fixing `area_asec2=1.625625` and `Emax_MeV=10.0`.
- The likelihood observable is brightness temperature on the legacy MCMC frequency grid, reconstructed from the validated wrapper result.
- Use `FP64` for correctness-sensitive runs.

## Validation commands

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex XDG_CACHE_HOME=/tmp/.cache-codex /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_validate.py --mode spectra --backend cuda
MPLCONFIGDIR=/tmp/matplotlib-codex XDG_CACHE_HOME=/tmp/.cache-codex /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_validate.py --mode cube --backend cuda
MPLCONFIGDIR=/tmp/matplotlib-codex XDG_CACHE_HOME=/tmp/.cache-codex /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_benchmark.py --backend cuda --batch-size 32 --repeats 10
```
