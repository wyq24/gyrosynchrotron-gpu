# JAX Progress Update

Date: 2026-04-14

Branch: `research/jax-vectorized-forward`

Scope remains locked to the narrow supported path:

- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched workflow
- `FP64` correctness anchor

## What Landed

### 1. JAX FFI forward path

The branch now has a working XLA FFI wrapper around the validated fused native RL path.

Key files:

- `jax_ffi/mw_approx_batch_contract.py`
- `jax_ffi/mw_approx_batch_jax.py`
- `source/XLAInterface.cpp`
- `source/XLAInterfaceCuda.cu`
- `source/XLAInterfaceCommon.h`
- `source/makefile`

Current behavior:

- explicit legacy 8D and canonical 10D parameter handling
- current fitting observable reproduced from native RL output
- CPU and CUDA FFI targets registered through `jax.ffi`
- `FP64` only
- additive to the current wrapper path; no legacy behavior replaced

### 2. Validation and runtime tooling

Added tools:

- `tools/jax_env_check.py`
- `tools/jax_phase0_contract_validate.py`
- `tools/jax_ffi_smoke_test.py`
- `tools/jax_phase1_validate.py`
- `tools/jax_phase1_benchmark.py`

Validated on the `ml` conda environment:

- `jax==0.6.2`
- `jaxlib==0.6.2`
- `jax.ffi` available
- CUDA device visible to JAX outside the sandbox

### 3. JAX sampler integration

The forward path is now wired into additive JAX-native sampler entry points.

Key files:

- `jax_ffi/numpyro_ess.py`
- `jax_ffi/__init__.py`
- `mcmc_example/mcmc_backend_gpu_batched.py`
- `mcmc_example/numpyro_ess_migration_skeleton.py`
- `tools/jax_ess_trial.py`
- `tools/jax_ess_validate.py`
- `tools/mcmc_gpu_batched_validate.py`

Implemented sampler work:

- bounded ESS path with explicit unconstrained transform and Jacobian
- reusable ESS runner so cube fits do not rebuild the sampler every node
- multi-seed ESS validation harness
- additive JAX AIES path as a faster JAX-native alternative to ESS
- additive cube-fit entry points for both `jax-ess` and `jax-aies`

## Validation Snapshot

### Forward path

Confirmed on `ml` with GPU:

- `tools/jax_phase0_contract_validate.py` passes against the current native library
- `tools/jax_ffi_smoke_test.py` passes on CUDA
- `tools/jax_phase1_validate.py` passes on CUDA for:
  - `real-sweep`
  - `stress-sweep`
  - batch sizes `1 8 32 128 512`

Warm forward-only benchmark behavior is reasonable. The staged JAX FFI observable path is close to the native CUDA wrapper at larger batch sizes and reached about `1.08x` of the native CUDA wrapper at batch `512` in the last benchmark run.

### Sampler path

Validated:

- `tools/jax_ess_validate.py` passed the current median-drift gate across multiple seeds
- `tools/mcmc_gpu_batched_validate.py --mode cube --sampler jax-ess` passed
- `tools/mcmc_gpu_batched_validate.py --mode cube --sampler jax-aies` passed

Measured behavior:

- reusable ESS fixed the worst recompilation waste, but ESS remained slow
- JAX AIES improved warm runtime by about `5.9x` over the reusable ESS path on the toy benchmark
- JAX AIES is still slower than the existing CPU `emcee` + GPU forward baseline on the toy benchmark

Toy timing snapshot from the latest run:

- CPU `emcee` + GPU forward: about `0.36 s`
- JAX ESS + GPU forward, warm: about `27.8 s`
- JAX AIES + GPU forward, warm: about `4.75 s`

## Current Interpretation

The JAX forward model work is successful.

- The narrow GPU forward path is callable from JAX on both CPU and CUDA.
- The current observable contract is validated.
- JAX-native MCMC is now possible on the branch through additive entry points.

The JAX sampler work is usable but not yet a throughput replacement for the existing batched `emcee` path.

- `jax-ess` is experimental and mainly useful as an integration checkpoint.
- `jax-aies` is the better current JAX-native sampler.
- The fastest production-style throughput path is still the existing CPU sampler plus GPU forward model.

## Still Out Of Scope

- `FP32` correctness work
- gradients, custom VJP, `HMC`, `NUTS`
- exact GS
- array DF
- transport
- beam-aware image-domain fitting

## Resume Commands

Typical resume commands on `ml`:

```bash
env JAX_ENABLE_X64=1 /home/walter/anaconda3/envs/ml/bin/python tools/jax_env_check.py --require-jax --require-ffi --show-devices

env JAX_ENABLE_X64=1 /home/walter/anaconda3/envs/ml/bin/python tools/jax_ffi_smoke_test.py \
  --lib source/MWTransferArr.so \
  --xla-lib source/MWTransferArrXLA.so \
  --reference-backend cuda \
  --jax-device-platform gpu

env JAX_ENABLE_X64=1 /home/walter/anaconda3/envs/ml/bin/python tools/jax_phase1_validate.py \
  --lib source/MWTransferArr.so \
  --xla-lib source/MWTransferArrXLA.so \
  --reference-backend cuda \
  --jax-device-platform gpu

env JAX_ENABLE_X64=1 /home/walter/anaconda3/envs/ml/bin/python tools/mcmc_gpu_batched_validate.py \
  --mode cube \
  --backend cuda \
  --sampler jax-aies \
  --lib source/MWTransferArr.so \
  --xla-lib source/MWTransferArrXLA.so
```
