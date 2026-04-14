# JAX FFI Handoff

Date: 2026-04-13

Branch: `research/jax-vectorized-forward`

Purpose: hand off the current JAX-enablement work for the already-supported GPU narrow path to a CUDA-capable machine.

Update: follow-on implementation status and validation results through
2026-04-14 are summarized in `doc/JAXProgress_2026-04-14.md`.

## Scope Locked

Keep the project constraints unchanged:

- Python only
- Preserve legacy single-call CPU path unchanged
- Narrow path only:
  - analytical `PLW + ISO`
  - `Nz=1`
  - approximate GS only
  - batched workflow
- `FP64` is the correctness anchor
- CPU reference / current validated supported-path behavior is the oracle
- Unsupported cases must fail loudly

This handoff is only for making the already-supported narrow native batch path callable from JAX. It is not a physics rewrite.

## What Was Decided

The roadmap was updated to:

- treat the oracle as the current validated `FP64` supported path, not "ctypes" specifically
- make the observable adapter mandatory
- make legacy 8D compatibility explicit instead of assuming only canonical 10D inputs
- define numerical acceptance as "meets current supported-path `FP64` validation thresholds"
- use modern `jax.ffi`, not the old `jax.extend.ffi`
- judge performance mainly on warm steady-state throughput for meaningful batch sizes
- stay with ESS / gradient-free integration first; defer pure-JAX physics and NUTS/HMC unless gradients are truly needed

Related roadmap doc already updated:

- [doc/internal/development/JAXVectorizedForwardFeasibility.md](/Users/walterwei/Library/CloudStorage/Dropbox/gyrosynchrotron-gpu/doc/internal/development/JAXVectorizedForwardFeasibility.md)

## Environment Used So Far

Preferred environment:

- `/Users/walterwei/miniforge3/envs/gsfit_cp`

Reason:

- Python `3.10`
- simpler fit for the current repo and toolchain

Installed in that env:

- `jax==0.6.2`
- `jaxlib==0.6.2`

Verified with:

```bash
/Users/walterwei/miniforge3/envs/gsfit_cp/bin/python tools/jax_env_check.py --require-jax --require-ffi --show-devices
```

Observed on the current machine:

- `jax.ffi` available
- XLA FFI headers available through `jaxlib`
- CPU only
- `jax_enable_x64` defaults to `False`, so runtime tests used `JAX_ENABLE_X64=1`

## Files Added Or Changed

Tracked modifications already on this branch:

- `doc/internal/development/JAXVectorizedForwardFeasibility.md`
- `jax_ffi/mw_approx_batch_jax_stub.py`
- `source/makefile`

New files added for the JAX phase:

- `jax_ffi/__init__.py`
- `jax_ffi/mw_approx_batch_contract.py`
- `jax_ffi/mw_approx_batch_jax.py`
- `source/XLAInterface.cpp`
- `tools/jax_env_check.py`
- `tools/jax_phase0_contract_validate.py`
- `tools/jax_ffi_smoke_test.py`

Unrelated local artifacts existed in the worktree and were intentionally left untouched:

- `.DS_Store`
- `.codex`
- `.idea/`
- `binaries/MWTransferArr_cpu_only_linux_x86_64.buildinfo`
- `binaries/MWTransferArr_cpu_only_linux_x86_64.so`
- `binaries/MWTransferArr_cuda_linux_x86_64.buildinfo`
- `binaries/MWTransferArr_cuda_linux_x86_64.so`
- `numpyro_ess_migration_skeleton.py`

## What Was Implemented

### 1. Contract Layer

`jax_ffi/mw_approx_batch_contract.py` freezes the supported JAX-facing parameter and observable contract without requiring JAX.

It provides:

- canonical 10D parameter handling
- legacy 8D expansion with fixed:
  - `area_asec2 = 1.625625`
  - `Emax_MeV = 10.0`
- conversion from current wrapper-packed batch input to canonical parameters
- validation for the supported narrow path
- extraction of the current fitting observable from native RL output
- legacy frequency-grid interpolation
- flux-to-brightness-temperature conversion

This makes the observable adapter a first-class piece of the integration instead of treating raw RL as the final fitting quantity.

### 2. Environment Check Tool

`tools/jax_env_check.py` reports:

- Python version
- NumPy version
- JAX version
- jaxlib version
- `jax.ffi` availability
- pycapsule support
- XLA header availability
- visible devices

This is the first thing to run on the CUDA machine.

### 3. Phase-0 Contract Validation Tool

`tools/jax_phase0_contract_validate.py` validates:

- parameter contract
- legacy 8D vs canonical 10D semantics
- observable extraction against the current validated supported path when a compatible shared library is available

This is meant to establish that the JAX-facing contract matches current supported-path behavior before relying on the XLA boundary.

### 4. Native XLA FFI Proof

`source/XLAInterface.cpp` adds a first XLA FFI CPU proof target around the fused RL native path.

Current behavior:

- exported symbol: `MWApproxBatchRlCpu`
- input: parameter tensor with trailing dimension `10`
- attrs:
  - `npoints`
  - `q_on`
  - `nu0_hz`
  - `dlog10_nu`
  - `d_sun_au`
- output:
  - `rl`
  - `freq`
- backend forced to CPU
- precision forced to `FP64`
- all leading dims are flattened into batch
- returns an FFI error on:
  - unsupported shape contract
  - native call failure
  - any non-zero per-item status

The current handler is intentionally narrow and strict.

### 5. JAX Wrapper

`jax_ffi/mw_approx_batch_jax.py` provides:

- registration of the FFI target via `jax.ffi.register_ffi_target(...)`
- canonicalization of either legacy 8D or full 10D parameters
- low-level JAX call to the RL native path
- a convenience wrapper that produces the current legacy fitting observable

Important output convention:

- the native RL buffer is treated as C-order `[batch..., nfreq, 7]`
- the JAX wrapper returns it as `[batch..., 7, nfreq]` after axis swap

### 6. Smoke Test

`tools/jax_ffi_smoke_test.py` compares:

- JAX FFI RL output vs current supported-path wrapper output
- JAX FFI frequency grid vs current supported-path wrapper output
- JAX legacy observable helper vs current supported-path observable
- JAX legacy 8D helper vs current supported-path observable

## Validation Already Completed

### Syntax / Import Level

The following compiled successfully with `py_compile`:

- `jax_ffi/__init__.py`
- `jax_ffi/mw_approx_batch_contract.py`
- `tools/jax_env_check.py`
- `tools/jax_phase0_contract_validate.py`
- `jax_ffi/mw_approx_batch_jax.py`
- `tools/jax_ffi_smoke_test.py`

### Contract Validation

Without a host-compatible library:

- parameter-contract validation passed
- runtime validation was skipped as expected because the checked-in `source/MWTransferArr.so` is Linux ELF and cannot be loaded on macOS

With a host-compatible macOS CPU build:

```bash
/Users/walterwei/miniforge3/envs/gsfit_cp/bin/python tools/jax_phase0_contract_validate.py \
  --lib /tmp/gyrosynchrotron-gpu-jax-build/source/MWTransferArr.so \
  --workload real-sweep \
  --batch-size 8 \
  --backend cpu \
  --precision fp64
```

Passed:

- parameter contract
- observable adapter against the current supported-path oracle

Observed:

- `postprocess_path: fused-native-rl-fast-path`
- `runtime_status_unique: [0]`
- `max_abs ~= 2.98e-08`
- `max_rel ~= 2.777e-16`

Interpretation:

- the contract and observable adapter are aligned with the current validated supported path on the tested workload

## Build Work Done On The Current Machine

The checked-in shared library was not usable on this macOS host because it is Linux ELF, so a temporary host-compatible build was made in:

- `/tmp/gyrosynchrotron-gpu-jax-build/source`

The build path was:

1. copy repo to `/tmp/gyrosynchrotron-gpu-jax-build`
2. build host-compatible native library
3. build new XLA FFI library

Build command that eventually succeeded:

```bash
make -B MWTransferArr MWTransferArrXLA JAX_PYTHON=/Users/walterwei/miniforge3/envs/gsfit_cp/bin/python
```

Artifacts produced there:

- `/tmp/gyrosynchrotron-gpu-jax-build/source/MWTransferArr.so`
- `/tmp/gyrosynchrotron-gpu-jax-build/source/MWTransferArrXLA.so`

## Important Makefile Change

`source/makefile` now includes a separate XLA build target:

- target: `MWTransferArrXLA`

It uses:

- `jaxlib` include headers from the selected Python env
- C++17 for the XLA interface object
- core native objects without `PythonInterface.o`

This is the path to keep for the CUDA machine. Build the reference library and the XLA library as separate artifacts.

## Current Known Issue

The last smoke test failed before the conversation ended, but the most likely cause was identified and patched immediately afterward.

### Failing Command

```bash
JAX_ENABLE_X64=1 /Users/walterwei/miniforge3/envs/gsfit_cp/bin/python tools/jax_ffi_smoke_test.py \
  --lib /tmp/gyrosynchrotron-gpu-jax-build/source/MWTransferArr.so \
  --xla-lib /tmp/gyrosynchrotron-gpu-jax-build/source/MWTransferArrXLA.so \
  --workload real-sweep \
  --batch-size 8
```

### Failure Seen

```text
XlaRuntimeError: INVALID_ARGUMENT: MWApproxBatchRunRL reported non-zero item status 1 at flattened batch index 0
```

### Most Likely Cause

The explicit FFI layout metadata in `jax_ffi/mw_approx_batch_jax.py` was backwards.

Originally:

- `_row_major_layout(ndim)` returned reversed order

That is likely wrong for `jax.ffi.ffi_call(...)`, which expects layout in major-to-minor order.

### Patch Already Applied

`jax_ffi/mw_approx_batch_jax.py` was updated so:

```python
def _row_major_layout(ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim))
```

The smoke test was not rerun after this fix because the conversation was interrupted.

This is the immediate next checkpoint on the CUDA machine.

## Exact Next Steps On The CUDA Machine

### 1. Sync This Branch

Use branch:

- `research/jax-vectorized-forward`

### 2. Confirm JAX Environment

Prefer starting with:

```bash
/Users/walterwei/miniforge3/envs/gsfit_cp/bin/python tools/jax_env_check.py --require-jax --require-ffi --show-devices
```

If the CUDA machine needs a different JAX install, fix that first. The environment must expose:

- `jax`
- `jaxlib`
- `jax.ffi`
- visible GPU devices if GPU execution is intended

### 3. Build Native Libraries On The CUDA Machine

From `source/`:

```bash
make -B MWTransferArr JAX_PYTHON=/Users/walterwei/miniforge3/envs/gsfit_cp/bin/python
make -B MWTransferArrXLA JAX_PYTHON=/Users/walterwei/miniforge3/envs/gsfit_cp/bin/python
```

If you use a different env, keep `JAX_PYTHON` consistent with that env.

### 4. Re-run Phase-0 Contract Validation

```bash
JAX_ENABLE_X64=1 /Users/walterwei/miniforge3/envs/gsfit_cp/bin/python tools/jax_phase0_contract_validate.py \
  --lib /absolute/path/to/MWTransferArr.so \
  --workload real-sweep \
  --batch-size 8 \
  --backend cpu \
  --precision fp64
```

This should still pass before relying on the FFI path.

### 5. Re-run The JAX FFI Smoke Test

```bash
JAX_ENABLE_X64=1 /Users/walterwei/miniforge3/envs/gsfit_cp/bin/python tools/jax_ffi_smoke_test.py \
  --lib /absolute/path/to/MWTransferArr.so \
  --xla-lib /absolute/path/to/MWTransferArrXLA.so \
  --workload real-sweep \
  --batch-size 8
```

This is the first critical checkpoint after the layout fix.

### 6. If The Smoke Test Still Fails

Investigate in this order:

1. confirm the FFI input/output layout semantics in `jax.ffi.ffi_call(...)`
2. temporarily remove explicit `input_layouts` / `output_layouts` to compare behavior
3. add narrow debug instrumentation in `source/XLAInterface.cpp` around the first flattened batch row
4. confirm that attrs are being received as intended
5. only after CPU FFI proof is solid, add the CUDA-target FFI boundary

## Planned Next Milestones After Smoke Test Passes

1. Add CUDA-side FFI registration / target.
2. Validate CPU-native vs CUDA-native through the JAX boundary on the supported packed inputs.
3. Benchmark warm throughput at meaningful batch sizes.
4. Integrate the JAX observable wrapper with the ESS sampler skeleton.
5. Keep pure-JAX physics rewrite out of scope unless gradients become a real requirement.

## Current Git Snapshot At Handoff

At the time of writing, the branch had:

- modified tracked files:
  - `doc/internal/development/JAXVectorizedForwardFeasibility.md`
  - `jax_ffi/mw_approx_batch_jax_stub.py`
  - `source/makefile`
- untracked JAX-work files:
  - `jax_ffi/__init__.py`
  - `jax_ffi/mw_approx_batch_contract.py`
  - `jax_ffi/mw_approx_batch_jax.py`
  - `source/XLAInterface.cpp`
  - `tools/jax_env_check.py`
  - `tools/jax_ffi_smoke_test.py`
  - `tools/jax_phase0_contract_validate.py`

## Short Resume Point

If you want the shortest possible restart instruction on the CUDA machine:

1. use branch `research/jax-vectorized-forward`
2. use env `/Users/walterwei/miniforge3/envs/gsfit_cp`
3. run `tools/jax_env_check.py`
4. build `MWTransferArr` and `MWTransferArrXLA`
5. run `tools/jax_phase0_contract_validate.py`
6. rerun `tools/jax_ffi_smoke_test.py`
7. if it still fails, debug FFI layout/attrs before touching CUDA-specific code
