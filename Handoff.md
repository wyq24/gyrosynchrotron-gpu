# Handoff

## Summary

This repo now has:

- the original single-call native CPU solver path preserved
- Python-side Phase 1 workload, profiling, validation, and benchmark tooling
- a new narrow native batch ABI for the real Python workload
- a CPU reference backend under that ABI
- a working opt-in CUDA backend under that ABI for the supported narrow path
- Python wrappers for the new native batch interface

No broad CUDA migration has been started beyond that narrow path.

Frozen baseline status:

- validated scope is Python-only analytical `PLW + ISO`, `Nz=1`, approximate GS, batched workflow first
- CUDA `FP64` passes validation on that supported path and remains the correctness anchor
- CUDA `FP32` is finite after narrow stability fixes, but it is still evaluation-only and not accepted yet
- the legacy-compatible single-call CPU Python wrapper path remains intact

## What Has Been Implemented

### Phase 1 Python Tooling

- Added workload packing for the real analytical `PLW + ISO`, `Nz=1` workflow in:
  - `examples/GScodes.py`
- Added profiling, validation, and benchmark scripts:
  - `tools/phase1_workloads.py`
  - `tools/phase1_validate.py`
  - `tools/phase1_benchmark.py`
  - `tools/phase1_profile.py`
- Added documentation:
  - `doc/Phase1GPU.md`

### Native Batch Interface

- Added backend-neutral native batch interface:
  - `source/ApproxBatch.h`
  - `source/ApproxBatchInternal.h`
  - `source/ApproxBatch.cpp`
- Added CPU reference backend:
  - `source/ApproxBatchCpu.cpp`
- Added CUDA backend source:
  - `source/ApproxBatchCuda.cu`
- Added CUDA-unavailable stub for non-CUDA builds:
  - `source/ApproxBatchCudaStub.cpp`
- Exported the native batch ABI to Python:
  - `source/PythonInterface.cpp`
- Exposed the new native path from Python:
  - `examples/GScodes.py`
- Updated build wiring:
  - `source/makefile`

### Phase 2 Validation / Benchmark Tooling

- Added native batch validation script:
  - `tools/phase2_validate.py`
- Added native batch benchmark script:
  - `tools/phase2_benchmark.py`

## Current Architecture

- Legacy path remains:
  - `pyGET_MW` / `pyGET_MW_SLICE`
  - full legacy solver path
  - unchanged by this work
- New narrow path:
  - Python packs batch inputs in `examples/GScodes.py`
  - Python calls `pyMW_Approx_Batch`
  - native dispatch goes through `MWApproxBatchRun`
  - CPU backend uses `ApproxBatchCpu.cpp`
  - CUDA backend is intended to use `ApproxBatchCuda.cu`
- Current native batch outputs are local approximate GS arrays:
  - `freq_hz`
  - `jX`
  - `kX`
  - `jO`
  - `kO`
- Python can convert those local outputs into single-voxel `RL`-style `L/R` intensities for comparison using:
  - `local_jk_to_single_voxel_rl`

## What Has Been Validated

### Phase 1

- `tools/phase1_validate.py` passed against repeated single `pyGET_MW` calls.
- `tools/phase1_profile.py` reproduced the expected hotspot split:
  - real wrapper path dominated by approximate GS
  - repo array baseline dominated by exact GS plus array DF
- `tools/phase1_benchmark.py` produced stable CPU batch timings for the real wrapper-shaped workload.

### Phase 2

- Local Linux/WSL rebuild of `source/MWTransferArr.so` succeeds for CPU and CUDA builds.
- `tools/phase2_validate.py --target cpu-reference --batch-size 4` passes.
- The CPU reference validation compares the supported rows:
  - frequency row
  - left intensity row
  - right intensity row
- Those rows match exactly for the supported approximate GS-only, single-voxel path.
- `make CUDA=1 MWTransferArr` now succeeds on the NVIDIA-enabled WSL machine.
- CUDA `FP64` now passes the narrow supported-path numerical gate after fixing a CUDA-only O-mode corrected-root mismatch in `ApproxBatchCuda.cu`.
- End-to-end wrapper validation on the real Python workflow also passes on the supported path.

Observed CPU-native benchmark output:

- batch `1`: `0.004677s` median, `213.79` spectra/s
- batch `8`: `0.023051s` median, `347.05` spectra/s
- batch `32`: `0.086109s` median, `371.62` spectra/s
- batch `128`: `0.341702s` median, `374.60` spectra/s
- batch `512`: `1.355203s` median, `377.80` spectra/s

Observed CUDA `FP64` benchmark output:

- backend, batch `1`: `0.009285s` median, `107.70` spectra/s
- backend, batch `8`: `0.007373s` median, `1085.10` spectra/s
- backend, batch `32`: `0.009177s` median, `3486.98` spectra/s
- backend, batch `128`: `0.017939s` median, `7135.17` spectra/s
- backend, batch `512`: `0.052406s` median, `9769.95` spectra/s
- wrapper, batch `1`: `0.007589s` median, `131.77` spectra/s
- wrapper, batch `8`: `0.008368s` median, `956.00` spectra/s
- wrapper, batch `32`: `0.011400s` median, `2806.92` spectra/s
- wrapper, batch `128`: `0.024944s` median, `5131.41` spectra/s
- wrapper, batch `512`: `0.073392s` median, `6976.28` spectra/s

Observed CUDA `FP64` speedup vs legacy single-call CPU wrapper:

- backend: `3.07x` at batch `8`, `10.42x` at batch `32`, `22.31x` at batch `128`, `27.60x` at batch `512`
- wrapper: `2.71x` at batch `8`, `8.39x` at batch `32`, `16.04x` at batch `128`, `19.71x` at batch `512`

Observed CUDA `FP64` validation maxima after the O-mode fix:

- local real-sweep:
  - `jx`: max rel `1.485e-09`
  - `jo`: max rel `1.755e-11`, max abs `1.951e-28`
- local stress-sweep:
  - `jx`: max rel `1.661e-11`
  - `jo`: max rel `4.520e-11`, max abs `3.719e-27`
- wrapper real-sweep:
  - `left`: max rel `1.504e-09`, max abs `5.853e-12`
  - `right`: max rel `1.525e-09`
- wrapper stress-sweep:
  - `left`: max rel `2.445e-10`, max abs `8.163e-11`
  - `right`: max rel `3.904e-11`

## What Has NOT Been Validated Yet

- `FP32` has not been accepted yet as an optional mode.
- No follow-up fused native `RL`/postprocessing path has been validated yet.
- No additional selective mixed-precision follow-up has been validated yet.
- End-to-end validation against the external `pygsfit` wrapper on the new native batch path has not been done.

## What Remains Out Of Scope

- exact GS
- array DF
- transport
- IDL
- broad public API redesign
- changing the external `pygsfit` wrapper

## Current Freeze Status

- The current narrow CUDA `FP64` baseline is frozen and suitable to publish as the preserved starting point for follow-on optimization branches.
- The validated baseline should be kept intact while optimization work proceeds additively in separate branches.

## Exact Next Milestone

From the frozen baseline, start only one additive optimization branch at a time:

- either reduce the wrapper/backend gap with a fused native `RL`/postprocessing batch path
- or continue the narrow `FP32` stabilization effort on the same supported path

Do not broaden scope beyond the current supported narrow path while doing this, and keep CUDA `FP64` plus the CPU reference as the correctness anchors.

## What Not To Work On Next

- Do not broaden into exact GS.
- Do not broaden into array DF.
- Do not broaden into transport.
- Do not redesign the public API broadly.
- Do not edit the external `pygsfit` wrapper unless explicitly requested later.
- Do not replace the legacy `pyGET_MW` path.
- Do not start broad code cleanup unrelated to the current milestone.

## Known Assumptions

- Python is the only in-scope integration surface.
- The real first target remains:
  - analytical `PLW + ISO`
  - `Nz=1`
  - approximate GS only
  - batched workflow first
- CPU reference behavior is authoritative.
- `FP64` correctness comes first.
- `FP32` is only an evaluation mode after `FP64` is working.
- Unsupported cases should fail loudly rather than silently falling back into broader solver behavior.
