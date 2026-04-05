# Project Instructions

These instructions are the stable constraints for this repository until explicitly revised.

## Scope

- Python only. IDL is out of scope.
- Preserve the existing single-call CPU path unchanged.
- Keep the current narrow Phase 1 target only:
  - analytical `PLW + ISO`
  - `Nz=1`
  - approximate GS only
  - batched workflow first
- Do not broaden yet to:
  - exact GS
  - array DF
  - transport
  - IDL
  - broader API redesign

## Design Constraints

- The new native batched path is additive. It must not replace or mutate the legacy `pyGET_MW` / `pyGET_MW_SLICE` behavior.
- Unsupported cases must fail loudly.
- The CPU reference backend is authoritative for validation.
- `FP64` is the correctness anchor.
- `FP32` is an evaluation mode only. It is not the primary correctness path.
- Prefer backend-neutral interfaces at the native boundary. Keep CUDA-specific logic isolated.
- Keep changes incremental and reviewable. Do not start broad rewrites.

## Current Supported Native Batch Path

- Python entry point: `pyMW_Approx_Batch`
- Supported workload only:
  - analytical power-law energy distribution (`PLW`)
  - isotropic pitch-angle distribution (`ISO`)
  - one voxel (`Nz=1`)
  - approximate GS branch
  - batched inputs
- Current outputs are local approximate GS quantities for both modes:
  - `jX`
  - `kX`
  - `jO`
  - `kO`

## Validation Policy

- Validate the narrow native batch backend against the CPU reference on identical packed inputs.
- Use `FP64` CPU-native results as the immediate reference for CUDA validation.
- Do not claim CUDA readiness without:
  - successful native CUDA build
  - CPU-vs-CUDA validation on the supported packed inputs
  - benchmarks across realistic batch sizes

## Files To Treat As Current Interfaces

- Python batch wrapper and helpers:
  - `examples/GScodes.py`
- Native batch interface:
  - `source/ApproxBatch.h`
  - `source/PythonInterface.cpp`
- CPU reference backend:
  - `source/ApproxBatchCpu.cpp`
- CUDA backend:
  - `source/ApproxBatchCuda.cu`
- Validation and benchmarks:
  - `tools/phase1_validate.py`
  - `tools/phase1_benchmark.py`
  - `tools/phase1_profile.py`
  - `tools/phase2_validate.py`
  - `tools/phase2_benchmark.py`
