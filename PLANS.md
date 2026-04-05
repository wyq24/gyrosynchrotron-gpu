# Plans

## Completed Milestones

### Milestone 1: Phase 1 Workload Lock

Completed.

Delivered:

- real Python workload model for the analytical `PLW + ISO`, `Nz=1` path
- profiling path for the real workload and repo array baseline
- CPU batch packing and CPU reference batch API in Python
- Phase 1 validation and benchmark scripts

Primary files:

- `examples/GScodes.py`
- `tools/phase1_workloads.py`
- `tools/phase1_validate.py`
- `tools/phase1_benchmark.py`
- `tools/phase1_profile.py`

### Milestone 2: Native Batch Interface Skeleton

Completed for CPU and source-level CUDA handoff.

Delivered:

- backend-neutral native batch ABI
- CPU reference backend
- Python export and Python wrapper
- CUDA backend source file plus non-CUDA stub
- build-system hook for `CUDA=1`

Primary files:

- `source/ApproxBatch.h`
- `source/ApproxBatch.cpp`
- `source/ApproxBatchCpu.cpp`
- `source/ApproxBatchCuda.cu`
- `source/ApproxBatchCudaStub.cpp`
- `source/PythonInterface.cpp`

## Current Milestone

### Milestone 4: Frozen Narrow CUDA FP64 Baseline

Status: complete and ready to publish as the preserved baseline.

Scope:

- preserve the exact current narrow target only
- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched workflow first
- keep CUDA `FP64` as the correctness anchor
- keep CUDA `FP32` evaluation-only and not yet accepted
- keep the legacy-compatible Python wrapper intact

## Next Milestone

### Milestone 5: Additive Optimization Branch Selection

Only start this from the frozen Milestone 4 baseline.

Possible outcomes:

- `opt/fused-native-rl`
- `opt/fp32-salvage`
- work only one branch at a time, after ROI comparison

Current branch decision:

- `opt/fp32-salvage` is paused as a diagnostic branch
- catastrophic FP32 failure classes were removed, but `FP32` is still not acceptable
- the remaining source is the corrected-root / `FindMu0` / `H1` / `lnq2` path
- a local selective mixed-precision attempt regressed and was reverted
- any further FP32 work would require broader mixed precision and is not the current priority
- active attention moves to `opt/fused-native-rl`

## Acceptance Criteria

### For The Current Milestone

- `make CUDA=1 MWTransferArr` still succeeds on the WSL/NVIDIA machine.
- `tools/phase2_validate.py --target cuda-fp64 --batch-size 4` succeeds.
- the full supported-path CUDA `FP64` validation suites pass for local and wrapper outputs.
- the supported-path wrapper/backend benchmark comparison exists for CUDA `FP64`.
- the freeze documentation records validated scope, non-validated scope, and out-of-scope items.
- CUDA `FP32` remains explicitly marked as evaluation-only and not yet acceptable.

### Numerical Acceptance

- CPU reference and CUDA `FP64` remain the correctness anchors.
- `FP32` does not replace the `FP64` gate.
- frozen-baseline claim is limited to the validated narrow CUDA `FP64` path.

### Performance Acceptance

- For the additive batched API:
  - measure CUDA `FP64` backend speedup vs legacy single-call CPU wrapper
  - measure CUDA `FP64` wrapper speedup vs legacy single-call CPU wrapper
  - characterize wrapper/backend gap conservatively
- If future optimization work is started, keep it additive and branch-isolated.

## Validation Commands

Run from repo root unless noted.

### Phase 1 Baselines

```bash
python3 tools/phase1_validate.py --batch-size 4
python3 tools/phase1_benchmark.py --repeats 2 --batch-sizes 1 8
python3 tools/phase1_profile.py --scenario real_wrapper --sample-seconds 3 --work-seconds 6
python3 tools/phase1_profile.py --scenario array_baseline --sample-seconds 3 --work-seconds 6
```

### Phase 2 CPU

```bash
python3 tools/phase2_validate.py --target cpu-reference --batch-size 4
python3 tools/phase2_benchmark.py --backend cpu --precision fp64 --repeats 2 --batch-sizes 1 8 32 128
```

### Phase 2 CUDA Freeze

```bash
cd source
make CUDA=1 MWTransferArr
cd ..
python3 tools/phase2_validate.py --target cuda-fp64 --workload real-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_validate.py --target cuda-fp64 --workload stress-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_validate.py --target wrapper-cuda-fp64 --workload real-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_validate.py --target wrapper-cuda-fp64 --workload stress-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_benchmark.py --mode compare-fp64 --workload real-sweep --repeats 3 --batch-sizes 1 8 32 128 512
```

### Milestone 3 Freeze Evidence

- `make CUDA=1 MWTransferArr` succeeds on the NVIDIA-enabled WSL machine.
- CUDA `FP64` now passes the supported-path numerical gate after fixing the narrow O-mode corrected-root divergence.
- Real-sweep local `jo`: max rel `1.755e-11`
- Stress-sweep local `jo`: max rel `4.520e-11`
- Real-sweep wrapper `left`: max rel `1.504e-09`
- Stress-sweep wrapper `left`: max rel `2.445e-10`
- CUDA `FP64` wrapper speedup vs legacy single-call CPU wrapper:
  - `2.71x` at batch `8`
  - `8.39x` at batch `32`
  - `16.04x` at batch `128`
  - `19.71x` at batch `512`
- CUDA `FP32` is finite after narrow stability fixes, but it is still not accepted:
  - real-sweep wrapper `left`: max rel `1.668e-01`
  - real-sweep wrapper `right`: max rel `1.158e-01`
  - stress-sweep wrapper `left`: max rel `7.342e-03`
  - stress-sweep wrapper `right`: max rel `3.239e-03`
- FP32 diagnosis stop point:
  - remaining error traced to corrected-root precision sensitivity, not the wrapper itself
  - a root-only local mixed-precision attempt regressed and was reverted
  - no retained code change was accepted on `opt/fp32-salvage`

## Benchmark Commands

Current local CPU benchmark command:

```bash
python3 tools/phase2_benchmark.py --backend cpu --precision fp64 --repeats 2 --batch-sizes 1 8 32 128
```

Planned WSL/NVIDIA benchmark command:

```bash
python3 tools/phase2_benchmark.py --backend cuda --precision fp64 --repeats 3 --batch-sizes 1 8 32 128 256 1024
```

## Stop Conditions / No-Go Conditions

- Stop if `FP32` shows unexpected `inf`/`nan` or severe instability on the supported path.
- Stop if `FP32` does not offer a worthwhile performance tradeoff against CUDA `FP64`.
- Stop if the remaining FP32 issue requires broader mixed precision in nonlinear corrected-root logic.
- Stop if supporting the current narrow path would require broadening into exact GS, array DF, transport, or API redesign.
- Stop if validation requires modifying the legacy single-call path.
