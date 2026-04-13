# Phase 1 GPU Migration Tooling

This repository now includes Python-side tooling for the evidence-first Phase 1 workflow described in the GPU migration plan.

## Files

- `examples/GScodes.py`
  - Preserves the existing `initGET_MW` and `initGET_MW_SLICE` entry points.
  - Adds a CPU reference batch API for the analytical `PLW + ISO`, `Nz=1` workload that matches the proposed first GPU target shape.
- `tools/phase1_workloads.py`
  - Defines the real wrapper workload builder and the repo array baseline workload.
- `tools/phase1_validate.py`
  - Verifies that the new CPU batch API matches repeated `pyGET_MW` calls exactly.
- `tools/phase1_benchmark.py`
  - Benchmarks `W1-small-real`, `W2-batched-real`, and `W3-secondary-array`.
- `tools/phase1_profile.py`
  - Runs a workload loop in a child process, attaches macOS `sample`, and summarizes runtime shares by Phase 1 category.
- `tools/phase2_validate.py`
  - Validates the new native analytical batch backend. The CPU path is checked against the existing single-call approximate GS-only solver, and the CUDA path compares packed CUDA results against the CPU reference when a CUDA build is available.
- `tools/phase2_benchmark.py`
  - Benchmarks the new native analytical batch backend across realistic batch sizes for CPU and, when available, CUDA FP64/FP32.

## Verified commands

```bash
python3 tools/phase1_validate.py --batch-size 4
python3 tools/phase1_benchmark.py --repeats 2 --batch-sizes 1 8
python3 tools/phase1_profile.py --scenario real_wrapper --sample-seconds 3 --work-seconds 6
python3 tools/phase1_profile.py --scenario array_baseline --sample-seconds 3 --work-seconds 6
python3 tools/phase2_validate.py --target cpu-reference --batch-size 4
python3 tools/phase2_benchmark.py --backend cpu --precision fp64 --repeats 2 --batch-sizes 1 8 32 128
cd source
make CUDA=1 MWTransferArr
cd ..
python3 tools/phase2_validate.py --target cuda-fp64 --workload real-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_validate.py --target cuda-fp64 --workload stress-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_validate.py --target wrapper-cuda-fp64 --workload real-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_validate.py --target wrapper-cuda-fp64 --workload stress-sweep --suite-batch-sizes 1 8 32 128 512
python3 tools/phase2_benchmark.py --mode compare-fp64 --workload real-sweep --repeats 3 --batch-sizes 1 8 32 128 512
```

## Notes

- The batch API started as a CPU reference path to lock the interface, packing, and validation behavior before CUDA work.
- The native batch API now exists in the shared library as `pyMW_Approx_Batch`, with a CPU reference backend enabled by default and an optional CUDA backend gated behind `make CUDA=1 MWTransferArr`.
- On the NVIDIA-enabled WSL machine, the supported CUDA `FP64` milestone now passes after fixing a narrow O-mode corrected-root mismatch in the CUDA Brent root path.
- This validated `FP64` state is the frozen baseline for the new `gyrosynchrotron-gpu` repo and for any follow-on optimization branches.
- Post-fix CUDA `FP64` validation maxima on the supported path:
  - local real-sweep `jo`: max rel `1.755e-11`
  - local stress-sweep `jo`: max rel `4.520e-11`
  - wrapper real-sweep `left`: max rel `1.504e-09`
  - wrapper stress-sweep `left`: max rel `2.445e-10`
- Post-fix CUDA `FP64` wrapper speedup vs legacy single-call CPU wrapper:
  - `2.71x` at batch `8`
  - `8.39x` at batch `32`
  - `16.04x` at batch `128`
  - `19.71x` at batch `512`
- CUDA `FP32` remains evaluation-only after narrow stability fixes:
  - non-finite failures are removed
  - stress-sweep wrapper maxima are finite but still non-negligible
  - real-sweep wrapper maxima are still too large to accept `FP32` as an optional mode yet
- The measured wrapper/backend gap on the supported path is dominated by Python-side `RL` postprocessing rather than by the Python/native call boundary or GPU transfer costs.
- `tools/phase1_profile.py` depends on the macOS `sample` tool for category shares. If `sample` is unavailable or blocked, use `tools/phase1_benchmark.py` for timing-only evidence.
- The scripts default to the locally built `source/MWTransferArr.so` when present and fall back to the prebuilt binaries directory otherwise.
