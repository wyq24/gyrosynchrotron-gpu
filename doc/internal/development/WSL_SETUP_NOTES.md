# WSL Setup Notes

## Purpose

These notes are only for resuming this project on WSL or another Linux environment after the current Mac-based handoff.

## What Can Be Tested Without NVIDIA

These parts should work on any Linux/WSL environment with a working C++ toolchain and Python:

- Python import sanity for:
  - `examples/GScodes.py`
- Phase 1 scripts:
  - `tools/phase1_validate.py`
  - `tools/phase1_benchmark.py`
  - `tools/phase1_profile.py` only if an equivalent sampling setup is available
- Phase 2 CPU-native path:
  - `tools/phase2_validate.py --target cpu-reference --batch-size 4`
  - `tools/phase2_benchmark.py --backend cpu --precision fp64 ...`
- Non-CUDA native build:
  - `cd source && make MWTransferArr`

## What Requires A Real NVIDIA Environment

These require actual NVIDIA driver/runtime support and a CUDA toolkit:

- `make CUDA=1 MWTransferArr`
- `source/ApproxBatchCuda.cu`
- `pyMW_Approx_Batch` with backend `cuda`
- `tools/phase2_validate.py --target cuda-fp64 --batch-size 4`
- `tools/phase2_validate.py --target cuda-fp32 --batch-size 4`
- `tools/phase2_benchmark.py --backend cuda ...`

## Commands To Run First On The New Machine

From the repo root:

```bash
python3 -m py_compile examples/GScodes.py tools/phase1_workloads.py tools/phase1_validate.py tools/phase1_benchmark.py tools/phase1_profile.py tools/phase2_validate.py tools/phase2_benchmark.py
cd source
make MWTransferArr
cd ..
python3 tools/phase1_validate.py --batch-size 4
python3 tools/phase2_validate.py --target cpu-reference --batch-size 4
python3 tools/phase2_benchmark.py --backend cpu --precision fp64 --repeats 2 --batch-sizes 1 8 32 128
```

If the machine has real NVIDIA support:

```bash
which nvcc
nvidia-smi
cd source
make CUDA=1 MWTransferArr
cd ..
python3 tools/phase2_validate.py --target cuda-fp64 --batch-size 4
python3 tools/phase2_benchmark.py --backend cuda --precision fp64 --repeats 3 --batch-sizes 1 8 32 128 256 1024
python3 tools/phase2_validate.py --target cuda-fp32 --batch-size 4
python3 tools/phase2_benchmark.py --backend cuda --precision fp32 --repeats 3 --batch-sizes 1 8 32 128 256 1024
```

The supported-path CUDA `FP64` milestone is now verified on the current NVIDIA-enabled WSL machine.

Verified CUDA `FP64` milestone commands:

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

Observed CUDA `FP64` result summary:

- local real-sweep `jo`: max rel `1.755e-11`
- local stress-sweep `jo`: max rel `4.520e-11`
- wrapper real-sweep `left`: max rel `1.504e-09`
- wrapper stress-sweep `left`: max rel `2.445e-10`
- wrapper speedup vs legacy single-call CPU wrapper:
  - `2.71x` at batch `8`
  - `8.39x` at batch `32`
  - `16.04x` at batch `128`
  - `19.71x` at batch `512`

## Expected Build Limitations By Platform

### Mac

- Local implementation was done on macOS arm64.
- No CUDA toolkit was available.
- No NVIDIA runtime was available.
- The CUDA backend could not be compiled or executed there.

### WSL / Linux Without NVIDIA

- CPU-only build and validation should be possible.
- CUDA-specific milestones will remain blocked.
- `tools/phase2_validate.py --target cuda-fp64` should be expected to report CUDA unavailable unless the library was built with CUDA and a real runtime is present.

### NVIDIA Linux / WSL With CUDA

- This is now the verified platform for the supported CUDA `FP64` milestone.
- The next bounded step on this machine is `FP32` evaluation only on the same narrow path.
- Do not broaden scope if `FP32` exposes integration or stability gaps. Fix the current narrow path only.

## Path And Build Assumptions To Verify Immediately

- Verify the repo path used by scripts is correct and synced fully.
- Verify `source/MWTransferArr.so` is rebuilt locally instead of trusting a copied binary from macOS.
- Verify the compiler and OpenMP flags in `source/makefile`.
- Verify whether `clang++` or `g++` is the correct host compiler on WSL.
- Verify CUDA include and library discovery for `nvcc`.
- Verify that `python3` resolves to the environment intended for the repo scripts.
- Verify that the external `pygsfit` tree is not required for the in-repo validation scripts. It was not modified in this handoff.

## Immediate Sanity Checks For The New Session

- Confirm `AGENTS.md`, `doc/internal/development/Handoff.md`, `doc/internal/development/PLANS.md`, and `doc/internal/development/WSL_SETUP_NOTES.md` are present.
- Confirm the narrow native batch files exist:
  - `source/ApproxBatch.h`
  - `source/ApproxBatch.cpp`
  - `source/ApproxBatchCpu.cpp`
  - `source/ApproxBatchCuda.cu`
  - `source/ApproxBatchCudaStub.cpp`
- Confirm the Python entry points exist in:
  - `examples/GScodes.py`
- Confirm the current validation scripts exist in:
  - `tools/phase1_validate.py`
  - `tools/phase2_validate.py`
- Confirm the current benchmark scripts exist in:
  - `tools/phase1_benchmark.py`
  - `tools/phase2_benchmark.py`
