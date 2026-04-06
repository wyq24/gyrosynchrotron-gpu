# Build Libraries

## Scope

This build guide is only for the current validated narrow path:

- Python only
- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched workflow first

It does not broaden support to exact GS, array DF, transport, IDL, or a broader API redesign.

## Outputs

Primary build output:

- `source/MWTransferArr.so`

Optional convenience copies:

- `binaries/MWTransferArr_cpu_only_linux_x86_64.so`
- `binaries/MWTransferArr_cuda_linux_x86_64.so`

## Supported Build Environments

### CPU-only Linux / WSL

Required:

- Linux or WSL
- `g++`
- OpenMP-capable toolchain
- `make`
- Python for validation scripts

### CUDA Linux / WSL

Required:

- all CPU-only requirements
- NVIDIA driver/runtime visible to the machine
- `nvcc` in `PATH`
- CUDA runtime library visible to the linker

## Build Commands

From repo root:

### CPU-only build

```bash
cd source
make clean
make MWTransferArr
cd ..
```

This produces:

- `source/MWTransferArr.so`

Optional stable copy:

```bash
cp source/MWTransferArr.so binaries/MWTransferArr_cpu_only_linux_x86_64.so
cp source/MWTransferArr.buildinfo binaries/MWTransferArr_cpu_only_linux_x86_64.buildinfo
```

### CUDA build

```bash
cd source
make clean
make CUDA=1 MWTransferArr
cd ..
```

This produces:

- `source/MWTransferArr.so`

Optional stable copy:

```bash
cp source/MWTransferArr.so binaries/MWTransferArr_cuda_linux_x86_64.so
cp source/MWTransferArr.buildinfo binaries/MWTransferArr_cuda_linux_x86_64.buildinfo
```

## Build Behavior

- `make MWTransferArr` builds the CPU/reference-capable shared library.
- `make CUDA=1 MWTransferArr` adds the CUDA backend.
- CUDA is optional.
- If `CUDA=1` is requested and `nvcc` is not available, the build is expected to fail fast.

## Recommended Verification

### CPU-only machine

```bash
python3 -m py_compile examples/GScodes.py tools/phase2_validate.py tools/phase2_benchmark.py
python3 tools/phase2_validate.py --target cpu-reference --batch-size 4
python3 tools/phase2_benchmark.py --mode native-backend --backend cpu --precision fp64 --workload real-sweep --repeats 2 --batch-sizes 8 32
```

### CUDA machine

```bash
python3 tools/phase2_validate.py --target cuda-fp64 --batch-size 4
python3 tools/phase2_validate.py --target wrapper-cuda-fp64 --batch-size 4
python3 tools/phase2_benchmark.py --mode compare-integrated-fp64 --workload real-sweep --repeats 3 --batch-sizes 8 32 128 512
```

## Runtime Notes

- The shared-library interface is still the same `.so` model.
- Existing exports remain available:
  - `pyGET_MW`
  - `pyMW_Approx_Batch`
  - `pyMW_Approx_Batch_RL`
- Recommended validated Python workflow call is documented in:
  - `doc/CallingEfficiently.md`

## Important Constraints

- `FP64` is the correctness anchor.
- CPU reference behavior remains authoritative.
- `FP32` remains evaluation-only.
- The fused native RL fast path is only used automatically on the validated narrow `FP64` batch workflow.
