# Build Libraries

## What gets built

The main local build output is:

- `source/MWTransferArr.so`

That shared library preserves the legacy exports and, in this repo, also includes the additive batched exports used by the GPU-oriented Python workflow:

- `pyGET_MW`
- `pyGET_MW_SLICE`
- `pyMW_Approx_Batch`
- `pyMW_Approx_Batch_RL`

Optional local convenience copies can be made in `binaries/`, but the recommended workflow is to build `source/MWTransferArr.so` locally on the target machine.

## Scope reminder

The new batched GPU-oriented path is validated only for:

- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched Python workflow

The legacy solver remains in the library, but the newer validation and benchmark tooling is intentionally scoped to that narrow path.

## Linux / WSL CPU-only build

Requirements:

- `make`
- `g++`
- OpenMP-capable toolchain
- Python for validation scripts

Build from the repo root:

```bash
cd source
make clean
make MWTransferArr
cd ..
```

## Linux / WSL CUDA build

Requirements:

- all CPU-only requirements
- NVIDIA driver/runtime visible on the machine
- `nvcc` in `PATH`
- CUDA runtime libraries visible to the linker

Build from the repo root:

```bash
cd source
make clean
make CUDA=1 MWTransferArr
cd ..
```

If `CUDA=1` is requested and `nvcc` is unavailable, the build should fail fast.

## Windows build

Open [source/gscodes.vcxproj](../../source/gscodes.vcxproj) in Visual Studio and build the shared-library target there.

## Optional local copies

If you want stable machine-specific copies in `binaries/`, copy them explicitly after a successful build:

```bash
cp source/MWTransferArr.so binaries/MWTransferArr_cpu_only_linux_x86_64.so
cp source/MWTransferArr.buildinfo binaries/MWTransferArr_cpu_only_linux_x86_64.buildinfo
```

or, for a CUDA build:

```bash
cp source/MWTransferArr.so binaries/MWTransferArr_cuda_linux_x86_64.so
cp source/MWTransferArr.buildinfo binaries/MWTransferArr_cuda_linux_x86_64.buildinfo
```

These Linux convenience copies are treated as local build artifacts, not as the main published interface.

## Recommended verification

CPU-only:

```bash
python3 -m py_compile examples/GScodes.py tools/phase2_validate.py tools/phase2_benchmark.py
python3 tools/phase2_validate.py --target cpu-reference --batch-size 4
python3 tools/phase2_benchmark.py --backend cpu --precision fp64 --repeats 2 --batch-sizes 1 8 32 128
```

CUDA:

```bash
python3 tools/phase2_validate.py --target cuda-fp64 --batch-size 4
python3 tools/phase2_validate.py --target wrapper-cuda-fp64 --batch-size 4
python3 tools/phase2_benchmark.py --mode compare-fp64 --workload real-sweep --repeats 3 --batch-sizes 1 8 32 128 512
```

## Next step

For Python usage, see [CallingFromPython.md](CallingFromPython.md).
