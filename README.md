# gyrosynchrotron-gpu

This repository is a GPU-oriented continuation of Alexey Kuznetsov's gyrosynchrotron shared-library codebase. It preserves the original shared-library model and the legacy single-call interfaces, then adds an additive batched native path for a narrow validated Python workflow.

## Original vs current repo

The original project is a general shared-library solver callable from Python or IDL, with broader legacy functionality and examples.

The current `gyrosynchrotron-gpu` repo keeps that baseline intact and adds:

- a native batched interface for the validated analytical `PLW + ISO`, `Nz=1`, approximate-GS workflow
- a CPU reference backend for that batch ABI
- an optional CUDA backend for the same narrow path
- validation and benchmark tooling for the new batch path
- a higher-throughput Python wrapper around the native batch interface

The legacy single-call path is still present and is not replaced by the GPU work.

## Current validated scope

- Python-first workflow
- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched workflow first
- CPU reference and `FP64` are the correctness anchors

Out of scope for the current GPU path:

- exact GS
- array DF
- transport beyond the existing supported wrapper path
- broader API redesign

## Repository guide

- Public docs start in [doc/README.md](doc/README.md)
- Build instructions: [doc/public/BuildLibraries.md](doc/public/BuildLibraries.md)
- Python calling guide: [doc/public/CallingFromPython.md](doc/public/CallingFromPython.md)
- Batched MCMC guide: [doc/public/GpuBatchedMCMC.md](doc/public/GpuBatchedMCMC.md)
- Original reference PDFs: [doc/reference/CallingConventions.pdf](doc/reference/CallingConventions.pdf), [doc/reference/AnalyticalDistributions.pdf](doc/reference/AnalyticalDistributions.pdf), [doc/reference/Diagram.pdf](doc/reference/Diagram.pdf)
- Internal development history is intentionally deeper under [doc/internal/](doc/internal/)

## Build quick start

CPU-only Linux / WSL build:

```bash
cd source
make clean
make MWTransferArr
cd ..
```

CUDA-enabled Linux / WSL build:

```bash
cd source
make clean
make CUDA=1 MWTransferArr
cd ..
```

Windows build:

- open `source/gscodes.vcxproj` in Visual Studio

The main local build output is `source/MWTransferArr.so`. More detail is in [doc/public/BuildLibraries.md](doc/public/BuildLibraries.md).

## Python quick start

For the current high-throughput validated path:

```python
from examples import GScodes

lib = GScodes.resolve_library_path(prefer_source=True)
batch = GScodes.build_wrapper_powerlaw_iso_batch(batch_size=128)

result = GScodes.run_powerlaw_iso_batch_wrapper(
    lib,
    batch,
    backend="cpu",
    precision="fp64",
    npoints=16,
    q_on=True,
)

print(result.rl.shape)
print(result.postprocess_path)
```

After a CUDA build, change `backend="cpu"` to `backend="cuda"`.

For the original single-call path, see the examples in `examples/` and the legacy exports `pyGET_MW` / `pyGET_MW_SLICE`.
