# Calling From Python

## Shared library location

The recommended local build output is `source/MWTransferArr.so`.

In Python, prefer resolving the library through `examples/GScodes.py` instead of hard-coding a platform-specific binary path:

```python
from examples import GScodes

lib = GScodes.resolve_library_path(prefer_source=True)
```

## Recommended batched workflow

For the current validated high-throughput path, build a batch object and call the wrapper helper:

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

After a CUDA build, set `backend="cuda"`.

This is the main public entry point for the additive batched path. It keeps the legacy library model but routes the supported narrow workflow through the newer native batch ABI.

## Current supported batch scope

The newer batched path is intentionally narrow:

- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched Python workflow

Unsupported cases should fail loudly rather than silently broadening behavior.

## What the wrapper returns

`run_powerlaw_iso_batch_wrapper(...)` returns a `BatchRunResult` with:

- `rl`
- `native_freq_hz`
- `status`
- `postprocess_path`
- `postprocess_reason`

The wrapper automatically chooses the additive native RL path when the validated fast-path conditions are met, and otherwise falls back to local `j/k` outputs plus Python-side RL postprocessing.

## Lower-level entry points

If you want the local approximate-GS quantities directly, call:

```python
kernel = GScodes.run_powerlaw_iso_batch_native(
    lib,
    batch,
    backend="cpu",
    precision="fp64",
)

print(kernel.jx.shape)
print(kernel.jo.shape)
```

That returns `jX`, `kX`, `jO`, and `kO` for the supported batch.

There is also a direct `run_powerlaw_iso_batch_native_rl(...)` helper for the additive RL export, but most callers should use the wrapper unless they are specifically validating or profiling the lower-level interface.

## Preserved legacy path

The original single-call workflow is still available through the legacy exports:

- `pyGET_MW`
- `pyGET_MW_SLICE`

For those calls, see the example scripts in `examples/`, especially:

- `Example_Analytical_SingleThread.py`
- `Example_Analytical_MultiThreads.py`
- `Example_Array_SingleThread.py`
- `Example_Array_MultiThreads.py`

## Precision note

`FP64` is the correctness anchor for the batched path.

`FP32` is exposed for evaluation, but it is not the primary correctness path.
