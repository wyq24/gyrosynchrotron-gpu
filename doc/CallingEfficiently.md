# Efficient Calling

## Shared Library

Yes, the code is still delivered as the same shared-library style interface.

Primary local build path:

- `source/MWTransferArr.so`

Convenience build copies produced locally on this machine:

- `binaries/MWTransferArr_cpu_only_linux_x86_64.so`
- `binaries/MWTransferArr_cuda_linux_x86_64.so`

Existing interfaces remain available:

- legacy single-call wrapper: `pyGET_MW`
- native local batch interface: `pyMW_Approx_Batch`
- additive native RL batch interface: `pyMW_Approx_Batch_RL`

## Recommended Python Call

For the currently validated narrow high-throughput path, use:

```python
from examples import GScodes

lib = "source/MWTransferArr.so"
batch = GScodes.build_wrapper_powerlaw_iso_batch(batch_size=128)

result = GScodes.run_powerlaw_iso_batch_wrapper(
    lib,
    batch,
    backend="cuda",
    precision="fp64",
    npoints=16,
    q_on=True,
)
```

This keeps the calling convention compatible at the Python level and automatically selects the fused native RL fast path only when the validated conditions are met.

Outputs:

- `result.rl`
- `result.native_freq_hz`
- `result.status`
- `result.postprocess_path`
- `result.postprocess_reason`

## Fast Path Conditions

The fused native RL fast path is used only when all of these are true:

- batch type is `AnalyticalPowerLawIsoBatch`
- `backend="cuda"` or `backend="cpu"` on the supported batch path
- `precision="fp64"`
- `npoints=16`
- `q_on=True`
- `d_sun_au=1.0`
- `nu_cr_factor=0.0`
- `nu_wh_factor=0.0`

Otherwise the workflow falls back to the existing native-local-`j/k` plus Python RL postprocess path.

Example inspection:

```python
print(result.postprocess_path)
print(result.postprocess_reason)
```

## When To Use Other Calls

Use `run_powerlaw_iso_batch_native(...)` if you explicitly want local approximate-GS mode outputs:

- `jX`
- `kX`
- `jO`
- `kO`

Use `run_powerlaw_iso_batch_native_rl(...)` only if you explicitly want the additive native RL batch entry point itself.

## FP32 Note

`FP32` remains evaluation-only.

- `FP64` and CPU reference remain the correctness anchors.
- On the current branch, the integrated wrapper intentionally does not use the fused RL fast path for `FP32`; it falls back to the older Python RL postprocess path.
