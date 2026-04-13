# opt/fp32-salvage

## Status

Paused diagnostic branch.

## What This Branch Established

- catastrophic FP32 failure classes were removed on the supported narrow path:
  - invalid-wave sentinel conversion issue
  - one dangerous normalization expression for `norm_a`
- CUDA `FP64` and the CPU reference remain the correctness anchors
- CUDA `FP32` remains faster than CUDA `FP64`, but it is still not acceptable as an optional mode

## Remaining Error Source

The remaining FP32 error is not primarily in Python wrapper postprocessing.

It localizes to the corrected-root path in the CUDA kernel:

- `FindMu0`
- `H1`
- corrected `mu0`
- `lnq2`

Those quantities then perturb the local emissivity path on the supported workload.

## Stop Point

- a local selective mixed-precision attempt in the corrected-root path regressed badly and was reverted
- no retained source change was accepted after that attempt
- any further FP32 effort would require broader mixed precision in that nonlinear corrected-root logic
- broader mixed precision is not the current priority

## Scope Reminder

- Python only
- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched workflow first
- no exact GS, array DF, transport, IDL, or broader API redesign
