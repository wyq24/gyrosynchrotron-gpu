# opt/fused-native-rl

## Goal

Reduce the current wrapper/backend gap on the validated narrow path by moving the current Python-side `RL`/output postprocessing into an additive native-side fast batch path.

## Current Evidence

Measured on the validated CUDA `FP64` real-sweep workflow:

- batch `128`
  - backend total `0.020549s`
  - wrapper total `0.026422s`
  - wrapper/backend gap `0.005873s`
  - Python postprocess `0.005842s`
  - call boundary `0.000075s`
  - host-to-device `0.000570s`
  - device-to-host `0.000261s`
- batch `512`
  - backend total `0.054187s`
  - wrapper total `0.075154s`
  - wrapper/backend gap `0.020967s`
  - Python postprocess `0.021831s`
  - call boundary `0.000084s`
  - host-to-device `0.000613s`
  - device-to-host `0.000494s`

Interpretation:

- the biggest end-to-end gap is the current Python-side `local_jk_to_single_voxel_rl` work
- the Python/native call boundary is not the main bottleneck
- transfer costs are not the main bottleneck on the supported path
- the highest-ROI additive optimization is native-side `RL` postprocessing for the already supported narrow batch path

## Scope

- Python only
- analytical `PLW + ISO`
- `Nz=1`
- approximate GS only
- batched workflow first
- keep the legacy-compatible Python wrapper intact
- do not broaden to exact GS, array DF, transport, IDL, or broader API redesign

## First Implementation Milestone

1. mirror the existing `local_jk_to_single_voxel_rl` math natively for the currently supported path only
2. expose it through an additive native/Python batch entry point without changing the existing wrapper contract
3. validate native `RL` outputs against the current Python postprocessing on the same packed inputs
4. benchmark end-to-end wrapper vs fused-native `RL` batch output on the real-sweep workload

Narrowest native-side work with the biggest expected gain:

- keep the current local `jX/kX/jO/kO` kernel as-is
- add only the single-voxel `RL` conversion now done in Python:
  - optical-depth handling
  - source-function / `delta` handling
  - left/right mode mapping based on `theta`
  - existing frequency row emission
- do not fuse unsupported physics or broader workflow logic

## Success Criteria

- CPU reference and CUDA `FP64` remain the correctness anchors
- unsupported cases still fail loudly
- the current validated baseline on `main` and `baseline/fp64-validated` stays unchanged
- the new fast path reduces the wrapper/backend gap on realistic batch sizes
