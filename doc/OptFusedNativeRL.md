# opt/fused-native-rl

## Goal

Reduce the current wrapper/backend gap on the validated narrow path by moving the current Python-side `RL`/output postprocessing into an additive native-side fast batch path.

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

## Success Criteria

- CPU reference and CUDA `FP64` remain the correctness anchors
- unsupported cases still fail loudly
- the current validated baseline on `main` and `baseline/fp64-validated` stays unchanged
- the new fast path reduces the wrapper/backend gap on realistic batch sizes
