# JAX-Vectorized Forward Model Feasibility Assessment

**Branch:** `research/jax-vectorized-forward`  
**Date:** 2026-04-05  
**Target scope:** narrow validated path only — analytical PLW + ISO, Nz=1, approximate GS, FP64, batch interface

---

## Evidence base

Files read during this assessment:

| File | What it established |
|------|---------------------|
| `examples/GScodes.py` | Full ctypes signature, memory layout rules, batch dataclass, fast-path conditions |
| `numpyro_ess_migration_skeleton.py` | The frozen JAX interface contract; explicit statement of two acceptable paths |
| `mcmc_backend_gpu_batched.py` | Current sampler architecture: emcee + `vectorize=True` over batched GPU backend |
| `spec_utils.py` | Unit conversions, parameter ordering, single-voxel RL reduction |
| `npec_helpers.py` | Normalizer and SBI model wrapper; observation processing pipeline |
| `doc/CallingEfficiently.md` | Fast-path activation conditions; exact ctypes call sequence |
| `doc/Phase1GPU.md` | Validation scope and correctness baseline |
| `doc/OptFusedNativeRL.md` | Speedup numbers; fused RL path; Python RL postprocessing as prior bottleneck |
| `doc/OptFP32Salvage.md` | FP32 status: paused, evaluation-only, not production-correct |
| `doc/BuildLibraries.md` | Shared library exports; build system |
| `source/` (directory listing) | C++/CUDA source structure; physics modules |

---

## Shared-library boundary: actual facts

The validated batch interface exposes the following call shape at the ctypes level:

```
pyMW_Approx_Batch_RL(
    backend_code, precision_code,      # int scalars
    batch_size, nfreq, npoints, q_on,  # int scalars
    nu0_hz, dlog10_nu,                 # float64 scalars
    area_cm2[B], depth_cm[B],          # C-contiguous float64
    bmag_g[B], temperature_k[B],
    thermal_density[B], nonthermal_density[B],
    delta[B], theta_deg[B],
    emin_mev[B], emax_mev[B],
    status[B: int32],
    native_freq_hz[F],                 # output, C-contiguous float64
    rl[7, F, B],                       # output, F-contiguous float64
) -> int
```

Key boundary properties:
- All floating-point data is **float64** on both sides
- Input physical arrays: **C-contiguous**, 1D, shape `[B]`
- Output RL array: **F-contiguous**, shape `[7, F, B]`
- There is no CUDA-side JAX device-pointer integration; memory is allocated in Python numpy, pinned by ctypes, and the library writes results into those buffers
- The call is **synchronous** and returns control to Python only after the CUDA kernel has finished
- The build system produces a traditional `dlopen`-able shared library, not an XLA plugin

---

## Path A: Python callback bridge (`jax.pure_callback`)

### What it actually does

```python
# Conceptual sketch — not production code
def _forward_numpy(params_batch_flat):
    # unpack, call ctypes library, return spectrum array
    ...

result_shape = jax.ShapeDtypeStruct((batch_size, nfreq), jnp.float64)
spectra = jax.pure_callback(_forward_numpy, result_shape, params_jax)
```

### Honest evaluation

`jax.pure_callback` (JAX ≥ 0.4.1) is **not a no-op interop trick**. It does allow JAX to trace and JIT a function that contains a Python host callback. However:

**What works:**
- The JIT compilation succeeds; XLA treats the callback as an opaque primitive
- The function can be called inside `jax.jit` and `lax.scan` without tracing errors
- The function can be batched via `vmap` if `vectorized=True` is set, meaning JAX calls it once with the full batch instead of looping
- The `logdensity` function in `numpyro_ess_migration_skeleton.py` would JIT-compile without error
- This path works for **gradient-free** samplers: NumPyro ESS, emcee-via-JAX wrappers, Markov kernels that don't differentiate

**What does not work:**
- **Autodiff is impossible.** `pure_callback` has no reverse-mode rule. Calling `jax.grad(logdensity)` raises an error. This rules out NUTS, HMC, and any second-order gradient-based sampler.
- **Host-device round-trips occur every call.** A JAX GPU array must be transferred to host memory before the numpy ctypes call, and results must be transferred back. At batch=512 with 100 frequency points, this is ~1.6 MB each way. From `OptFusedNativeRL.md`, host transfers are 0.001s total at batch=512 — not the dominant cost — but they occur *every* log-density evaluation, which at thousands of MCMC steps accumulates.
- **The XLA execution stream stalls at the callback.** While the callback executes synchronously on the host, the GPU compute pipeline is idle unless the ctypes library internally launches CUDA work (which it does, but the synchronization point is at the return boundary).
- **No XLA fusion.** The forward model cannot be fused with downstream JAX operations (normalization, likelihood computation) because XLA cannot see inside the callback.

**Verdict on Path A:**  
Path A is an **interoperability bridge, not a vectorized JAX path**. It is useful in exactly one scenario: you want to use NumPyro's ESS sampler (gradient-free), you want to write the log-density in JAX for composability with the normalizer, and you accept the host-dispatch overhead per call. It does not unlock JIT fusion, autodiff, or vmap over parameter dimensions. It does not replace the emcee batched path for throughput. It is a valid **interim integration layer** for ESS-style sampling but not the migration target described in `numpyro_ess_migration_skeleton.py`.

---

## Path B: JAX FFI / custom-call wrapper around existing CUDA backend

### What this requires

JAX's current FFI mechanism (`jax.extend.ffi`, stabilized in JAX 0.4.28) allows C++/CUDA functions to be registered as XLA custom-call targets. Once registered:
- The kernel is callable as a native JAX operation
- JIT compilation works: XLA sees the call as a first-class graph node
- Shape inference is handled by metadata registered at module import time
- `vmap` can be layered on top (with manual batching rule or implicitly if batch dim is explicit)
- **Autodiff still requires a hand-written custom VJP rule** unless the physics is differentiable by construction; a numerical VJP via finite differences is always possible

**What would need to change at the boundary:**

1. **New build target.** The existing `MWTransferArr.so` is a traditional shared library loaded via `ctypes`. An XLA plugin requires a different compilation target: the CUDA kernel must be exposed through `XLA_FFI_CALL_HANDLER` or the older `Custom_call` registration API, compiled against JAX's `include/xla/ffi/api/c_api.h`. The physics C++/CUDA code itself (`ApproxBatch.cpp`, `ApproxBatchCuda.cu`) does **not** need to change. Only the boundary layer (`PythonInterface.cpp` and a new `XLAInterface.cpp`) changes.

2. **Static shape declaration.** XLA requires knowing `(batch_size, nfreq, 7)` at trace time. `batch_size` and `nfreq` must be compile-time constants or shape-polymorphic abstract values. The current API already takes these as scalar ints, so this is not a physics change — it is a registration-time shape constraint.

3. **Buffer semantics.** The current ctypes path pre-allocates numpy output buffers and passes raw pointers. The XLA FFI path receives `XLA_FFI_Buffer*` descriptors from XLA's buffer pool; the kernel writes into those. The CUDA kernel call itself (`cuLaunchKernel` style or a CUDA Runtime call) remains identical — the difference is *who allocates the output buffer*.

4. **FP64 on CUDA.** XLA supports FP64 (`F64`) on CUDA. No change required here; the kernel already runs FP64.

5. **Memory layout reconciliation.** The current output is F-contiguous `[7, F, B]`. JAX arrays are by default C-contiguous. You can register the custom call with a layout annotation (`xla_client.Shape.array_shape(..., minor_to_major=[0,1,2])`) to tell XLA the kernel produces F-order output. Alternatively, a thin transpose is applied after the call. This is low-effort.

**Engineering effort estimate:**

| Task | Effort |
|------|--------|
| Write `XLAInterface.cpp` with registration boilerplate | 1–2 days |
| Add new build target to Makefile | 0.5 days |
| Write Python `jax_ffi_wrapper.py` that loads plugin and declares shapes | 1 day |
| Write a `jax.ShapeDtypeStruct` shape-inference rule | 0.5 days |
| Validate M0: bit-exact output against ctypes path, single batch | 1 day |
| Write custom VJP (optional, numerical finite-diff first) | 1 day |
| Total | ~5–7 focused days |

This is a moderate engineering investment, not a rewrite.

**Main blockers:**

1. **JAX FFI API stability.** The `jax.extend.ffi` API stabilized in JAX 0.4.28 (late 2024). Older JAX versions require the undocumented `_custom_call` path. Pin JAX version ≥ 0.4.28 explicitly.

2. **Two shared library artifacts.** The repo would have `MWTransferArr.so` (legacy ctypes path, currently validated) and a new `MWTransferArrXLA.so` (FFI plugin). This is normal for a phased migration; the ctypes path remains the validated reference until the FFI path is validated to numerical equivalence.

3. **No CUDA device-pointer sharing with JAX device.** JAX manages its own CUDA memory pool (via `jaxlib`'s allocator). The XLA FFI buffers are allocated from this pool. The CUDA kernel receives raw `void*` device pointers from XLA. This is the standard pattern for XLA custom calls and is well-supported.

4. **emcee is not JAX.** The current sampler (`mcmc_backend_gpu_batched.py`) uses emcee, which is a numpy-based Python library. Path B creates a JAX-native forward call that a JAX-based sampler (NumPyro ESS/NUTS, BlackJAX NUTS/MALA) can use. The emcee path continues to work via the ctypes library; Path B is an additional capability layer, not a replacement.

**Verdict on Path B:**  
Path B is **technically feasible** and preserves all validated physics code. It is the recommended path if the goal is a JAX-native forward model that can be JIT-compiled and composed with JAX-based samplers. It requires a moderate but well-scoped engineering effort, and the existing validated path is never touched.

---

## Path C: Pure JAX rewrite of the validated narrow path

### What this means

Reimplement the gyrosynchrotron emissivity computation for the analytical PLW+ISO, Nz=1 case entirely using `jax.numpy` operations, with `jax.config.update("jax_enable_x64", True)` for FP64.

**What the current C++/CUDA code provides as reference:**
- Physics specification (Ramaty/Dulk approximate GS formulae)
- Numerical implementation of Bessel function approximations
- Radiative transfer integration (Nz=1 means simple optical-depth calculation)
- The fused RL postprocessing (Rayleigh-Jeans brightness temperature, etc.)

**Assessment:**

The approximate GS formula for the narrow validated path (PLW+ISO, Nz=1) is well-specified in the literature (Ramaty 1969, Dulk & Marsh 1982, Fleishman & Kuznetsov 2010). A JAX implementation would:
- Use `jax.scipy.special` for special functions where available
- Implement the Bessel function approximations directly (they are polynomial approximations already in the C++ source)
- Express the frequency integration and parameter mapping in `jnp` operations
- Be fully differentiable by construction via JAX's autodiff

**What is preserved from current work:**
- The FP64 ctypes path becomes the **validation oracle** for bit-level correctness checks
- The memory layout and parameter ordering established in `GScodes.py` directly inform the JAX function signature
- The unit conversion logic in `spec_utils.py` is directly portable
- The normalizer in `npec_helpers.py` is already pure numpy and trivially portable to JAX

**What is discarded:**
- The CUDA backend compute (it becomes irrelevant once JAX compiles to XLA/CUDA natively)
- The ctypes boundary engineering
- The F-contiguous memory layout requirement

**Honest assessment:**
This is a **physics reimplementation project**, not a software integration project. The approximate GS path is not algorithmically complex (it is a table-lookup + integration scheme), but it requires careful numerical matching to the existing reference, especially for edge cases in the Bessel approximations and the corrected-root path. Expect 2–4 weeks for a competent implementation with full validation against the ctypes oracle. The FP32 failures documented in `OptFP32Salvage.md` (FindMu0, H1, lnq2 paths) suggest numerical subtleties that would need to be understood before reimplementation.

**Verdict on Path C:**  
Path C is the **cleanest long-term outcome** (fully differentiable, XLA-compiled, no external dependencies), but it is a new forward-model project that happens to have an excellent validation oracle. It is not a JAX migration — it is a JAX reimplementation. The existing GPU work contributes significantly as a correctness reference, but the compute investment (CUDA kernels) is not reused.

---

## Path D: Conclusion that a real JAX path is not currently worth it

This conclusion is **not warranted** given the current repo state. Here is why:

1. The `numpyro_ess_migration_skeleton.py` already exists with a frozen interface contract, indicating planned migration
2. The batch API already has the right semantics for JAX: `[B, P]` → `[B, F]` with fixed `F` and `P`
3. The emcee sampler is efficient for the current task but cannot use gradient-based samplers (NUTS, HMC) without differentiating through the forward model
4. Path B is moderate effort and preserves all existing work

However, Path D **is** the right conclusion for the current near-term operational priority: the existing emcee batched GPU path is already achieving 16–20x speedups over the legacy wrapper, and the inference pipeline is not yet blocked on sampler efficiency. The JAX migration adds value only when the sampler becomes the bottleneck or when NUTS sampling is needed.

---

## Recommended path

**Recommendation: Phased B → C**

1. **Now (M0–M1):** Pursue Path B (XLA FFI wrapper). This is moderate engineering effort, preserves all validated physics, and delivers a real JIT-able forward call for JAX-based samplers. It unblocks the `numpyro_ess_migration_skeleton.py` interface.

2. **Later (when sampler needs gradients):** Pursue Path C incrementally. The FFI wrapper validates the interface contract and provides a correctness oracle. The pure JAX reimplementation targets autodiff for NUTS/HMC.

3. **Interim (if needed quickly):** Path A (`pure_callback`) can be used for ESS-style sampling without blocking on FFI engineering. It is not the end state.

**Current GPU work is preserved in all cases.** The ctypes path remains the operational backend and the validation oracle throughout.

---

## Feasibility verdict

| Aspect | Assessment |
|--------|-----------|
| Recommended first path | **B (XLA FFI)** |
| Confidence | **High** — the batch interface already has the right shape semantics; the CUDA physics code does not need to change |
| Preserves existing GPU work | **Yes** — physics code unchanged; ctypes path remains validated reference |
| Engineering effort for M1 proof-of-concept | **~5–7 days** of focused boundary work |
| Main technical risk | JAX FFI API version pinning; XLA buffer ownership semantics |
| Blocker for NUTS/HMC | Requires custom VJP (numerical finite-diff viable as first pass) |
| FP32 impact | None — Path B targets FP64 only, consistent with current policy |

---

## Scope

**In scope for Phase 1 JAX FFI (M0–M2):**
- Analytical PLW + ISO, Nz=1
- Approximate GS only
- FP64 only
- Batch interface: `[B, 10 params]` → `[B, F]` (F = nfreq = 100, fixed)
- The fused RL path (`pyMW_Approx_Batch_RL`) is the FFI target (not the local j/k path)
- CUDA backend with CPU fallback

**Out of scope:**
- Exact GS
- Array DF
- Full transport
- FP32
- Nz > 1
- IDL interface
- Any physics broadening beyond the current narrow path

---

## Proposed JAX-facing interface

```python
# Proposed signature for the FFI-wrapped forward model
# (placed in a new file, e.g., jax_ffi/mw_approx_batch_jax.py)

def mw_approx_batch_rl_jax(
    params: jax.Array,      # [B, 10] float64, physical units (see parameter order below)
    nfreq: int = 100,
    nu0_hz: float = 0.8e9,
    dlog10_nu: float = 0.02,
    npoints: int = 16,
    q_on: bool = True,
    d_sun_au: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """
    Returns:
        rl:   [B, 7, F] float64, Stokes + derived quantities
        freq: [F] float64 Hz, frequency grid
    """
    ...

# Parameter order in params[:, i]:
# 0: area_cm2
# 1: depth_cm
# 2: bmag_g
# 3: temperature_k
# 4: thermal_density_cm3
# 5: nonthermal_density_cm3
# 6: delta
# 7: theta_deg
# 8: emin_mev
# 9: emax_mev
```

Notes:
- The output layout is transposed to `[B, 7, F]` (C-contiguous in batch dimension) from the existing F-contiguous `[7, F, B]` to match JAX's default C-contiguous semantics. The transposition is absorbed into the XLA layout descriptor at registration time.
- `nfreq`, `npoints`, `q_on` are compile-time-static for a given JIT compilation.
- FP64 requires `jax.config.update("jax_enable_x64", True)` in the calling code.

---

## Engineering milestones

### M0: Boundary audit and memory-layout audit
**Deliverable:** Written record of exact buffer sizes, alignment requirements, and F-contiguous output indexing in the current ctypes path. Confirm XLA FFI buffer API can satisfy the same layout contract.  
**Acceptance:** Can explain exactly what `PythonInterface.cpp` does at the pointer level; no surprises in M1.

### M1: Minimal XLA FFI proof-of-concept for one batch
**Deliverable:** `source/XLAInterface.cpp` + updated `Makefile` producing `MWTransferArrXLA.so`. Python loader `jax_ffi/mw_approx_batch_jax.py` wrapping the plugin. Single-batch test showing bit-exact output against the ctypes reference path at `batch_size=8, nfreq=100`.  
**Acceptance:** `allclose(result_ffi, result_ctypes, rtol=1e-12)` passes.

### M2: Vectorized batch proof and correctness validation
**Deliverable:** M1 extended to `batch_size=512`. Correctness validation at multiple batch sizes. `jax.jit` compilation verified. Optional: `vmap` over the batch dimension verified. Timing comparison against ctypes path.  
**Acceptance:** Correctness: same tolerance as M1. Throughput: no worse than 2x the ctypes path for `batch_size=512`.

### M3: Sampler integration trial with NumPyro ESS
**Deliverable:** Wire the FFI wrapper into `numpyro_ess_migration_skeleton.py` by implementing `TODOJAXForwardModel.simulate_batch_jax`. Run a toy single-pixel ESS trial. Compare posterior to the emcee reference posterior.  
**Acceptance:** ESS posterior is statistically consistent with emcee posterior on a test case.

### M4: Performance evaluation and go/no-go for NUTS
**Deliverable:** Measure ESS/second for the JAX path vs emcee batched GPU path. Assess whether NUTS is worth the custom VJP investment.  
**Acceptance:** Go/no-go decision with quantitative justification.

---

## Risks and blockers

| Risk | Severity | Notes |
|------|----------|-------|
| JAX FFI API instability | Medium | Pin JAX ≥ 0.4.28; test against 0.4.30+ |
| XLA buffer ownership conflict with CUDA allocator | Medium | Standard pattern; mitigated by using XLA-provided device pointers |
| F-contiguous ↔ C-contiguous layout mismatch | Low | Handled via XLA layout annotation or cheap transpose |
| FP64 on CUDA (XLA side) | Low | XLA F64 on CUDA is supported; already validated in ctypes path |
| Batch size must be static at JIT time | Medium | Requires recompilation for different batch sizes; use `jit` cache keyed on shape |
| No autodiff without custom VJP | High (for NUTS) | Numerical finite-diff VJP is viable first pass; analytical VJP is a separate project |
| Two build artifacts to maintain | Low | `MWTransferArr.so` (ctypes, validated) + `MWTransferArrXLA.so` (FFI) coexist peacefully |
| emcee path disrupted | None | emcee path is independent; no changes required |

---

## What should be done next

**Immediate (before starting M0):**  
- Confirm JAX version requirements with the compute environment  
- Check whether `jax.extend.ffi` is available or whether the older `jax.lib.xla_client.register_custom_call_target` path is needed  
- Read `source/PythonInterface.cpp` and `source/ApproxBatch.h` to understand the exact pointer conventions used in the CUDA kernel dispatch

**Short term (M0–M1):**  
- Write the XLA interface boundary as described above  
- Keep the ctypes path entirely untouched  
- Validate bit-exact output before proceeding

**Do not start:**  
- A broad rewrite of the physics  
- Any changes to `MWTransferArr.so`, `GScodes.py`, or `mcmc_backend_gpu_batched.py`  
- FP32 investigation (remains paused per `OptFP32Salvage.md`)  
- Physics scope expansion

---

## Summary

| Question | Answer |
|----------|--------|
| Is the current GPU forward work preserved? | **Yes** — the ctypes path is untouched in all viable paths |
| Is a real JAX-vectorized future technically worthwhile? | **Yes** — Path B (XLA FFI) is feasible with ~5–7 days of focused boundary work |
| What is the most realistic JAX path? | **Path B first, Path C later when autodiff is needed** |
| Does Path A count as a real JAX path? | **No** — it is an interop bridge; useful interim, not the migration target |
| What should be done next? | Read `PythonInterface.cpp`, confirm JAX version, begin M0 boundary audit |
