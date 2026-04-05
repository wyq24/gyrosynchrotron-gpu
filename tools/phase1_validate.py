import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import GScodes
import phase1_workloads
import runtime_env


def validate_batch_api(libname, batch_size):
    libname = phase1_workloads.default_library_path(libname)
    batch = phase1_workloads.build_real_wrapper_sweep(batch_size)
    batch_result = GScodes.run_powerlaw_iso_batch_cpu(libname, batch)

    single_call = GScodes.initGET_MW(libname)
    max_abs = 0.0
    max_rel = 0.0
    for idx in range(batch.batch_size):
        lparms, rparms, parms, dummy, out = GScodes._build_powerlaw_iso_single_inputs(batch, idx)
        status = int(single_call(lparms, rparms, parms, dummy, dummy, dummy, out))
        if status != int(batch_result.status[idx]):
            raise AssertionError(f"status mismatch at batch index {idx}: {status} != {batch_result.status[idx]}")

        candidate = batch_result.rl[:, :, idx]
        ref = out
        abs_err = np.max(np.abs(candidate - ref))
        denom = np.maximum(np.abs(ref), 1.0e-300)
        rel_err = np.max(np.abs(candidate - ref) / denom)
        max_abs = max(max_abs, float(abs_err))
        max_rel = max(max_rel, float(rel_err))

        if not np.array_equal(candidate, ref):
            raise AssertionError(f"RL mismatch at batch index {idx}: max_abs={abs_err:.3e}, max_rel={rel_err:.3e}")

    return {
        "library": libname,
        "batch_size": batch_size,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate the Phase 1 CPU batch API against repeated single pyGET_MW calls.")
    parser.add_argument("--lib", help="Path to the MWTransferArr shared library.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of spectra in the validation batch.")
    args = parser.parse_args()

    runtime_env.print_runtime_summary(args.lib)
    result = validate_batch_api(args.lib, args.batch_size)
    print("Phase 1 batch API validation passed")
    print(f"library: {result['library']}")
    print(f"batch_size: {result['batch_size']}")
    print(f"max_abs_error: {result['max_abs_error']:.3e}")
    print(f"max_rel_error: {result['max_rel_error']:.3e}")


if __name__ == "__main__":
    main()
