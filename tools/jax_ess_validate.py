import argparse
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = REPO_ROOT / "examples"

for path in (REPO_ROOT, TOOLS_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from jax_ffi.cli_reentry import maybe_reexec_tool_main

maybe_reexec_tool_main(
    "jax_ess_validate",
    script_path=__file__,
    repo_root=REPO_ROOT,
    extra_paths=(TOOLS_DIR, EXAMPLES_DIR),
)

from tools import jax_ess_trial


def build_arg_parser() -> argparse.ArgumentParser:
    parser = jax_ess_trial.build_arg_parser()
    parser.description = "Run a multi-seed NumPyro ESS validation against the staged JAX FFI forward path."
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of consecutive seeds starting at --seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list. Overrides --seed and --num-seeds when provided.",
    )
    parser.add_argument(
        "--allowed-median-failures",
        type=int,
        default=1,
        help="Maximum number of seeds allowed to exceed --posterior-span-tol on the median-drift gate.",
    )
    return parser


def _summarize_param_stack(name: str, stack: np.ndarray) -> tuple[str, float]:
    worst_idx = int(np.argmax(np.max(stack, axis=0)))
    return jax_ess_trial.PARAM_NAMES_8D[worst_idx], float(np.max(stack[:, worst_idx]))


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(seed) for seed in args.seeds]
    else:
        seeds = [int(args.seed) + offset for offset in range(int(args.num_seeds))]
    if not seeds:
        raise ValueError("At least one seed is required.")

    failures = []
    results = []
    median_stack = []
    q16_stack = []
    q84_stack = []

    print("JAX ESS multi-seed validation")
    print(f"seeds: {seeds}")
    print(f"posterior_span_tol: {args.posterior_span_tol:.3e}")
    print(f"allowed_median_failures: {args.allowed_median_failures}")

    shared = vars(args).copy()
    shared.pop("num_seeds", None)
    shared.pop("seeds", None)
    shared.pop("allowed_median_failures", None)
    shared.pop("posterior_span_tol", None)

    for seed in seeds:
        result = jax_ess_trial.run_trial(**{**shared, "seed": seed})
        results.append(result)
        median_stack.append(np.asarray(result["median_span_fraction"], dtype=np.float64))
        q16_stack.append(np.asarray(result["q16_span_fraction"], dtype=np.float64))
        q84_stack.append(np.asarray(result["q84_span_fraction"], dtype=np.float64))

        status = "PASS"
        if result["max_median_span_fraction"] > args.posterior_span_tol:
            failures.append(seed)
            status = "FAIL"

        print(
            f"seed={seed} status={status} "
            f"median={result['max_median_span_fraction']:.3e} "
            f"q16={result['max_q16_span_fraction']:.3e} "
            f"q84={result['max_q84_span_fraction']:.3e} "
            f"worst_median={result['worst_median_param']} "
            f"worst_q84={result['worst_q84_param']}"
        )

    median_stack_arr = np.stack(median_stack, axis=0)
    q16_stack_arr = np.stack(q16_stack, axis=0)
    q84_stack_arr = np.stack(q84_stack, axis=0)

    worst_median_param, worst_median_value = _summarize_param_stack("median", median_stack_arr)
    worst_q16_param, worst_q16_value = _summarize_param_stack("q16", q16_stack_arr)
    worst_q84_param, worst_q84_value = _summarize_param_stack("q84", q84_stack_arr)

    median_per_seed = np.asarray([row["max_median_span_fraction"] for row in results], dtype=np.float64)
    q16_per_seed = np.asarray([row["max_q16_span_fraction"] for row in results], dtype=np.float64)
    q84_per_seed = np.asarray([row["max_q84_span_fraction"] for row in results], dtype=np.float64)

    print("Aggregate summary")
    print(f"median_failures: {len(failures)}/{len(seeds)}")
    print(f"median_fail_seed_list: {failures}")
    print(f"median_per_seed_mean: {float(np.mean(median_per_seed)):.3e}")
    print(f"median_per_seed_max: {float(np.max(median_per_seed)):.3e}")
    print(f"q16_per_seed_mean: {float(np.mean(q16_per_seed)):.3e}")
    print(f"q16_per_seed_max: {float(np.max(q16_per_seed)):.3e}")
    print(f"q84_per_seed_mean: {float(np.mean(q84_per_seed)):.3e}")
    print(f"q84_per_seed_max: {float(np.max(q84_per_seed)):.3e}")
    print(f"worst_param_over_seeds_median: {worst_median_param} {worst_median_value:.3e}")
    print(f"worst_param_over_seeds_q16: {worst_q16_param} {worst_q16_value:.3e}")
    print(f"worst_param_over_seeds_q84: {worst_q84_param} {worst_q84_value:.3e}")

    if len(failures) > args.allowed_median_failures:
        raise RuntimeError(
            f"Median-drift failures exceeded allowance: {len(failures)} > {args.allowed_median_failures}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
