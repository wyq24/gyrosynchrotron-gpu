import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import GScodes
import phase1_workloads


PROFILE_CATEGORIES = {
    "approximate_gs": (
        "GSIntegrandApprox",
        "GS_jk_approx",
        "trapzdLog",
        "BrentRoot",
        "SecantRoot",
        "IntegrableFunctionLog",
    ),
    "exact_gs": (
        "FindBesselJ",
        "qromb(",
        "GS_jk(",
    ),
    "analytical_df": (
        "Std_DF::FE",
        "PLWdf::FE",
        "ISOdf::g1short",
    ),
    "array_df": (
        "Arr_DF::FE",
        "Arr_DF::Fp",
        "Spline2D::Interpolate",
        "LQInterpolate2D",
    ),
    "radiation_transfer": (
        "RadiationTransfer",
    ),
    "free_free": (
        "FF_jk",
    ),
}


def _category_for_symbol(symbol):
    for category, prefixes in PROFILE_CATEGORIES.items():
        if any(prefix in symbol for prefix in prefixes):
            return category
    return None


def _parse_counts(sample_text):
    counts = {}
    in_collapsed = False
    for raw_line in sample_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("Sort by top of stack"):
            in_collapsed = True
            continue
        if in_collapsed and not line.strip():
            break
        if not in_collapsed:
            continue
        match = re.match(r"\s+(.+?)\s+\(in MWTransferArr\.so\)\s+(\d+)\s*$", line)
        if not match:
            continue
        symbol = match.group(1).strip()
        count = int(match.group(2))
        counts[symbol] = counts.get(symbol, 0) + count
    return counts


def summarize_sample(sample_path):
    text = Path(sample_path).read_text()
    counts = _parse_counts(text)
    category_counts = {key: 0 for key in PROFILE_CATEGORIES}
    unmatched = 0
    total = 0

    for symbol, count in counts.items():
        total += count
        category = _category_for_symbol(symbol)
        if category is None:
            unmatched += count
        else:
            category_counts[category] += count

    shares = {}
    if total > 0:
        for category, count in category_counts.items():
            shares[category] = count / total
        shares["unmatched"] = unmatched / total
    else:
        for category in category_counts:
            shares[category] = 0.0
        shares["unmatched"] = 0.0

    return {
        "symbol_counts": counts,
        "category_counts": category_counts,
        "total_collapsed_samples": total,
        "category_shares": shares,
    }


def run_real_wrapper_loop(libname, work_seconds):
    batch = phase1_workloads.build_real_wrapper_workload(batch_size=1)
    mwfunc = GScodes.initGET_MW(libname)
    lparms, rparms, parms, dummy, out = GScodes._build_powerlaw_iso_single_inputs(batch, 0)
    return _run_loop(mwfunc, work_seconds, lparms, rparms, parms, dummy, dummy, dummy, out)


def run_array_baseline_loop(libname, work_seconds):
    baseline = phase1_workloads.build_repo_array_baseline()
    mwfunc = GScodes.initGET_MW(libname)
    return _run_loop(
        mwfunc,
        work_seconds,
        baseline["lparms"],
        baseline["rparms"],
        baseline["parms"],
        baseline["e_arr"],
        baseline["mu_arr"],
        baseline["f_arr"],
        baseline["rl"],
    )


def _run_loop(mwfunc, work_seconds, *call_args):
    print(os.getpid(), flush=True)
    deadline = time.time() + work_seconds
    count = 0
    status = 0
    while time.time() < deadline:
        status = int(mwfunc(*call_args))
        count += 1
    print(f"DONE {count} {status}", flush=True)


def spawn_and_sample(scenario, libname, sample_seconds, work_seconds):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--scenario",
        scenario,
        "--lib",
        libname,
        "--work-seconds",
        str(work_seconds),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        pid_line = proc.stdout.readline().strip()
        if not pid_line:
            stderr = proc.stderr.read()
            raise RuntimeError(f"Worker failed to start: {stderr}")
        pid = int(pid_line)
        with tempfile.NamedTemporaryFile(prefix=f"phase1_{scenario}_", suffix=".txt", delete=False) as handle:
            sample_path = handle.name
        subprocess.run(
            ["sample", str(pid), str(sample_seconds), "1", "-mayDie", "-file", sample_path],
            check=True,
        )
        stdout, stderr = proc.communicate()
    finally:
        if proc.poll() is None:
            proc.kill()

    done_match = re.search(r"DONE\s+(\d+)\s+(-?\d+)", stdout)
    if not done_match:
        raise RuntimeError(f"Worker did not report completion.\nstdout={stdout}\nstderr={stderr}")
    return {
        "sample_path": sample_path,
        "iterations": int(done_match.group(1)),
        "status": int(done_match.group(2)),
    }


def profile_scenario(scenario, libname, sample_seconds, work_seconds):
    libname = phase1_workloads.default_library_path(libname)
    sampled = spawn_and_sample(scenario, libname, sample_seconds, work_seconds)
    summary = summarize_sample(sampled["sample_path"])

    if scenario == "real_wrapper":
        workload = phase1_workloads.build_real_wrapper_workload(batch_size=1)
        descriptor = {
            "n_pix": 1,
            "n_z": 1,
            "n_nu": workload.nfreq,
            "fraction_above_nu_cr": float(workload.fraction_above_nu_cr()[0]),
            "fraction_above_nu_wh": float(workload.fraction_above_nu_wh()[0]),
        }
    else:
        baseline = phase1_workloads.build_repo_array_baseline()
        descriptor = {
            "n_pix": 1,
            "n_z": baseline["nsteps"],
            "n_nu": baseline["nfreq"],
            "n_e": baseline["ne"],
            "n_mu": baseline["nmu"],
            "fraction_above_nu_cr": baseline["above_nu_cr_fraction"],
            "fraction_above_nu_wh": baseline["above_nu_cr_fraction"],
        }

    descriptor["iterations"] = sampled["iterations"]
    descriptor["calls_per_second"] = sampled["iterations"] / work_seconds
    descriptor["status"] = sampled["status"]
    descriptor["sample_path"] = sampled["sample_path"]
    descriptor["collapsed_samples"] = summary["total_collapsed_samples"]
    descriptor["category_counts"] = summary["category_counts"]
    descriptor["category_shares"] = summary["category_shares"]
    return descriptor


def main():
    parser = argparse.ArgumentParser(description="Collect sample-based runtime shares for the Phase 1 workloads.")
    subparsers = parser.add_subparsers(dest="mode", required=False)

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--scenario", choices=["real_wrapper", "array_baseline"], required=True)
    worker.add_argument("--lib", required=True)
    worker.add_argument("--work-seconds", type=float, required=True)

    parser.add_argument("--lib", help="Path to the MWTransferArr shared library.")
    parser.add_argument(
        "--scenario",
        choices=["real_wrapper", "array_baseline", "all"],
        default="all",
        help="Scenario to profile.",
    )
    parser.add_argument("--sample-seconds", type=int, default=10, help="Duration to pass to macOS sample.")
    parser.add_argument("--work-seconds", type=int, default=20, help="How long the worker should run.")
    args = parser.parse_args()

    if args.mode == "_worker":
        if args.scenario == "real_wrapper":
            run_real_wrapper_loop(args.lib, args.work_seconds)
        else:
            run_array_baseline_loop(args.lib, args.work_seconds)
        return

    scenarios = ["real_wrapper", "array_baseline"] if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        result = profile_scenario(scenario, args.lib, args.sample_seconds, args.work_seconds)
        print(f"Scenario: {scenario}")
        print(
            f"  shape: Npix={result['n_pix']} Nz={result['n_z']} Nnu={result['n_nu']}"
            + (f" NE={result['n_e']} Nmu={result['n_mu']}" if "n_e" in result else "")
        )
        print(
            f"  fraction_above_nu_cr={result['fraction_above_nu_cr']:.3f} "
            f"fraction_above_nu_wh={result['fraction_above_nu_wh']:.3f}"
        )
        print(
            f"  iterations={result['iterations']} calls_per_second={result['calls_per_second']:.2f} "
            f"collapsed_samples={result['collapsed_samples']}"
        )
        for category in [
            "approximate_gs",
            "exact_gs",
            "analytical_df",
            "array_df",
            "radiation_transfer",
            "free_free",
            "unmatched",
        ]:
            share = result["category_shares"].get(category, 0.0)
            count = result["category_counts"].get(category, 0)
            print(f"  {category}: share={share:.3f} count={count}")
        print(f"  sample_file={result['sample_path']}")


if __name__ == "__main__":
    main()
