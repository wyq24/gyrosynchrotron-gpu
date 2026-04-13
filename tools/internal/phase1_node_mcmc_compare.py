#!/usr/bin/env python3
"""Internal comparison helper for single-node MCMC step-count studies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner


DEFAULT_DATA = None
DEFAULT_SCF_ROOT = None
DEFAULT_GS_ROOT = None
DEFAULT_LIB = None

DEFAULT_ALL_PARAM_BOUNDS = [
    (5.0, 40.0),
    (0.1, 8.0),
    (0.1, 30.0),
    (9.5, 12.0),
    (4.0, 9.0),
    (1.0, 12.0),
    (0.0, 90.0),
    (5.0, 50.0),
]
DEFAULT_VARY_INDICES = [1, 3, 4, 5, 6]
DEFAULT_FIXED_PARAMS = [20.0, 2.5, 15.0, 6.0, 7.0, 2.0, 80.0, 20.0]
DEFAULT_X_LOG_BOUNDS = (4.0, 9.0)
PARAM_NAMES = [
    "area_asec2",
    "depth_asec",
    "Bx100G",
    "T_MK",
    "log_nth",
    "log_nnth",
    "delta",
    "theta_deg",
]


def _align_valid_mask(cube: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cube.shape[:2] == valid_mask.shape:
        return cube, valid_mask

    h, w = cube.shape[:2]
    hm, wm = valid_mask.shape
    y0_cube = 0
    x0_cube = 0
    y0_mask = 0
    x0_mask = 0
    y_size = min(h, hm)
    x_size = min(w, wm)

    if h > hm:
        y0_cube = (h - hm) // 2
        y_size = hm
    elif hm > h:
        y0_mask = (hm - h) // 2
        y_size = h

    if w > wm:
        x0_cube = (w - wm) // 2
        x_size = wm
    elif wm > w:
        x0_mask = (wm - w) // 2
        x_size = w

    cube = cube[y0_cube : y0_cube + y_size, x0_cube : x0_cube + x_size, :]
    valid_mask = valid_mask[y0_mask : y0_mask + y_size, x0_mask : x0_mask + x_size]
    return cube, valid_mask


def _maybe_json(value: str | None, default):
    return default if not value else json.loads(value)


def _downsample(samples: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if samples.shape[0] <= max_points:
        return samples
    rng = np.random.default_rng(seed)
    sel = rng.choice(samples.shape[0], size=max_points, replace=False)
    return samples[np.sort(sel)]


def _plot_corner(samples: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.39, 0.86),
    )
    fig.suptitle(title, y=1.02)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(
    samples_small: np.ndarray,
    samples_large: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    fig = corner.corner(
        samples_small,
        labels=labels,
        color="tab:blue",
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.39, 0.86),
    )
    corner.corner(
        samples_large,
        fig=fig,
        labels=labels,
        color="tab:orange",
        show_titles=False,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.39, 0.86),
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=2, label="500 steps"),
            Line2D([0], [0], color="tab:orange", lw=2, label="3000 steps"),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.suptitle(title, y=1.02)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _summary(samples: np.ndarray) -> dict[str, list[float]]:
    return {
        "q16": np.quantile(samples, 0.16, axis=0).tolist(),
        "median": np.median(samples, axis=0).tolist(),
        "q84": np.quantile(samples, 0.84, axis=0).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-node MCMC step-count comparison")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--scf-root", default=DEFAULT_SCF_ROOT)
    ap.add_argument("--gs-root", default=DEFAULT_GS_ROOT)
    ap.add_argument("--mcmc-lib", default=DEFAULT_LIB)
    ap.add_argument("--out", default="artifacts/scf_node_compare")
    ap.add_argument("--node-index", type=int, default=1449)
    ap.add_argument("--obs-key", default="tbs_spl_conv")
    ap.add_argument("--valid-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--walkers", type=int, default=512)
    ap.add_argument("--batch-capacity", type=int, default=512)
    ap.add_argument("--steps-small", type=int, default=500)
    ap.add_argument("--steps-large", type=int, default=3000)
    ap.add_argument("--burn-in-small", type=int, default=100)
    ap.add_argument("--burn-in-large", type=int, default=600)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--noise-level", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param-bounds", default=None)
    ap.add_argument("--vary-indices", default=None)
    ap.add_argument("--fixed-params", default=None)
    ap.add_argument("--x-log-bounds", default=None)
    args = ap.parse_args()

    if not args.data:
        raise ValueError("--data is required")
    if not args.scf_root:
        raise ValueError("--scf-root is required")

    repo_root = Path(__file__).resolve().parents[2]
    scf_root = Path(args.scf_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(scf_root) not in sys.path:
        sys.path.insert(0, str(scf_root))

    from spatial_coherence_fitter.data_io import load_npz_dataset
    from spatial_coherence_fitter.mcmc_backend import (
        _build_forward_backend,
        _load_gpu_backend_module,
        _resolve_gs_root,
        _resolve_lib_path,
    )

    all_param_bounds = _maybe_json(args.param_bounds, DEFAULT_ALL_PARAM_BOUNDS)
    vary_indices = _maybe_json(args.vary_indices, DEFAULT_VARY_INDICES)
    fixed_params = _maybe_json(args.fixed_params, DEFAULT_FIXED_PARAMS)
    x_log_bounds = tuple(_maybe_json(args.x_log_bounds, list(DEFAULT_X_LOG_BOUNDS)))
    labels = [PARAM_NAMES[idx] for idx in vary_indices]

    dataset = load_npz_dataset(
        args.data,
        obs_key_candidates=(args.obs_key,),
        true_key_candidates=("Tb_true", "cube_true", "true"),
        theta_key_candidates=("theta_true", "theta", "params"),
    )
    cube = np.asarray(dataset.cube_obs, dtype=np.float64)

    valid_mask = None
    if args.valid_only:
        with np.load(args.data, allow_pickle=True) as data:
            valid_mask = np.isfinite(np.asarray(data["log_nnth_fit"]))
        cube, valid_mask = _align_valid_mask(cube, valid_mask)

    h, w, _ = cube.shape
    if args.node_index < 0 or args.node_index >= h * w:
        raise ValueError(f"node_index {args.node_index} is out of range for {h}x{w}")
    y, x = divmod(args.node_index, w)
    if valid_mask is not None and not bool(valid_mask[y, x]):
        raise ValueError(f"node_index {args.node_index} at (y={y}, x={x}) is not valid")

    spectrum = np.asarray(cube[y, x, :], dtype=np.float64)
    if not np.all(np.isfinite(spectrum)):
        raise ValueError(f"node_index {args.node_index} contains non-finite spectrum bins")

    gs_root = _resolve_gs_root(args.gs_root or str(repo_root))
    lib_path = _resolve_lib_path(args.mcmc_lib or str(repo_root / "source" / "MWTransferArr.so"), gs_root)
    gpu_mod = _load_gpu_backend_module(gs_root)
    backend, selected_device = _build_forward_backend(
        gpu_mod=gpu_mod,
        lib_path=lib_path,
        batch_capacity=args.batch_capacity,
        device=args.device,
        precision=args.precision,
    )

    vary_bounds = [all_param_bounds[i] for i in vary_indices]

    def run_once(n_steps: int, burn_in: int) -> np.ndarray:
        cfg = gpu_mod.SamplingConfig(
            n_walkers=args.walkers,
            n_steps=n_steps,
            burn_in=burn_in,
            thin=args.thin,
            noise_level=args.noise_level,
        )
        return np.asarray(
            gpu_mod.run_single_mcmc_gpu_batched(
                spectrum=spectrum,
                vary_bounds=vary_bounds,
                vary_indices=vary_indices,
                fixed_params=fixed_params,
                x_log_bounds=x_log_bounds,
                sampling_cfg=cfg,
                forward_backend=backend,
                seed=args.seed,
            ),
            dtype=np.float64,
        )

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    small = run_once(args.steps_small, args.burn_in_small)
    large = run_once(args.steps_large, args.burn_in_large)

    np.save(out_dir / f"node_{args.node_index:06d}_steps_{args.steps_small}.npy", small)
    np.save(out_dir / f"node_{args.node_index:06d}_steps_{args.steps_large}.npy", large)

    small_plot = _downsample(small, max_points=15000, seed=0)
    large_plot = _downsample(large, max_points=15000, seed=1)

    _plot_corner(
        small_plot,
        labels=labels,
        title=f"Node {args.node_index} corner, {args.steps_small} steps",
        out_path=out_dir / f"node_{args.node_index:06d}_corner_{args.steps_small}.png",
    )
    _plot_corner(
        large_plot,
        labels=labels,
        title=f"Node {args.node_index} corner, {args.steps_large} steps",
        out_path=out_dir / f"node_{args.node_index:06d}_corner_{args.steps_large}.png",
    )
    _plot_overlay(
        small_plot,
        large_plot,
        labels=labels,
        title=f"Node {args.node_index}: 500 vs 3000 steps",
        out_path=out_dir / f"node_{args.node_index:06d}_corner_overlay.png",
    )

    summary = {
        "node_index": int(args.node_index),
        "pixel_yx": [int(y), int(x)],
        "obs_key": args.obs_key,
        "device": str(selected_device),
        "precision": str(args.precision),
        "lib_path": str(lib_path),
        "walkers": int(args.walkers),
        "batch_capacity": int(args.batch_capacity),
        "thin": int(args.thin),
        "noise_level": float(args.noise_level),
        "vary_indices": list(map(int, vary_indices)),
        "labels": labels,
        "small": {
            "steps": int(args.steps_small),
            "burn_in": int(args.burn_in_small),
            "sample_count": int(small.shape[0]),
            "summary": _summary(small),
        },
        "large": {
            "steps": int(args.steps_large),
            "burn_in": int(args.burn_in_large),
            "sample_count": int(large.shape[0]),
            "summary": _summary(large),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")

    print(f"node_index={args.node_index} pixel=({y},{x})")
    print(f"device={selected_device} precision={args.precision}")
    print(f"lib_path={lib_path}")
    print(f"small_samples={small.shape[0]} large_samples={large.shape[0]}")
    print(out_dir / f"node_{args.node_index:06d}_corner_{args.steps_small}.png")
    print(out_dir / f"node_{args.node_index:06d}_corner_{args.steps_large}.png")
    print(out_dir / f"node_{args.node_index:06d}_corner_overlay.png")
    print(out_dir / "summary.json")


if __name__ == "__main__":
    main()
