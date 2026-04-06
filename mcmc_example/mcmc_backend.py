"""MCMC backend for Phase 1 fitting with resumable execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp

import numpy as np

from npec_helpers import (
    ParameterNormalizer,
    simulator_8d,
)

from .assemble import node_theta_to_pixel_map
from .segmentation import Segmentation, make_block_segmentation, make_pixel_segmentation


@dataclass
class MCMCFitResult:
    theta_map: np.ndarray
    q16_map: np.ndarray
    q84_map: np.ndarray
    node_thetas: np.ndarray
    q16_nodes: np.ndarray
    q84_nodes: np.ndarray
    seg: Segmentation
    done_nodes: np.ndarray
    debug: dict


@dataclass
class _NormalizedLogProbEvaluator:
    """Picklable log-prob evaluator for process-based emcee execution."""

    vary_bounds: list[tuple[float, float]]
    vary_indices: list[int]
    fixed_params: list[float]
    x_log_bounds: tuple[float, float]
    observation_norm: np.ndarray
    noise_level: float

    def __post_init__(self) -> None:
        self.normalizer = ParameterNormalizer(
            self.vary_bounds,
            x_log_bounds=self.x_log_bounds,
            target_range=(-5, 5),
        )
        self.vary_indices_arr = np.asarray(self.vary_indices, dtype=np.int64)
        self.fixed_params_arr = np.asarray(self.fixed_params, dtype=np.float64)
        self.obs_norm_arr = np.asarray(self.observation_norm, dtype=np.float64).reshape(-1)

    def __call__(self, params_norm: np.ndarray) -> float:
        p = np.asarray(params_norm, dtype=np.float64).reshape(-1)
        if np.any(p < -5.0) or np.any(p > 5.0):
            return -np.inf

        params_denorm = np.asarray(self.normalizer.denormalize_params(p), dtype=np.float64).reshape(-1)
        full_params = np.array(self.fixed_params_arr, copy=True)
        full_params[self.vary_indices_arr] = params_denorm

        sim = simulator_8d(full_params)
        if hasattr(sim, "detach"):
            sim = sim.detach().cpu().numpy()
        sim = np.asarray(sim, dtype=np.float64).reshape(-1)
        sim_norm = np.asarray(self.normalizer.normalize_observation(sim), dtype=np.float64).reshape(-1)

        residuals = self.obs_norm_arr - sim_norm
        sigma2 = self.noise_level ** 2
        log_like = -0.5 * np.sum((residuals ** 2) / sigma2)
        log_like -= 0.5 * self.obs_norm_arr.size * np.log(2.0 * np.pi * sigma2)
        return float(log_like)


def _extract_node_spectra(
    cube: np.ndarray,
    seg: Segmentation,
    valid_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    h, w, _ = cube.shape
    spectra_list: list[np.ndarray] = []
    mask_flat = None
    if valid_mask is not None:
        if valid_mask.shape != (h, w):
            raise ValueError(f"valid_mask must be [H,W]={h,w}, got {valid_mask.shape}")
        mask_flat = valid_mask.ravel()

    for pixels in seg.nodes:
        px = pixels
        if valid_mask is not None:
            keep = mask_flat[px]
            px = px[keep]
        if px.size == 0:
            spectra_list.append(np.zeros((0, cube.shape[2]), dtype=np.float32))
            continue
        ys, xs = np.unravel_index(px, (h, w))
        spectra = np.asarray(cube[ys, xs, :], dtype=np.float32)
        finite = np.all(np.isfinite(spectra), axis=1)
        spectra_list.append(spectra[finite])
    return spectra_list


def _save_resume(
    resume_path: Path,
    node_thetas: np.ndarray,
    q16_nodes: np.ndarray,
    q84_nodes: np.ndarray,
    done_nodes: np.ndarray,
) -> None:
    np.savez_compressed(
        resume_path,
        node_thetas=node_thetas,
        q16_nodes=q16_nodes,
        q84_nodes=q84_nodes,
        done_nodes=done_nodes,
    )


def _load_resume(
    resume_path: Path,
    n_nodes: int,
    d: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not resume_path.exists():
        node_thetas = np.full((n_nodes, d), np.nan, dtype=np.float64)
        q16_nodes = np.full((n_nodes, d), np.nan, dtype=np.float64)
        q84_nodes = np.full((n_nodes, d), np.nan, dtype=np.float64)
        done_nodes = np.zeros((n_nodes,), dtype=bool)
        return node_thetas, q16_nodes, q84_nodes, done_nodes

    data = np.load(resume_path, allow_pickle=True)
    node_thetas = np.asarray(data["node_thetas"])
    q16_nodes = np.asarray(data["q16_nodes"])
    q84_nodes = np.asarray(data["q84_nodes"])
    done_nodes = np.asarray(data["done_nodes"]).astype(bool)
    if node_thetas.shape != (n_nodes, d):
        raise ValueError(
            f"Resume shape mismatch: node_thetas {node_thetas.shape} vs expected {(n_nodes, d)}"
        )
    return node_thetas, q16_nodes, q84_nodes, done_nodes


def _run_single_mcmc(
    spectrum: np.ndarray,
    normalizer: ParameterNormalizer,
    vary_bounds: list[tuple[float, float]],
    vary_indices: list[int],
    fixed_params: list[float],
    x_log_bounds: tuple[float, float],
    n_dim: int,
    n_walkers: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    noise_level: float,
    pool,
    seed: int,
) -> np.ndarray:
    import emcee

    if n_walkers < 2 * n_dim:
        raise ValueError(f"n_walkers must be >= 2*D ({2*n_dim}), got {n_walkers}")

    rng = np.random.default_rng(seed)
    obs_norm = normalizer.normalize_observation(spectrum)
    log_prob_evaluator = _NormalizedLogProbEvaluator(
        vary_bounds=vary_bounds,
        vary_indices=vary_indices,
        fixed_params=fixed_params,
        x_log_bounds=x_log_bounds,
        observation_norm=obs_norm,
        noise_level=noise_level,
    )
    p0 = rng.normal(loc=0.0, scale=0.25, size=(n_walkers, n_dim))
    p0 = np.clip(p0, -4.9, 4.9)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        log_prob_evaluator,
        pool=pool,
    )
    sampler.run_mcmc(p0, n_steps, progress=False)

    flat_norm = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    if flat_norm.shape[0] == 0:
        raise RuntimeError("Empty MCMC chain after burn-in/thin. Increase n_steps or adjust burn-in.")
    return np.asarray(normalizer.denormalize_params(flat_norm), dtype=np.float64)


def fit_cube_mcmc_resumable(
    cube: np.ndarray,
    all_param_bounds: list[tuple[float, float]],
    vary_indices: list[int],
    fixed_params: list[float],
    x_log_bounds: tuple[float, float],
    segmentation: str,
    block_k: int,
    valid_mask: np.ndarray | None,
    out_dir: str,
    resume_path: str,
    n_walkers: int,
    n_steps: int,
    burn_in: int,
    thin: int,
    noise_level: float,
    n_threads: int,
    checkpoint_every: int,
    max_nodes: int | None,
    save_samples: bool,
    seed: int,
    reuse_pixel_samples_dir: str | None = None,
    reuse_max_samples_per_pixel: int | None = None,
) -> MCMCFitResult:
    cube_arr = np.asarray(cube)
    if cube_arr.ndim != 3:
        raise ValueError(f"cube must be [H,W,F], got {cube_arr.shape}")

    h, w, _ = cube_arr.shape
    if segmentation == "pixel":
        seg = make_pixel_segmentation(h, w)
    elif segmentation == "block":
        seg = make_block_segmentation(h, w, block_k)
    else:
        raise ValueError(f"Unknown segmentation: {segmentation}")

    vary_bounds = [all_param_bounds[i] for i in vary_indices]
    d = len(vary_bounds)
    normalizer = ParameterNormalizer(vary_bounds, x_log_bounds=x_log_bounds, target_range=(-5, 5))

    node_spectra = _extract_node_spectra(cube_arr, seg, valid_mask=valid_mask)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resume = Path(resume_path)
    samples_dir = out_root / "mcmc_posteriors"
    if save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)
    reuse_samples_dir = Path(reuse_pixel_samples_dir) if reuse_pixel_samples_dir else None

    node_thetas, q16_nodes, q84_nodes, done_nodes = _load_resume(resume, seg.n_nodes, d)
    processed = 0
    sampled = 0
    pool = None
    if reuse_samples_dir is None and n_threads > 1:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=n_threads)

    try:
        for n in range(seg.n_nodes):
            if done_nodes[n]:
                continue
            if max_nodes is not None and sampled >= max_nodes:
                break

            spectra = node_spectra[n]
            if spectra.shape[0] == 0:
                done_nodes[n] = True
                processed += 1
                if processed % checkpoint_every == 0:
                    _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)
                continue

            if reuse_samples_dir is not None:
                # Reuse previously saved pixel-level MCMC samples:
                # pixel node id is the linear index y*W+x for pixel segmentation.
                chunked = []
                for lin in seg.nodes[n]:
                    pth = reuse_samples_dir / f"node_{int(lin):06d}.npy"
                    if not pth.exists():
                        continue
                    s = np.load(pth, allow_pickle=False)
                    if s.ndim != 2 or s.shape[1] != d or s.shape[0] == 0:
                        continue
                    if reuse_max_samples_per_pixel is not None and s.shape[0] > reuse_max_samples_per_pixel:
                        s = s[:reuse_max_samples_per_pixel]
                    chunked.append(s)
                if not chunked:
                    done_nodes[n] = True
                    processed += 1
                    if processed % checkpoint_every == 0:
                        _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)
                    continue
                samples = np.concatenate(chunked, axis=0)
            else:
                node_spec = np.nanmedian(spectra, axis=0)
                samples = _run_single_mcmc(
                    spectrum=node_spec,
                    normalizer=normalizer,
                    vary_bounds=vary_bounds,
                    vary_indices=vary_indices,
                    fixed_params=fixed_params,
                    x_log_bounds=x_log_bounds,
                    n_dim=d,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    burn_in=burn_in,
                    thin=thin,
                    noise_level=noise_level,
                    pool=pool,
                    seed=seed + n,
                )
            node_thetas[n] = np.median(samples, axis=0)
            q16_nodes[n] = np.quantile(samples, 0.16, axis=0)
            q84_nodes[n] = np.quantile(samples, 0.84, axis=0)
            if save_samples:
                np.save(samples_dir / f"node_{n:06d}.npy", samples)

            done_nodes[n] = True
            sampled += 1
            processed += 1
            if processed % checkpoint_every == 0:
                _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    _save_resume(resume, node_thetas, q16_nodes, q84_nodes, done_nodes)

    theta_map = node_theta_to_pixel_map(seg, node_thetas)
    q16_map = node_theta_to_pixel_map(seg, q16_nodes)
    q84_map = node_theta_to_pixel_map(seg, q84_nodes)

    return MCMCFitResult(
        theta_map=theta_map,
        q16_map=q16_map,
        q84_map=q84_map,
        node_thetas=node_thetas,
        q16_nodes=q16_nodes,
        q84_nodes=q84_nodes,
        seg=seg,
        done_nodes=done_nodes,
        debug={
            "resume_path": str(resume),
            "processed_nodes": int(processed),
            "sampled_nodes": int(sampled),
            "n_threads": int(n_threads),
            "reuse_pixel_samples_dir": None if reuse_samples_dir is None else str(reuse_samples_dir),
            "reuse_max_samples_per_pixel": None
            if reuse_max_samples_per_pixel is None
            else int(reuse_max_samples_per_pixel),
            "n_walkers": int(n_walkers),
            "n_steps": int(n_steps),
            "burn_in": int(burn_in),
            "thin": int(thin),
            "noise_level": float(noise_level),
        },
    )
