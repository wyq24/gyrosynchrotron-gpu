"""Helpers for expanding node-level fit results back to pixel maps."""

from __future__ import annotations

import numpy as np

try:
    from .segmentation import Segmentation
except ImportError:
    from segmentation import Segmentation


def node_theta_to_pixel_map(seg: Segmentation, node_values: np.ndarray) -> np.ndarray:
    values = np.asarray(node_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"node_values must be [N,D], got {values.shape}")
    if values.shape[0] != seg.n_nodes:
        raise ValueError(f"node_values first dimension {values.shape[0]} does not match {seg.n_nodes} nodes")

    out = np.full((seg.height, seg.width, values.shape[1]), np.nan, dtype=np.float64)
    for node_index, pixels in enumerate(seg.nodes):
        if pixels.size == 0:
            continue
        ys, xs = np.unravel_index(pixels, (seg.height, seg.width))
        out[ys, xs, :] = values[node_index]
    return out
