"""Minimal segmentation helpers for pixel- and block-wise fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segmentation:
    """Map node indices to flattened pixel indices."""

    height: int
    width: int
    nodes: list[np.ndarray]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def make_pixel_segmentation(height: int, width: int) -> Segmentation:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    nodes = [np.array([index], dtype=np.int64) for index in range(height * width)]
    return Segmentation(height=height, width=width, nodes=nodes)


def make_block_segmentation(height: int, width: int, block_k: int) -> Segmentation:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if block_k <= 0:
        raise ValueError("block_k must be positive")

    nodes: list[np.ndarray] = []
    for y0 in range(0, height, block_k):
        for x0 in range(0, width, block_k):
            pixels = []
            for yy in range(y0, min(y0 + block_k, height)):
                row_offset = yy * width
                for xx in range(x0, min(x0 + block_k, width)):
                    pixels.append(row_offset + xx)
            nodes.append(np.asarray(pixels, dtype=np.int64))
    return Segmentation(height=height, width=width, nodes=nodes)
