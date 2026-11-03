"""Sensor model: optical blur (PSF) and read/shot noise applied after shading (V3).

Blur needs a whole band, so it runs on the region before tiling. Noise is drawn per tile from its
own named sub-stream, so a tile's noise does not depend on how many tiles surround it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.filters import gaussian_blur


def apply_psf(rgb: Any, sigma_px: float) -> Any:
    """Gaussian point-spread blur of a (3, H, W) uint8 image, one band at a time."""
    if sigma_px <= 0:
        return rgb
    out = np.empty_like(rgb)
    for band in range(rgb.shape[0]):
        blurred = gaussian_blur(rgb[band].astype("float32"), sigma_px)
        out[band] = np.clip(np.rint(blurred), 0, 255).astype("uint8")
    return out


def add_noise(tile: Any, sigma: float, seed: int, name: str) -> Any:
    """Add zero-mean Gaussian noise (std ``sigma`` in digital numbers) to a (3, H, W) uint8 tile."""
    if sigma <= 0:
        return tile
    rng = substream(seed, f"sensor:noise:{name}")
    noise = rng.standard_normal(tile.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(np.rint(tile.astype("float32") + noise), 0, 255).astype("uint8")


def blank_wedge(
    shape: tuple[int, int], prob: float, max_fraction: float, seed: int, name: str
) -> Any:
    """Boolean mask of a no-data region cut off one side of a tile, like a scene footprint edge.

    With probability ``prob`` a half-plane at a random angle covers 5% up to ``max_fraction`` of
    the tile; otherwise nothing is blank. Deterministic per seed and tile name.
    """
    blank = np.zeros(shape, dtype=bool)
    if prob <= 0:
        return blank
    rng = substream(seed, f"blank:{name}")
    if rng.uniform() >= prob:
        return blank
    angle = rng.uniform(0, 2 * np.pi)
    fraction = rng.uniform(0.05, max(0.05, max_fraction))
    rows, cols = np.indices(shape, dtype=np.float32)
    distance = (cols - shape[1] / 2) * np.cos(angle) + (rows - shape[0] / 2) * np.sin(angle)
    return np.asarray(distance > np.quantile(distance, 1 - fraction))
