"""Sensor model: optical blur (PSF) and read/shot noise applied after shading (V3).

The arithmetic lives in ``fraktal.kernels.sensor`` (shared with the 3D path); this module keeps
the seeded wrappers and the no-data wedge.

Blur needs a whole band, so it runs on the region before tiling. Noise is drawn per tile from its
own named sub-stream, so a tile's noise does not depend on how many tiles surround it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.sensor import add_gaussian_noise, apply_psf


def add_noise(tile: Any, sigma: float, seed: int, name: str) -> Any:
    """Gaussian noise (std ``sigma`` in digital numbers) from the tile's own named sub-stream."""
    return add_gaussian_noise(tile, sigma, substream(seed, f"sensor:noise:{name}"))


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


__all__ = ["add_noise", "apply_psf", "blank_wedge"]
