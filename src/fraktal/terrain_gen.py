"""Procedural terrain: fractal height fields, and fine detail on top of a coarse real DEM.

Randomness comes only from named sub-streams (ADR-0003); the smooth noise is the
``value_noise`` kernel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.noise import lattice_shape, value_noise


def _octaves(
    shape: tuple[int, int],
    gsd_m: float,
    seed: int,
    stream: str,
    largest_cell_m: float,
    smallest_cell_m: float,
    persistence: float,
) -> np.ndarray[Any, Any]:
    """Sum of noise octaves from ``largest_cell_m`` down to ``smallest_cell_m``, scaled to a
    peak-to-peak amplitude of about one."""
    if not 0 < persistence < 1:
        raise ValueError("persistence must be between 0 and 1")
    if smallest_cell_m <= 0 or largest_cell_m < smallest_cell_m:
        raise ValueError("cell sizes must satisfy 0 < smallest <= largest")
    field = np.zeros(shape, dtype=np.float32)
    total, cell_m, n, amp = 0.0, largest_cell_m, 0, 1.0
    while cell_m >= smallest_cell_m:
        cell_px = max(cell_m / gsd_m, 1.0)
        lattice = substream(seed, f"{stream}:{n}").uniform(
            -1, 1, size=lattice_shape(shape, cell_px)
        )
        field += np.float32(amp) * value_noise(lattice, cell_px, shape)
        total += amp
        cell_m, n, amp = cell_m / 2, n + 1, amp * persistence
    return field / np.float32(total)


def fractal_terrain(
    shape: tuple[int, int],
    gsd_m: float,
    seed: int,
    relief_m: float = 40.0,
    largest_cell_m: float = 800.0,
    smallest_cell_m: float = 4.0,
    persistence: float = 0.5,
) -> np.ndarray[Any, Any]:
    """A height field (metres, float32) with about ``relief_m`` of relief, from a seed alone."""
    field = _octaves(shape, gsd_m, seed, "terrain", largest_cell_m, smallest_cell_m, persistence)
    out: np.ndarray[Any, Any] = field * np.float32(relief_m / 2)
    return out


def add_detail(
    dem: np.ndarray[Any, Any],
    gsd_m: float,
    seed: int,
    amplitude_m: float = 1.5,
    largest_cell_m: float = 30.0,
    smallest_cell_m: float = 2.0,
    persistence: float = 0.55,
) -> np.ndarray[Any, Any]:
    """``dem`` plus fine relief below the real data's resolution (a 30 m DEM has none finer)."""
    detail = _octaves(
        dem.shape, gsd_m, seed, "terrain-detail", largest_cell_m, smallest_cell_m, persistence
    )
    out: np.ndarray[Any, Any] = dem.astype(np.float32) + detail * np.float32(amplitude_m / 2)
    return out
