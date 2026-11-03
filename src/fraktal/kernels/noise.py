"""Smooth value noise on a lattice (ADR-0005): arrays in, an array out, no random state.

The caller draws the lattice (through ``fraktal.ids.substream``), so determinism stays with the
caller and this kernel is a pure function of its inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _interpolate(
    lattice: np.ndarray[Any, Any], cell_px: float, rows: int, cols: int
) -> np.ndarray[Any, Any]:
    out = np.empty((rows, cols), dtype=np.float32)
    for r in prange(rows):
        y = r / cell_px
        y0 = int(y)
        fy = y - y0
        wy = fy * fy * (3.0 - 2.0 * fy)
        for c in range(cols):
            x = c / cell_px
            x0 = int(x)
            fx = x - x0
            wx = fx * fx * (3.0 - 2.0 * fx)
            top = lattice[y0, x0] * (1.0 - wx) + lattice[y0, x0 + 1] * wx
            bottom = lattice[y0 + 1, x0] * (1.0 - wx) + lattice[y0 + 1, x0 + 1] * wx
            out[r, c] = top * (1.0 - wy) + bottom * wy
    return out


def lattice_shape(shape: tuple[int, int], cell_px: float) -> tuple[int, int]:
    """Lattice size ``value_noise`` needs to cover an output of ``shape``."""
    return (int((shape[0] - 1) / cell_px) + 2, int((shape[1] - 1) / cell_px) + 2)


def value_noise(
    lattice: np.ndarray[Any, Any], cell_px: float, shape: tuple[int, int]
) -> np.ndarray[Any, Any]:
    """Smoothstep-interpolated noise (float32, ``shape``), equal to the lattice at its nodes.

    Output pixel (r, c) lies at lattice coordinate (r / cell_px, c / cell_px), so the values stay
    within the lattice's min and max.
    """
    if lattice.ndim != 2:
        raise ValueError("lattice must be 2D")
    if cell_px <= 0:
        raise ValueError("cell_px must be positive")
    need = lattice_shape(shape, cell_px)
    if lattice.shape[0] < need[0] or lattice.shape[1] < need[1]:
        raise ValueError(f"lattice must be at least {need} for this shape and cell size")
    grid = np.ascontiguousarray(lattice, dtype=np.float64)
    result: np.ndarray[Any, Any] = _interpolate(grid, float(cell_px), shape[0], shape[1])
    return result
