"""D8 flow direction and accumulation over a height field (ADR-0005): arrays in, arrays out.

Each cell drains to its steepest lower neighbour among the eight around it. Cells with no lower
in-bounds neighbour are outlets (pits and the low edge) and keep what drains into them.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

# Neighbour offsets (row, column), north first and clockwise.
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


@njit(cache=True)
def _directions(
    dem: np.ndarray[Any, Any], dr: np.ndarray[Any, Any], dc: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    rows, cols = dem.shape
    out = np.full((rows, cols), -1, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            best, best_drop = -1, 0.0
            for k in range(8):
                rr, cc = r + dr[k], c + dc[k]
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                dist = math.sqrt(2.0) if dr[k] != 0 and dc[k] != 0 else 1.0
                drop = (dem[r, c] - dem[rr, cc]) / dist
                if drop > best_drop:
                    best, best_drop = k, drop
            out[r, c] = best
    return out


@njit(cache=True)
def _accumulate(
    direction: np.ndarray[Any, Any],
    order: np.ndarray[Any, Any],
    dr: np.ndarray[Any, Any],
    dc: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    rows, cols = direction.shape
    acc = np.ones((rows, cols), dtype=np.float64)
    for flat in order:  # highest first, so a cell is complete before it drains downhill
        r, c = flat // cols, flat % cols
        k = direction[r, c]
        if k >= 0:
            acc[r + dr[k], c + dc[k]] += acc[r, c]
    return acc


def flow_accumulation(
    dem: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Cells draining through each cell (itself included) and the D8 direction index per cell.

    Directions index (north, north-east, east, ... north-west); -1 marks an outlet.
    """
    if dem.ndim != 2:
        raise ValueError("dem must be 2D")
    surface = np.ascontiguousarray(dem, dtype=np.float64)
    direction = _directions(surface, DR, DC)
    order = np.argsort(-surface.ravel(), kind="stable")  # stable: ties break by cell index
    acc: np.ndarray[Any, Any] = _accumulate(direction, order, DR, DC)
    return acc, direction
