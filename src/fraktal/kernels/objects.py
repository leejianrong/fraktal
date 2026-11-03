"""Paint small objects (tree crowns, vehicles) onto a grid (ADR-0005): arrays in, arrays out.

Each object is an ellipse (kind 0, a dome whose height tapers to the edge) or a rectangle (kind 1,
flat-topped). Objects are painted in array order, so later ones win overlaps, and they never
paint onto ``blocked`` pixels (buildings), so building labels are untouched.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True)
def _paint(
    height: np.ndarray[Any, Any],
    colour: np.ndarray[Any, Any],
    covered: np.ndarray[Any, Any],
    blocked: np.ndarray[Any, Any],
    shapes: np.ndarray[Any, Any],
    heights: np.ndarray[Any, Any],
    colours: np.ndarray[Any, Any],
) -> None:
    rows, cols = height.shape
    for i in range(shapes.shape[0]):
        r0, c0, a, b, theta, kind = shapes[i]
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        reach = int(math.ceil(max(a, b))) + 1
        for r in range(max(0, int(r0) - reach), min(rows, int(r0) + reach + 1)):
            for c in range(max(0, int(c0) - reach), min(cols, int(c0) + reach + 1)):
                if blocked[r, c]:
                    continue
                dy, dx = r - r0, c - c0
                u = dx * cos_t + dy * sin_t
                v = -dx * sin_t + dy * cos_t
                if kind == 0:
                    rho2 = (u / a) ** 2 + (v / b) ** 2
                    if rho2 > 1.0:
                        continue
                    h = heights[i] * math.sqrt(1.0 - rho2)
                else:
                    if abs(u) > a or abs(v) > b:
                        continue
                    h = heights[i]
                height[r, c] = h
                covered[r, c] = 1
                for k in range(3):
                    colour[r, c, k] = colours[i, k]


def paint_objects(
    shape: tuple[int, int],
    shapes: np.ndarray[Any, Any],
    heights: np.ndarray[Any, Any],
    colours: np.ndarray[Any, Any],
    blocked: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Height (float32), colour (H, W, 3 float32) and a 0/1 coverage mask for the objects.

    ``shapes`` is (N, 6): row and column of the centre, the two semi-axes in pixels, the angle
    in radians and the kind (0 ellipse, 1 rectangle). ``heights`` is (N,), ``colours`` (N, 3).
    """
    if shapes.ndim != 2 or shapes.shape[1] != 6:
        raise ValueError("shapes must be (N, 6)")
    if not (len(heights) == len(colours) == len(shapes)):
        raise ValueError("shapes, heights and colours must have the same length")
    if blocked.shape != shape:
        raise ValueError("blocked must match the grid shape")
    height = np.zeros(shape, dtype=np.float32)
    colour = np.zeros((*shape, 3), dtype=np.float32)
    covered = np.zeros(shape, dtype=np.uint8)
    _paint(
        height,
        colour,
        covered,
        np.ascontiguousarray(blocked, dtype=np.uint8),
        np.ascontiguousarray(shapes, dtype=np.float64),
        np.ascontiguousarray(heights, dtype=np.float64),
        np.ascontiguousarray(colours, dtype=np.float64),
    )
    return height, colour, covered
