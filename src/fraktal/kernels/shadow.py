"""Cast-shadow ray march over a digital surface model (ADR-0004, ADR-0005).

Arrays and plain numbers in, an array out. Rows run north to south, columns west to east.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _march(
    dsm: np.ndarray[Any, Any], d_row: float, d_col: float, rise_per_px: float, max_steps: int
) -> np.ndarray[Any, Any]:
    rows, cols = dsm.shape
    out = np.zeros((rows, cols), dtype=np.uint8)
    for r in prange(rows):
        for c in range(cols):
            h0 = dsm[r, c]
            for s in range(1, max_steps + 1):
                rr = int(round(r + s * d_row))
                cc = int(round(c + s * d_col))
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    break
                if dsm[rr, cc] > h0 + s * rise_per_px:
                    out[r, c] = 1
                    break
    return out


def shadow_mask(
    dsm: np.ndarray[Any, Any],
    pixel_size_m: float,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> np.ndarray[Any, Any]:
    """1 where the sun is blocked, else 0 (uint8, same shape as ``dsm``).

    ``sun_azimuth_deg`` is the compass bearing of the sun, clockwise from north. The sun is
    sampled once per pixel step along its bearing, with nearest-pixel lookups.
    """
    if dsm.ndim != 2:
        raise ValueError("dsm must be 2D")
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    if not 0 < sun_elevation_deg < 90:
        raise ValueError("sun_elevation_deg must be strictly between 0 and 90")
    surface = np.ascontiguousarray(dsm, dtype=np.float64)
    az = math.radians(sun_azimuth_deg)
    rise = math.tan(math.radians(sun_elevation_deg)) * pixel_size_m
    relief = float(surface.max() - surface.min())
    max_steps = min(int(relief / rise) + 2, max(surface.shape))
    result: np.ndarray[Any, Any] = _march(surface, -math.cos(az), math.sin(az), rise, max_steps)
    return result
