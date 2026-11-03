"""Procedural land cover from terrain (V2), in ESA WorldCover codes.

Water gathers where a lot of land drains into low ground. Elsewhere the class follows a wetness
score (catchment over slope) with seeded noise, shifted by aridity, so valleys are green and
ridges are bare. Real classes are kept wherever they are given.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.kernels.hydrology import flow_accumulation
from fraktal.terrain_gen import octave_noise

TREE, SHRUB, GRASS, CROP, BARE, WATER = 10, 20, 30, 40, 60, 80
STEEP_DEG = 25.0


def slope_degrees(dem: np.ndarray[Any, Any], gsd_m: float) -> np.ndarray[Any, Any]:
    dy, dx = np.gradient(dem.astype("float64"), gsd_m)
    out: np.ndarray[Any, Any] = np.degrees(np.arctan(np.hypot(dx, dy)))
    return out


def generate_landcover(
    dem: np.ndarray[Any, Any],
    gsd_m: float,
    seed: int,
    real: np.ndarray[Any, Any] | None = None,
    aridity: float = 0.5,
    water_min_catchment_m2: float = 1_000.0,
    water_elevation_quantile: float = 0.15,
) -> np.ndarray[Any, Any]:
    """A uint8 WorldCover-coded raster the shape of ``dem``.

    ``real`` is an optional raster of real classes where 0 means missing: those cells are kept.
    ``aridity`` runs from 0 (lush) to 1 (barren).
    """
    if dem.ndim != 2:
        raise ValueError("dem must be 2D")
    if not 0.0 <= aridity <= 1.0:
        raise ValueError("aridity must be between 0 and 1")
    if real is not None and real.shape != dem.shape:
        raise ValueError("real must match the shape of dem")
    acc, _ = flow_accumulation(dem)
    slope = slope_degrees(dem, gsd_m)
    catchment = acc * gsd_m**2
    twi = np.log((acc * gsd_m) / (np.tan(np.radians(slope)) + 0.01))
    lo, hi = np.quantile(twi, [0.05, 0.95])
    wet = np.clip((twi - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    noise = octave_noise(dem.shape, gsd_m, seed, "landcover", 120.0, 8.0, 0.5)
    score = wet + 0.3 * noise - 0.7 * (aridity - 0.5)
    out = np.full(dem.shape, BARE, dtype=np.uint8)
    out[score > 0.20] = SHRUB
    out[score > 0.38] = GRASS
    out[score > 0.55] = CROP
    out[score > 0.75] = TREE
    out[slope > STEEP_DEG] = BARE
    low = dem <= np.quantile(dem, water_elevation_quantile)
    out[(catchment >= water_min_catchment_m2) & low] = WATER
    if real is not None:
        out = np.where(real > 0, real, out).astype(np.uint8)
    return out
