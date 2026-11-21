"""Rivers and lakes (FRK-187): meandering channels and rounded basins in the height field.

Painted after land cover, like the coast: the water cells become ``WATER`` (class ground in the
labels), the height field is lowered under them, and a green bank forms beside the water where the
climate is not dry. Roads are built afterwards, so an arterial that meets a river bridges it.
Randomness comes from the sub-streams ``rivers:*`` and ``lakes:*`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import LineString, Polygon

from fraktal.coast import box_any
from fraktal.ids import substream
from fraktal.landcover_gen import BARE, GRASS, SHRUB, WATER
from fraktal.terrain_gen import octave_noise

BANK_M = 14.0  # green bank on each side of the water
CARVE_M = 2.5  # how far the water surface sits below the ground it replaces


@dataclass(frozen=True)
class WaterConfig:
    rivers: int = 0
    river_width_m: tuple[float, float] = (10.0, 28.0)
    meander_m: float = 140.0  # sideways swing of a river about its straight course
    lakes: int = 0
    lake_radius_m: tuple[float, float] = (50.0, 160.0)


def _river(side_m: float, rng: Any, config: WaterConfig, index: int, seed: int) -> LineString:
    """A course from one edge to the opposite one, swinging sideways by smooth noise."""
    n = 80
    t = np.linspace(0.0, 1.0, n)
    swing = octave_noise(
        (1, n), side_m / n, seed, f"rivers:swing:{index}", side_m / 1.5, side_m / 10, 0.5
    )[0]
    swing = swing - swing.mean()
    swing = swing / max(float(np.abs(swing).max()), 1e-6) * config.meander_m
    start = float(rng.uniform(0.2, 0.8)) * side_m
    end = float(rng.uniform(0.2, 0.8)) * side_m
    across = start + (end - start) * t + swing
    along = t * side_m
    pts = (
        np.column_stack([across, along])
        if rng.uniform() < 0.5
        else np.column_stack([along, across])
    )
    return LineString(pts)


def _lake(side_m: float, rng: Any, config: WaterConfig) -> Polygon:
    r = float(rng.uniform(*config.lake_radius_m))
    cx, cy = (float(rng.uniform(0.15, 0.85)) * side_m for _ in range(2))
    angles = np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False)
    wobble = 1.0 + 0.3 * np.convolve(rng.uniform(-1, 1, 36 + 4), np.ones(5) / 5, mode="valid")[:36]
    return Polygon(
        np.column_stack([cx + r * wobble * np.cos(angles), cy + r * wobble * np.sin(angles)])
    )


def paint_waters(
    dem: Any, land: Any, gsd_m: float, seed: int, config: WaterConfig, aridity: float
) -> tuple[Any, Any]:
    """``(dem, land)`` with the configured rivers and lakes carved in and painted as water."""
    if config.rivers <= 0 and config.lakes <= 0:
        return dem, land
    side_m = dem.shape[0] * gsd_m
    shapes: list[tuple[Any, int]] = []
    for i in range(config.rivers):
        rng = substream(seed, f"rivers:course:{i}")
        width = float(rng.uniform(*config.river_width_m))
        shapes.append((_river(side_m, rng, config, i, seed).buffer(width / 2), 1))
    for j in range(config.lakes):
        shapes.append((_lake(side_m, substream(seed, f"lakes:shape:{j}"), config), 1))
    scaled = [(_scale(g, 1.0 / gsd_m), v) for g, v in shapes]  # metres to pixels, y as rows
    water = rasterize(scaled, dem.shape, transform=Affine.identity(), dtype="uint8") > 0
    out_dem = np.where(water, dem - CARVE_M, dem).astype(dem.dtype)
    out = land.copy()
    if aridity < 0.7:
        bank = box_any(water, max(1, int(BANK_M / gsd_m))) & ~water
        out[bank & np.isin(out, (BARE, SHRUB))] = GRASS
    out[water] = WATER
    return out_dem, out


def _scale(geom: Any, factor: float) -> Any:
    from shapely import affinity

    return affinity.scale(geom, factor, factor, origin=(0, 0))


__all__ = ["WaterConfig", "paint_waters"]
