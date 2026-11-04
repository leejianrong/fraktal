"""A tidal coast (V4): sea on one side, a low shore, mangrove belts and creeks.

The coast is carved into the height field before land cover is made, so drainage sees it, and
its classes are painted over the land cover afterwards. The shore runs across the tile at a
seeded angle with a wobble; height rises slowly inland, so the tidal flat (up to 0.4 m above the
sea) and the mangrove belt (up to about 2 m) are bands of ground, not drawn shapes. Creeks are
least-cost paths from the mangrove belt to the sea, carved below sea level. Randomness comes from
the sub-streams ``coast:shore`` and ``coast:creeks`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.pathing import least_cost_path
from fraktal.landcover_gen import WATER
from fraktal.terrain_gen import octave_noise

TIDAL_FLAT = 90  # ESA WorldCover herbaceous wetland: bare tidal mud
MANGROVE = 95
SEA_LEVEL_M = 300.0  # the terrain generator's base height
UNBUILDABLE = (WATER, TIDAL_FLAT, MANGROVE)


@dataclass(frozen=True)
class CoastConfig:
    sea_share: float = 0.0  # share of the tile that is open sea; 0 means no coast
    shore_wobble_m: float = 70.0
    inland_gradient: float = 0.005  # metres of height per metre inland
    flat_height_m: float = 0.4  # tidal flat: below this above the sea
    mangrove_height_m: float = 2.2  # mangrove belt: up to this above the sea
    creeks: int = 4
    creek_width_m: float = 14.0
    creek_bank_m: float = 30.0  # mangrove hugs creeks out to this distance


@dataclass(frozen=True)
class CoastMasks:
    sea: Any  # open sea and creeks (water)
    flat: Any
    mangrove: Any
    creek: Any
    inland_m: Any  # signed distance inland from the shore (negative at sea)
    pond_zone: Any = None  # dry ground just behind the mangrove belt, where ponds are dug


def box_any(mask: Any, radius: int) -> Any:
    """True where any cell within ``radius`` (a square window) of a cell is set."""
    if radius <= 0:
        return mask.copy()
    pad = np.pad(mask.astype("int32"), radius + 1)
    csum = pad.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    h, w = mask.shape
    total = (
        csum[k : k + h, k : k + w] - csum[0:h, k : k + w] - csum[k : k + h, 0:w] + csum[0:h, 0:w]
    )
    return total > 0


def shape_coast(dem: Any, gsd_m: float, seed: int, config: CoastConfig) -> tuple[Any, CoastMasks]:
    """``dem`` re-cut as a coast, and the masks that land cover is painted from."""
    h, w = dem.shape
    rng = substream(seed, "coast:shore")
    angle = float(rng.uniform(0.0, 2 * math.pi))
    rr, cc = np.mgrid[0:h, 0:w].astype("float64")
    proj = ((cc - w / 2) * math.cos(angle) + (rr - h / 2) * math.sin(angle)) * gsd_m
    wobble = octave_noise((h, w), gsd_m, seed, "coast:shore", 400.0, 40.0, 0.5)
    proj = proj + config.shore_wobble_m * wobble
    shore = float(np.quantile(proj, 1.0 - config.sea_share))
    inland = shore - proj  # metres inland; negative over the sea
    local = np.clip(inland / 800.0, 0.0, 1.0)  # the ground only roughens away from the shore
    height = np.clip(inland * config.inland_gradient, -3.0, 8.0) + (dem - SEA_LEVEL_M) * local
    sea = height <= 0.0
    creek = _creeks(height, sea, inland, gsd_m, seed, config)
    height = np.where(creek, np.minimum(height, -0.3), height)
    sea = height <= 0.0
    noise = octave_noise((h, w), gsd_m, seed, "coast:bands", 120.0, 15.0, 0.5)
    band = height + 0.25 * noise
    near_creek = box_any(creek, int(config.creek_bank_m / gsd_m))
    flat = (~sea) & (band <= config.flat_height_m)
    mangrove = (~sea) & ~flat & ((band <= config.mangrove_height_m) | (near_creek & (height < 4.0)))
    zone = (~sea) & ~flat & ~mangrove & (inland > 380.0) & (inland < 1100.0) & (height < 6.0)
    masks = CoastMasks(sea, flat, mangrove, creek, inland, zone)
    return (SEA_LEVEL_M + height).astype("float32"), masks


def _creeks(
    height: Any, sea: Any, inland: Any, gsd_m: float, seed: int, config: CoastConfig
) -> Any:
    """Cells of creeks: winding least-cost paths from the mangrove belt to the sea."""
    h, w = height.shape
    out = np.zeros((h, w), dtype=bool)
    if config.creeks <= 0 or not sea.any():
        return out
    rng = substream(seed, "coast:creeks")
    wander = octave_noise((h, w), gsd_m, seed, "coast:creek_cost", 150.0, 20.0, 0.6)
    cost = 1.0 + 14.0 * (wander - wander.min()) + np.clip(height, 0, None)
    cost[sea] = 0.5
    belt = np.argwhere((inland > 150.0) & (inland < 450.0))
    sea_cells = np.argwhere(sea & (inland > -250.0) & (inland < -20.0))
    if len(belt) == 0 or len(sea_cells) == 0:
        return out
    width = max(1, int(round(config.creek_width_m / gsd_m / 2)))
    for _ in range(config.creeks):
        pick = belt[int(rng.integers(0, len(belt)))]
        start = (int(pick[0]), int(pick[1]))
        goal = sea_cells[int(np.argmin(np.hypot(*(sea_cells - np.array(start)).T)))]
        path = least_cost_path(cost, start, (int(goal[0]), int(goal[1])))
        if path is None:
            continue
        line = np.zeros((h, w), dtype=bool)
        line[path[:, 0], path[:, 1]] = True
        out |= box_any(line, width)
    return out


def paint_coast(land: Any, masks: CoastMasks) -> Any:
    """Land cover with sea and creeks as water, tidal flat as wetland and mangrove belts."""
    out = land.copy()
    out[masks.flat] = TIDAL_FLAT
    out[masks.mangrove] = MANGROVE
    out[masks.sea] = WATER
    return out


__all__ = [
    "MANGROVE",
    "SEA_LEVEL_M",
    "TIDAL_FLAT",
    "UNBUILDABLE",
    "box_any",
    "CoastConfig",
    "CoastMasks",
    "paint_coast",
    "shape_coast",
]
