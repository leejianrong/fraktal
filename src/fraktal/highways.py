"""Highways and an interchange (FRK-189, first slice): a wide road across the tile with ramps.

A highway is an ordinary ``arterial`` row, only wider, so lane markings, vehicles, traffic and the
renderers treat it like any arterial. Where it crosses water the run is a bridge. At its first
crossing with another arterial it gets a cloverleaf: four 270 degree loops, one in each quadrant.
Randomness comes from the sub-streams ``highways:*`` only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString, Point

from fraktal.coast import UNBUILDABLE
from fraktal.ids import substream
from fraktal.roads import PROVENANCE
from fraktal.terrain_gen import octave_noise
from fraktal.world import make_layer


@dataclass(frozen=True)
class HighwayConfig:
    count: int = 0
    width_m: float = 28.0
    ramp_width_m: float = 9.0
    loop_radius_m: float = 42.0
    swing_m: float = 220.0  # sideways drift of the course from a straight line


def _course(side_m: float, seed: int, index: int, config: HighwayConfig) -> LineString:
    rng = substream(seed, f"highways:course:{index}")
    n = 100
    t = np.linspace(0.0, 1.0, n)
    swing = octave_noise(
        (1, n), side_m / n, seed, f"highways:swing:{index}", side_m, side_m / 6, 0.5
    )[0]
    swing = swing - swing.mean()
    swing = swing / max(float(np.abs(swing).max()), 1e-6) * config.swing_m
    a, b = (float(rng.uniform(0.2, 0.8)) * side_m for _ in range(2))
    across, along = a + (b - a) * t + swing, t * side_m
    pts = (
        np.column_stack([across, along])
        if rng.uniform() < 0.5
        else np.column_stack([along, across])
    )
    return LineString(pts)


def _cloverleaf(
    crossing: Point, highway: Any, other: Any, config: HighwayConfig, other_width: float
) -> list[LineString]:
    """Four loops around ``crossing``: tangent to the highway on one side and the other road."""

    def heading(line: Any) -> tuple[float, float]:
        d = line.project(crossing)
        a, b = line.interpolate(max(d - 5.0, 0.0)), line.interpolate(min(d + 5.0, line.length))
        n = math.hypot(b.x - a.x, b.y - a.y) or 1.0
        return (b.x - a.x) / n, (b.y - a.y) / n

    ux, uy = heading(highway)  # along the highway
    vx, vy = heading(other)  # along the other road
    r = config.loop_radius_m
    along_off = other_width / 2 + r  # centre distance from the crossing, along the highway
    across_off = config.width_m / 2 + r  # and along the other road
    loops = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            cx = crossing.x + sx * along_off * ux + sy * across_off * vx
            cy = crossing.y + sx * along_off * uy + sy * across_off * vy
            # the 270 degree arc that leaves the highway edge and meets the other road's edge
            start = math.atan2(-sy * vy, -sy * vx) + 0.0
            sweep = -1.5 * math.pi * sx * sy
            angles = start + np.linspace(0.0, sweep, 28)
            loops.append(LineString([(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]))
    return loops


def add_highways(
    roads: gpd.GeoDataFrame,
    landcover: Any,
    gsd_m: float,
    transform: Affine,
    crs: str,
    seed: int,
    config: HighwayConfig,
    world: str = "roads",
) -> gpd.GeoDataFrame:
    """``roads`` plus the configured highways (bridged over water) and a cloverleaf each."""
    if config.count <= 0:
        return roads
    inverse = ~transform
    west, north = transform @ (0, 0)
    side_m = landcover.shape[0] * gsd_m
    rows = [
        {
            "geometry": r.geometry,
            "provenance": r.provenance,
            "road_class": r.road_class,
            "width_m": float(r.width_m),
            "bridge": float(r.bridge),
        }  # fmt: skip
        for r in roads.sort_values("id").itertuples()
    ]
    arterials = roads[roads.road_class == "arterial"].sort_values("id")

    def wet(x: float, y: float) -> bool:
        col, row = inverse @ (x, y)
        r, c = int(math.floor(row)), int(math.floor(col))
        inside = 0 <= r < landcover.shape[0] and 0 <= c < landcover.shape[1]
        return inside and bool(landcover[r, c] in UNBUILDABLE)

    for i in range(config.count):
        local = _course(side_m, seed, i, config)
        line = LineString([(west + x, north - side_m + y) for x, y in local.coords])
        n = max(2, int(line.length / 4.0))
        pts = [line.interpolate(k / n, normalized=True) for k in range(n + 1)]
        state = [wet(p.x, p.y) for p in pts]
        start = 0
        for k in range(1, len(pts) + 1):
            if k < len(pts) and state[k] == state[start]:
                continue
            lo, hi = (max(start - 1, 0), min(k + 1, len(pts))) if state[start] else (start, k)
            if hi - lo >= 2:
                rows.append(
                    {
                        "geometry": LineString([(p.x, p.y) for p in pts[lo:hi]]),
                        "provenance": PROVENANCE, "road_class": "arterial",
                        "width_m": config.width_m, "bridge": 1.0 if state[start] else 0.0,
                    }
                )  # fmt: skip
            start = k
        for _, other in arterials.iterrows():
            hit = line.intersection(other.geometry)
            point = hit if isinstance(hit, Point) else None
            if point is None or wet(point.x, point.y):
                continue
            for loop in _cloverleaf(point, line, other.geometry, config, float(other.width_m)):
                if not any(wet(x, y) for x, y in loop.coords):
                    rows.append(
                        {
                            "geometry": loop, "provenance": PROVENANCE, "road_class": "arterial",
                            "width_m": config.ramp_width_m, "bridge": 0.0,
                        }
                    )  # fmt: skip
            break
    return make_layer("road", world, rows, crs)


__all__ = ["HighwayConfig", "add_highways"]
