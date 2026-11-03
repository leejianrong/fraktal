"""Procedural road network (V2): arterials between settlement centres, then local streets.

Arterials follow least-cost routes over terrain and may cross water on bridge segments. Local
streets branch off arterials at intervals and stop at water, steep ground or an existing road.
Randomness comes only from the named sub-streams ``roads:centres``, ``roads:links`` and
``roads:local``, so nothing else in the world (buildings, say) can change the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from fraktal.ids import substream
from fraktal.kernels.pathing import least_cost_path
from fraktal.landcover_gen import WATER, slope_degrees
from fraktal.world import make_layer

PROVENANCE = "procedural:roads"


class RoadError(ValueError):
    pass


@dataclass(frozen=True)
class RoadConfig:
    arterial_width_m: float = 12.0
    local_width_m: float = 6.0
    min_centre_separation_m: float = 450.0
    centres_per_km2: float = 1.5
    local_spacing_m: float = 90.0
    local_length_m: tuple[float, float] = (60.0, 220.0)
    local_probability: float = 0.85
    max_local_slope_deg: float = 18.0
    smooth_tolerance_cells: float = 1.5  # simplify first; 0 keeps the raw path
    smooth_rounds: int = 3
    bridge_cost: float = 40.0  # per water cell for an arterial; about 1 on flat land


def _cost(slope: Any, water: Any, allow_water: bool, config: RoadConfig) -> Any:
    cost = 1.0 + (slope / 6.0) ** 2
    cost[slope > 35.0] = np.inf
    if allow_water:
        cost[water] = config.bridge_cost
    else:
        cost[water] = np.inf
        cost[slope > config.max_local_slope_deg] = np.inf
    return cost


def _choose_centres(
    slope: Any, water: Any, gsd_m: float, seed: int, config: RoadConfig
) -> list[tuple[int, int]]:
    rows, cols = water.shape
    area_km2 = rows * cols * gsd_m**2 / 1e6
    wanted = int(min(8, max(2, round(area_km2 * config.centres_per_km2))))
    rng = substream(seed, "roads:centres")
    candidates = np.column_stack(
        [rng.integers(0, rows, 60 * wanted), rng.integers(0, cols, 60 * wanted)]
    )
    ok = candidates[
        ~water[candidates[:, 0], candidates[:, 1]]
        & (slope[candidates[:, 0], candidates[:, 1]] < 8.0)
    ]
    separation = config.min_centre_separation_m / gsd_m
    chosen: list[tuple[int, int]] = []
    for r, c in ok:
        if all(math.hypot(r - a, c - b) >= separation for a, b in chosen):
            chosen.append((int(r), int(c)))
        if len(chosen) == wanted:
            break
    if len(chosen) < 2:
        raise RoadError("could not place two settlement centres on dry, gentle ground")
    return chosen


def _links(centres: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    """Minimum spanning tree over the centres, plus a few seeded extra links for loops."""
    n = len(centres)

    def dist(i: int, j: int) -> float:
        return math.hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1])

    in_tree = {0}
    edges: list[tuple[int, int]] = []
    while len(in_tree) < n:
        pairs = [(i, j) for i in sorted(in_tree) for j in range(n) if j not in in_tree]
        i, j = min(pairs, key=lambda e: dist(e[0], e[1]))
        edges.append((i, j))
        in_tree.add(j)
    used = {(min(a, b), max(a, b)) for a, b in edges}
    rng = substream(seed, "roads:links")
    extra: set[tuple[int, int]] = set()
    for i in range(n):
        near = sorted(
            (j for j in range(n) if j != i and (min(i, j), max(i, j)) not in used),
            key=lambda j: dist(i, j),
        )
        if near and rng.uniform() < 0.5:
            extra.add((min(i, near[0]), max(i, near[0])))
    return edges + sorted(extra)


def _xy(cells: Any, transform: Affine) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for r, c in cells:
        x, y = transform @ (float(c) + 0.5, float(r) + 0.5)
        out.append((float(x), float(y)))
    return out


def _chaikin(points: list[tuple[float, float]], rounds: int) -> list[tuple[float, float]]:
    """Corner cutting: each round replaces every corner by points a quarter of the way along."""
    for _ in range(rounds):
        cut = [points[0]]
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=True):
            cut.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            cut.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        cut.append(points[-1])
        points = cut
    return points


def smooth_arterial(
    coords: list[tuple[float, float]], gsd_m: float, config: RoadConfig
) -> list[tuple[float, float]]:
    """Straighten a cell staircase: simplify, then cut corners. Both end points stay put."""
    if len(coords) < 3 or config.smooth_tolerance_cells <= 0:
        return coords
    simple = LineString(coords).simplify(config.smooth_tolerance_cells * gsd_m)
    return _chaikin(list(simple.coords), config.smooth_rounds)


def _stays_dry(
    coords: list[tuple[float, float]], water: Any, inverse: Affine, step_m: float
) -> bool:
    """True if no sample along the line, every ``step_m``, falls in a water cell."""
    line = LineString(coords)
    for d in np.arange(0.0, line.length + step_m, step_m):
        p = line.interpolate(min(float(d), line.length))
        col, row = inverse @ (p.x, p.y)
        r, c = int(math.floor(row)), int(math.floor(col))
        if 0 <= r < water.shape[0] and 0 <= c < water.shape[1] and water[r, c]:
            return False
    return True


def _join_hooks(features: list[dict[str, Any]], hooks: list[Point]) -> None:
    """Add each street junction as a vertex of the arterial it sits on, so the graph is shared."""
    for hook in hooks:
        for feature in features:
            line = feature["geometry"]
            if feature["road_class"] != "arterial" or line.distance(hook) > 1e-6:
                continue
            at = line.project(hook)
            coords = list(line.coords)
            along = [line.project(Point(c)) for c in coords]
            if any(abs(a - at) < 1e-9 for a in along):
                break
            k = next(i for i, a in enumerate(along) if a > at)
            feature["geometry"] = LineString([*coords[:k], (hook.x, hook.y), *coords[k:]])
            break


def build_roads(
    dem: Any,
    landcover: Any,
    gsd_m: float,
    transform: Affine,
    crs: str,
    seed: int,
    config: RoadConfig | None = None,
    centres: list[tuple[int, int]] | None = None,
    world: str = "roads",
) -> gpd.GeoDataFrame:
    """A ``road`` layer (LineStrings with class, width and a bridge flag) for a terrain grid."""
    config = config or RoadConfig()
    if dem.shape != landcover.shape:
        raise RoadError("dem and landcover must have the same shape")
    slope = slope_degrees(dem, gsd_m)
    water = landcover == WATER
    arterial_cost = _cost(slope, water, True, config)
    local_cost = _cost(slope, water, False, config)
    picked = centres or _choose_centres(slope, water, gsd_m, seed, config)
    occupied = np.zeros(water.shape, dtype=bool)
    features: list[dict[str, Any]] = []
    arterials: list[Any] = []
    smoothed: list[LineString] = []
    inverse = ~transform

    def add(cells: Any, road_class: str, width: float, bridge: float) -> None:
        if len(cells) >= 2:
            coords = _xy(cells, transform)
            if road_class == "arterial" and bridge == 0.0:
                fair = smooth_arterial(coords, gsd_m, config)
                if fair is not coords and _stays_dry(fair, water, inverse, gsd_m / 2):
                    coords = fair
                smoothed.append(LineString(coords))
            features.append(
                {
                    "geometry": LineString(coords),
                    "provenance": PROVENANCE,
                    "road_class": road_class,
                    "width_m": width,
                    "bridge": bridge,
                }
            )

    for i, j in _links(picked, seed):
        path = least_cost_path(arterial_cost, picked[i], picked[j])
        if path is None:
            raise RoadError(f"no route between settlement centres {picked[i]} and {picked[j]}")
        arterials.append(path)
        occupied[path[:, 0], path[:, 1]] = True
        over_water = water[path[:, 0], path[:, 1]]
        start = 0
        for k in range(1, len(path) + 1):
            if k < len(path) and over_water[k] == over_water[start]:
                continue
            if over_water[start]:  # a bridge: the water run plus the bank cell at each end
                add(
                    path[max(start - 1, 0) : min(k + 1, len(path))],
                    "arterial",
                    config.arterial_width_m,
                    1.0,
                )
            else:  # dry road: land cells only, so a bridge joins it at a shared bank cell
                add(path[start:k], "arterial", config.arterial_width_m, 0.0)
            start = k
    rng = substream(seed, "roads:local")
    rows, cols = water.shape
    trunk = unary_union(smoothed) if smoothed else None
    hooks: list[Point] = []
    spacing = max(3, int(config.local_spacing_m / gsd_m))
    lo, hi = (v / gsd_m for v in config.local_length_m)
    for path in arterials:
        i = 6
        while i < len(path) - 6:
            tangent = path[min(i + 3, len(path) - 1)] - path[max(i - 3, 0)]
            norm = math.hypot(*tangent) or 1.0
            normal = np.array([-tangent[1], tangent[0]], dtype=float) / norm
            for sign in (1.0, -1.0):
                keep, length = rng.uniform() < config.local_probability, rng.uniform(lo, hi)
                if not keep:
                    continue
                cells = [tuple(path[i])]
                for step in range(1, int(length) + 1):
                    r = int(round(path[i][0] + sign * normal[0] * step))
                    c = int(round(path[i][1] + sign * normal[1] * step))
                    if not (0 <= r < rows and 0 <= c < cols) or not np.isfinite(local_cost[r, c]):
                        break
                    if (r, c) != cells[-1]:
                        cells.append((r, c))
                    if occupied[r, c] and step > 3:
                        break
                if len(cells) >= 6:
                    for r, c in cells[1:]:
                        occupied[r, c] = True
                    add(np.array(cells), "local", config.local_width_m, 0.0)
                    if trunk is not None:  # start the street on the smoothed arterial itself
                        feature = features[-1]
                        first = Point(feature["geometry"].coords[0])
                        hook = trunk.interpolate(trunk.project(first))
                        tail = list(feature["geometry"].coords)[1:]
                        feature["geometry"] = LineString([(hook.x, hook.y), *tail])
                        hooks.append(hook)
            i += max(1, int(spacing * rng.uniform(0.7, 1.3)))
    _join_hooks(features, hooks)
    return make_layer("road", world, features, crs)
