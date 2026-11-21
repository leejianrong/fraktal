"""Planned subdivisions (FRK-188): a rigid street grid or cul-de-sac comb with uniform lots.

The organic road generator grows streets outward from arterials, so it cannot make the grids and
loops of planned housing. This one lays the streets out directly, in a frame rotated by a
seed-drawn angle: horizontal streets spaced two lot depths apart so lots sit back to back, vertical
streets between blocks of a fixed number of lots, and every Nth vertical street a wider collector.
In ``cul_de_sac`` mode the horizontal streets are spurs from a vertical street that end in a round
bulb. Lots are identical rectangles facing their street; houses come from the ordinary building
stage, with narrow size ranges, so they come out near-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from affine import Affine
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.strtree import STRtree

from fraktal.ids import substream
from fraktal.roads import PROVENANCE as ROAD_PROVENANCE
from fraktal.settlement import PARCEL_PROVENANCE, _cell_is_water
from fraktal.world import make_layer

PATTERNS = ("grid", "cul_de_sac")
MIN_RUN_M = 30.0  # a street fragment left between water bodies is dropped


@dataclass(frozen=True)
class SubdivisionConfig:
    enabled: bool = False
    pattern: str = "grid"  # "grid" or "cul_de_sac"
    rotation_deg: float = -1.0  # negative: drawn from the seed
    frontage_m: tuple[float, float] = (17.0, 19.0)  # one value drawn per world: lots are identical
    depth_m: float = 29.5
    block_lots: int = 8  # lots along one side of a block
    street_width_m: float = 10.0
    collector_every: int = 4  # every Nth vertical street is a collector
    collector_width_m: float = 18.0
    bulb_radius_m: float = 11.0  # cul-de-sac turning circle


def _frame(shape: tuple[int, ...], transform: Affine) -> tuple[Any, float, float, float]:
    west, north = transform @ (0, 0)
    east, south = transform @ (shape[1], shape[0])
    region = box(min(west, east), min(south, north), max(west, east), max(south, north))
    cx, cy = region.centroid.x, region.centroid.y
    return (
        region,
        cx,
        cy,
        math.hypot(region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1]),
    )


def _to_world(geom: Any, angle: float, cx: float, cy: float) -> Any:
    return affinity.translate(affinity.rotate(geom, angle, origin=(0, 0)), cx, cy)


def _dry_runs(line: LineString, landcover: Any, transform: Affine) -> list[LineString]:
    """The parts of a line over land, from samples every few metres."""
    inverse = ~transform
    n = max(2, int(line.length / 4.0))
    pts = [line.interpolate(i / n, normalized=True) for i in range(n + 1)]
    dry = [not _cell_is_water(landcover, inverse, p.x, p.y) for p in pts]
    runs: list[LineString] = []
    start = None
    for i, ok in enumerate([*dry, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - 1 > start:
                run = LineString([(p.x, p.y) for p in pts[start:i]])
                if run.length >= MIN_RUN_M:
                    runs.append(run)
            start = None
    return runs


@dataclass(frozen=True)
class _Street:
    line: LineString  # local frame
    width: float
    road_class: str


def _layout(
    config: SubdivisionConfig, extent: float, frontage: float
) -> tuple[list[_Street], list[tuple[float, float, float, int, float, float]], list[Point]]:
    """Streets, lot rows and bulb centres in the local frame.

    A lot row is ``(v, u_start, u_step, count, side_hw, ...)`` flattened as
    ``(v, u0, du, count, street_half_width, depth_sign)`` and expanded by the caller.
    """
    w = config.street_width_m
    sv = 2 * config.depth_m + w
    su = config.block_lots * frontage + w
    half = extent / 2 + max(sv, su)
    n_h, n_v = int(half // sv) + 1, int(half // su) + 1
    vs = [k * sv for k in range(-n_h, n_h + 1)]
    us = [j * su for j in range(-n_v, n_v + 1)]
    wide = {j for j in range(-n_v, n_v + 1) if j % max(1, config.collector_every) == 0}
    hw_u = {j: (config.collector_width_m if j in wide else w) / 2 for j in range(-n_v, n_v + 1)}
    streets = [
        _Street(
            LineString([(j * su, -half), (j * su, half)]),
            hw_u[j] * 2,
            "arterial" if j in wide else "local",
        )
        for j in range(-n_v, n_v + 1)
    ]
    rows: list[tuple[float, float, float, int, float, float]] = []
    bulbs: list[Point] = []
    r = config.bulb_radius_m
    for ki, v in enumerate(vs):
        if config.pattern == "grid":
            streets.append(_Street(LineString([(-half, v), (half, v)]), w, "local"))
        for j in range(-n_v, n_v):
            left, right = us[j] + hw_u[j], us[j + 1] - hw_u[j + 1]
            if config.pattern == "grid":
                n = int((right - left) // frontage)
                start = left + ((right - left) - n * frontage) / 2
                step, count = frontage, n
            else:
                from_left = (ki + j) % 2 == 0
                n = int((right - left - 2 * r - 4.0) // frontage)
                if n < 1:
                    continue
                end = left + n * frontage + r if from_left else right - n * frontage - r
                streets.append(
                    _Street(
                        LineString(
                            [(us[j], v), (end, v)] if from_left else [(us[j + 1], v), (end, v)]
                        ),
                        w,
                        "local",
                    )
                )
                bulbs.append(Point(end, v))
                start, step, count = (left, frontage, n) if from_left else (right, -frontage, n)
            for side in (1.0, -1.0):
                rows.append((v, start, step, count, w / 2, side))
    return streets, rows, bulbs


def build_subdivision(
    landcover: Any,
    transform: Affine,
    crs: str,
    seed: int,
    config: SubdivisionConfig,
    road_world: str = "roads",
    parcel_world: str = "parcels",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """The ``road`` and ``parcel`` layers of a planned subdivision covering the region."""
    if config.pattern not in PATTERNS:
        raise ValueError(f"subdivision pattern must be one of {PATTERNS}, got {config.pattern!r}")
    rng = substream(seed, "subdivision:layout")
    angle = float(rng.uniform(0.0, 90.0)) if config.rotation_deg < 0 else config.rotation_deg
    frontage = float(rng.uniform(*config.frontage_m))
    region, cx, cy, diag = _frame(landcover.shape, transform)
    streets, rows, bulbs = _layout(config, diag, frontage)

    features: list[dict[str, Any]] = []

    def add(line: Any, width: float, road_class: str) -> None:
        features.append(
            {
                "geometry": line,
                "provenance": ROAD_PROVENANCE,
                "road_class": road_class,
                "width_m": width,
                "bridge": 0.0,
            }
        )

    for street in streets:
        placed = _to_world(street.line, angle, cx, cy).intersection(region)
        for part in getattr(placed, "geoms", [placed]):
            if isinstance(part, LineString) and part.length > 1.0:
                for run in _dry_runs(part, landcover, transform):
                    add(run, street.width, street.road_class)
    d = config.bulb_radius_m * 1.8
    for bulb in bulbs:  # a turning circle: two squares, one turned 45 degrees
        centre = _to_world(bulb, angle, cx, cy)
        if not region.buffer(-config.bulb_radius_m).contains(centre) or _cell_is_water(
            landcover, ~transform, centre.x, centre.y
        ):
            continue
        for t, turn in enumerate((0.0, 45.0)):
            line = LineString([(-d / 2, 0), (d / 2, 0)])
            line = affinity.translate(
                affinity.rotate(line, angle + turn, origin=(0, 0)),
                centre.x + 0.01 * t, centre.y,
            )  # fmt: skip
            add(line, d, "local")
    roads = make_layer("road", road_world, features, crs)

    lines = list(roads.geometry)
    ids = list(roads.id)
    widths = list(roads.width_m)
    carriage = [g.buffer(wd / 2, cap_style="flat") for g, wd in zip(lines, widths, strict=True)]
    tree = STRtree(carriage)
    line_tree = STRtree(lines)
    inverse = ~transform
    inside = region.buffer(-0.5)
    seen: set[tuple[float, float]] = set()
    lots: list[dict[str, Any]] = []
    for v, u0, du, count, hw, side in rows:
        for i in range(count):
            a, b = u0 + i * du, u0 + (i + 1) * du
            poly = Polygon(
                [(a, v + side * hw), (b, v + side * hw),
                 (b, v + side * (hw + config.depth_m)), (a, v + side * (hw + config.depth_m))]
            )  # fmt: skip
            lot = _to_world(poly, angle, cx, cy)
            c = lot.centroid
            key = (round(c.x, 2), round(c.y, 2))
            if key in seen or not inside.contains(lot):
                continue
            if any(
                _cell_is_water(landcover, inverse, x, y)
                for x, y in [*lot.exterior.coords, (c.x, c.y)]
            ):
                continue
            if any(lot.intersection(carriage[j]).area > 0.5 for j in tree.query(lot)):
                continue
            front = Point((lot.exterior.coords[0][0] + lot.exterior.coords[1][0]) / 2,
                          (lot.exterior.coords[0][1] + lot.exterior.coords[1][1]) / 2)  # fmt: skip
            near = int(line_tree.nearest(front))
            if lines[near].distance(front) > widths[near] / 2 + 0.6:
                continue  # its street was cut by water
            seen.add(key)
            lots.append(
                {
                    "geometry": lot,
                    "provenance": PARCEL_PROVENANCE,
                    "land_use": "residential",
                    "road_id": ids[near],
                }
            )
    if not lots:
        raise ValueError("no lots fit in this subdivision")
    return roads, make_layer("parcel", parcel_world, lots, crs)


__all__ = ["SubdivisionConfig", "build_subdivision"]
