"""Parcels along streets and the buildings on them (V2).

Parcels are rectangles laid along each road's frontage, with their front edge on the carriageway
edge, so every parcel touches a road. Overlaps, water and region edges are rejected. Buildings
sit inside their parcel with setbacks. Parcel shapes draw from a per-road sub-stream and
buildings from a per-parcel one, so changing the building settings never moves a parcel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from fraktal.ids import substream
from fraktal.landcover_gen import WATER
from fraktal.world import make_layer

PARCEL_PROVENANCE = "procedural:parcels"
BUILDING_PROVENANCE = "procedural:buildings"
ROOF_TYPES = ("flat", "gable", "hip")


@dataclass(frozen=True)
class ParcelConfig:
    frontage_m: tuple[float, float] = (14.0, 30.0)
    depth_m: tuple[float, float] = (22.0, 45.0)
    min_area_m2: float = 150.0
    local_only: bool = False  # parcels on arterials too by default


@dataclass(frozen=True)
class BuildingConfig:
    built_probability_core: float = 0.95
    built_probability_edge: float = 0.45
    core_radius_m: float = 350.0
    front_setback_m: float = 3.0
    side_setback_m: float = 1.5
    height_core_m: tuple[float, float] = (7.0, 22.0)
    height_edge_m: tuple[float, float] = (3.0, 8.0)


def _cell_is_water(landcover: Any, inverse: Affine, x: float, y: float) -> bool:
    col, row = inverse @ (x, y)
    r, c = int(math.floor(row)), int(math.floor(col))
    if not (0 <= r < landcover.shape[0] and 0 <= c < landcover.shape[1]):
        return True  # outside the grid counts as unusable
    return bool(landcover[r, c] == WATER)


def _frontage_parcels(
    line: Any, half_width: float, rng: Any, config: ParcelConfig
) -> list[Polygon]:
    out: list[Polygon] = []
    for side in (1.0, -1.0):
        s = 0.0
        while s < line.length:
            width = float(rng.uniform(*config.frontage_m))
            depth = float(rng.uniform(*config.depth_m))
            if s + width > line.length:
                break
            a, b = line.interpolate(s), line.interpolate(s + width)
            tx, ty = b.x - a.x, b.y - a.y
            norm = math.hypot(tx, ty)
            if norm > 0:
                nx, ny = -ty / norm * side, tx / norm * side
                near, far = half_width, half_width + depth
                out.append(
                    Polygon(
                        [
                            (a.x + nx * near, a.y + ny * near),
                            (b.x + nx * near, b.y + ny * near),
                            (b.x + nx * far, b.y + ny * far),
                            (a.x + nx * far, a.y + ny * far),
                        ]
                    )
                )
            s += width
    return out


def build_parcels(
    roads: gpd.GeoDataFrame,
    landcover: Any,
    transform: Affine,
    seed: int,
    config: ParcelConfig | None = None,
    world: str = "parcels",
) -> gpd.GeoDataFrame:
    """A ``parcel`` layer: non-overlapping rectangles fronting the roads, off water and roads."""
    config = config or ParcelConfig()
    inverse = ~transform
    dry_roads = roads[roads["bridge"] == 0.0] if "bridge" in roads else roads
    if config.local_only:
        dry_roads = dry_roads[dry_roads["road_class"] == "local"]
    carriageways = [
        (row.geometry.buffer(row.width_m / 2), row.id)
        for _, row in roads.sort_values("id").iterrows()
    ]
    road_tree = STRtree([g for g, _ in carriageways])
    candidates: list[tuple[Polygon, str, str]] = []
    for _, road in dry_roads.sort_values("id").iterrows():
        rng = substream(seed, f"parcels:{road.id}")
        for poly in _frontage_parcels(road.geometry, road.width_m / 2, rng, config):
            candidates.append((poly, road.id, road.road_class))
    polys = [p for p, _, _ in candidates]
    tree = STRtree(polys)
    accepted = [False] * len(candidates)
    for i, (poly, road_id, _) in enumerate(candidates):
        if poly.area < config.min_area_m2 or not poly.is_valid:
            continue
        if any(
            _cell_is_water(landcover, inverse, x, y)
            for x, y in [*poly.exterior.coords, (poly.centroid.x, poly.centroid.y)]
        ):
            continue
        crosses_road = any(
            poly.intersection(carriageways[j][0]).area > 1e-6 and carriageways[j][1] != road_id
            for j in road_tree.query(poly)
        )
        own = next(g for g, rid in carriageways if rid == road_id)
        if crosses_road or poly.intersection(own).area > 1e-6:
            continue
        if any(
            accepted[j] and j != i and poly.intersection(polys[j]).area > 1e-6
            for j in tree.query(poly)
        ):
            continue
        accepted[i] = True
    features = [
        {"geometry": p, "provenance": PARCEL_PROVENANCE, "land_use": "residential", "road_id": rid}
        for ok, (p, rid, _) in zip(accepted, candidates, strict=True)
        if ok
    ]
    if not features:
        raise ValueError("no parcels fit on this road network")
    return make_layer("parcel", world, features, str(roads.crs))


def build_buildings(
    parcels: gpd.GeoDataFrame,
    centres_xy: list[tuple[float, float]],
    seed: int,
    config: BuildingConfig | None = None,
    world: str = "buildings",
) -> gpd.GeoDataFrame:
    """A ``building`` layer with one footprint inside each chosen parcel, facing its road."""
    config = config or BuildingConfig()
    features: list[dict[str, Any]] = []
    for _, parcel in parcels.sort_values("id").iterrows():
        rng = substream(seed, f"buildings:{parcel.id}")
        centroid = parcel.geometry.centroid
        d = (
            min(math.hypot(centroid.x - x, centroid.y - y) for x, y in centres_xy)
            if centres_xy
            else 0.0
        )
        urban = math.exp(-d / config.core_radius_m)
        chance = (
            config.built_probability_edge
            + (config.built_probability_core - config.built_probability_edge) * urban
        )
        roll, depth_share, width_share = (
            rng.uniform(),
            rng.uniform(0.45, 0.8),
            rng.uniform(0.55, 0.85),
        )
        lo, hi = config.height_core_m if urban > 0.5 else config.height_edge_m
        height, roof = float(rng.uniform(lo, hi)), ROOF_TYPES[int(rng.integers(0, len(ROOF_TYPES)))]
        if roll >= chance:
            continue
        footprint = _footprint(parcel.geometry, config, width_share, depth_share)
        if footprint is None:
            continue
        features.append(
            {
                "geometry": footprint,
                "provenance": BUILDING_PROVENANCE,
                "height_m": height,
                "roof_type": roof,
                "land_use": "commercial" if urban > 0.5 else "residential",
                "parcel_id": parcel.id,
            }
        )
    if not features:
        raise ValueError("no buildings were placed")
    return make_layer("building", world, features, str(parcels.crs))


def _footprint(
    parcel: Polygon, config: BuildingConfig, width_share: float, depth_share: float
) -> Polygon | None:
    """A rectangle inside the parcel, set back from the road and the neighbours."""
    (x0, y0), (x1, y1), (_x2, _y2), (x3, y3) = list(parcel.exterior.coords)[:4]
    frontage, depth = math.hypot(x1 - x0, y1 - y0), math.hypot(x3 - x0, y3 - y0)
    usable_w = frontage - 2 * config.side_setback_m
    usable_d = depth - config.front_setback_m
    if usable_w < 4.0 or usable_d < 4.0:
        return None
    w, d = usable_w * width_share, usable_d * depth_share
    ux, uy = (x1 - x0) / frontage, (y1 - y0) / frontage
    vx, vy = (x3 - x0) / depth, (y3 - y0) / depth
    left = config.side_setback_m + (usable_w - w) / 2
    near = config.front_setback_m

    def corner(a: float, b: float) -> tuple[float, float]:
        return (x0 + ux * a + vx * b, y0 + uy * a + vy * b)

    return Polygon(
        [
            corner(left, near),
            corner(left + w, near),
            corner(left + w, near + d),
            corner(left, near + d),
        ]
    )


def arterial_centres(roads: gpd.GeoDataFrame) -> list[tuple[float, float]]:
    """Where the arterials start and end: the settlement centres the road generator linked."""
    ends: set[tuple[float, float]] = set()
    for _, road in roads[roads["road_class"] == "arterial"].iterrows():
        coords = list(road.geometry.coords)
        ends.update(
            {
                (round(coords[0][0], 3), round(coords[0][1], 3)),
                (round(coords[-1][0], 3), round(coords[-1][1], 3)),
            }
        )
    return sorted(ends)


__all__ = [
    "BuildingConfig",
    "ParcelConfig",
    "arterial_centres",
    "build_buildings",
    "build_parcels",
    "np",
]
