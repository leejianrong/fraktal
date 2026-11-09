"""Industrial sites: a walled-off lot beside an arterial road with one or two long halls (FRK-107).

Sites are carved out of the residential parcels: a rectangle with its front on the carriageway
edge replaces every parcel it touches, so a hall never overlaps a house. A site's yard (the
loading apron in front of the halls) is filled with containers, tanks and silos by
``props_scene.yard_props``. Randomness only through ``substream`` (ADR-0003); each road draws
from its own stream, so adding a road never moves another road's sites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from affine import Affine
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from fraktal.ids import feature_id, substream
from fraktal.settlement import BUILDING_PROVENANCE, PARCEL_PROVENANCE, _cell_is_water
from fraktal.world import make_layer

LAND_USE = "industrial"


@dataclass(frozen=True)
class IndustrialConfig:
    sites_per_km: float = 0.0  # expected sites per km of arterial road, both sides; 0 = none
    frontage_m: tuple[float, float] = (55.0, 100.0)
    depth_m: tuple[float, float] = (38.0, 60.0)
    yard_share: tuple[float, float] = (0.3, 0.45)  # of the depth, the open apron by the road
    height_m: tuple[float, float] = (6.5, 11.0)
    second_hall_share: float = 0.35  # of sites wide enough for two halls
    road_overlap_share: float = 0.03


def _site(line: Any, half_width: float, s: float, width: float, depth: float, side: float) -> Any:
    a, b = line.interpolate(s), line.interpolate(s + width)
    tx, ty = b.x - a.x, b.y - a.y
    norm = math.hypot(tx, ty)
    if norm == 0.0:
        return None
    ux, uy, nx, ny = tx / norm, ty / norm, -ty / norm * side, tx / norm * side
    return a.x, a.y, ux, uy, nx, ny, norm, half_width, depth


def _rect(frame: tuple[float, ...], x0: float, x1: float, y0: float, y1: float) -> Polygon:
    """A rectangle in the site's own frame: ``x`` along the road, ``y`` away from its edge."""
    ax, ay, ux, uy, nx, ny, _norm, near, _depth = frame
    return Polygon(
        [(ax + ux * x + nx * (near + y), ay + uy * x + ny * (near + y))
         for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    )  # fmt: skip


def carve(
    parcels: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    landcover: Any,
    transform: Affine,
    seed: int,
    config: IndustrialConfig,
    world: str = "parcels",
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
    """The parcels with industrial sites carved in, and the halls to add to the buildings."""
    if config.sites_per_km <= 0.0:
        return parcels, []
    inverse = ~transform
    carriageways = [
        (row.geometry.buffer(row.width_m / 2), row.id)
        for _, row in roads.sort_values("id").iterrows()
    ]
    road_tree = STRtree([g for g, _ in carriageways])
    sites: list[tuple[Polygon, str, tuple[float, ...], float]] = []
    arterials = roads[roads["road_class"] == "arterial"] if "road_class" in roads else roads
    for _, road in arterials.sort_values("id").iterrows():
        if "bridge" in road and road["bridge"] > 0.0:
            continue
        rng = substream(seed, f"industrial:{road.id}")
        count = int(rng.poisson(config.sites_per_km * road.geometry.length / 1000.0))
        for _ in range(count):
            width, depth = rng.uniform(*config.frontage_m), rng.uniform(*config.depth_m)
            side = 1.0 if rng.uniform() < 0.5 else -1.0
            yard = float(rng.uniform(*config.yard_share))
            s = float(rng.uniform(0.0, max(road.geometry.length - width, 0.0)))
            if s + width > road.geometry.length:
                continue
            frame = _site(road.geometry, road.width_m / 2, s, width, depth, side)
            if frame is None:
                continue
            poly = _rect(frame, 0.0, frame[6], 0.0, depth)
            probes = [*poly.exterior.coords, (poly.centroid.x, poly.centroid.y)]
            if any(_cell_is_water(landcover, inverse, x, y) for x, y in probes):
                continue
            limit = max(1e-6, config.road_overlap_share * poly.area)
            hits = road_tree.query(poly)
            if any(poly.intersection(carriageways[j][0]).area > limit for j in hits):
                continue
            if any(poly.intersection(other).area > 1e-6 for other, _, _, _ in sites):
                continue
            sites.append((poly, str(road.id), frame, yard))
    if not sites:
        return parcels, []
    tree = STRtree([p for p, _, _, _ in sites])
    kept = parcels[
        [
            not any(g.intersection(sites[j][0]).area > 1e-6 for j in tree.query(g))
            for g in parcels.geometry
        ]
    ]
    features = [
        {"geometry": g, "provenance": PARCEL_PROVENANCE, "land_use": u, "road_id": r}
        for g, u, r in zip(kept.geometry, kept["land_use"], kept["road_id"], strict=True)
    ]
    halls: list[dict[str, Any]] = []
    for poly, road_id, frame, yard in sites:
        features.append(
            {
                "geometry": poly,
                "provenance": PARCEL_PROVENANCE,
                "land_use": LAND_USE,
                "road_id": road_id,
            }  # fmt: skip
        )
        c = poly.centroid
        parcel_id = feature_id(world, "parcel", f"{c.x:.3f}", f"{c.y:.3f}")
        halls += _halls(frame, yard, parcel_id, seed, config)
    return make_layer("parcel", world, features, str(parcels.crs)), halls


def _halls(
    frame: tuple[float, ...], yard: float, parcel_id: str, seed: int, config: IndustrialConfig
) -> list[dict[str, Any]]:
    """One or two long halls at the back of the site, their long side parallel to the road."""
    rng = substream(seed, f"industrial:halls:{parcel_id}")
    width, depth = frame[6], frame[8]
    margin, gap = 4.0, 7.0
    y0, y1 = depth * yard, depth - 3.0
    two = width >= 80.0 and rng.uniform() < config.second_hall_share
    spans = [(margin, width - margin)]
    if two:
        cut = float(rng.uniform(0.45, 0.6)) * width
        spans = [(margin, cut - gap / 2), (cut + gap / 2, width - margin)]
    out = []
    for x0, x1 in spans:
        back = y1 - float(rng.uniform(0.0, 4.0))
        front = max(y0, back - float(rng.uniform(14.0, 22.0)))
        if x1 - x0 < 18.0 or back - front < 10.0:
            continue
        out.append(
            {
                "geometry": _rect(frame, x0, x1, front, back),
                "provenance": BUILDING_PROVENANCE,
                "height_m": float(rng.uniform(*config.height_m)),
                "roof_type": "gable",
                "land_use": LAND_USE,
                "parcel_id": parcel_id,
            }
        )
    return out


__all__ = ["IndustrialConfig", "LAND_USE", "carve"]
