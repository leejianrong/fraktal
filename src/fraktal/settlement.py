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
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import split, unary_union
from shapely.strtree import STRtree

from fraktal.coast import UNBUILDABLE
from fraktal.ids import substream
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
    road_overlap_share: float = 0.03  # sliver a curved carriageway may clip off a parcel edge
    skew_deg: float = 0.0  # random shear of each parcel's side edges; 0 keeps them square
    blocks: bool = False  # carve street blocks and subdivide them, interior lots included
    block_depth_m: float = 60.0  # blocks reach this far from a street edge, no further


@dataclass(frozen=True)
class BuildingConfig:
    built_probability_core: float = 0.95
    built_probability_edge: float = 0.45
    core_radius_m: float = 350.0
    front_setback_m: float = 3.0
    side_setback_m: float = 1.5
    height_core_m: tuple[float, float] = (7.0, 22.0)
    height_edge_m: tuple[float, float] = (3.0, 8.0)
    width_share: tuple[float, float] = (0.55, 0.85)  # of the usable parcel frontage
    depth_share: tuple[float, float] = (0.45, 0.8)  # of the usable parcel depth
    roof_materials: bool = False  # record tin or tile on each building
    tin_share: float = 0.6  # share of tin (corrugated metal) roofs when materials are recorded
    shape_mix: tuple[float, float, float] = (0.0, 0.0, 0.0)  # share of L, T and shed-attached
    size_tail: float = 0.0  # log-normal spread of the footprint size; 0 keeps the uniform mix
    shed_height_m: tuple[float, float] = (2.4, 3.2)


def _cell_is_water(landcover: Any, inverse: Affine, x: float, y: float) -> bool:
    col, row = inverse @ (x, y)
    r, c = int(math.floor(row)), int(math.floor(col))
    if not (0 <= r < landcover.shape[0] and 0 <= c < landcover.shape[1]):
        return True  # outside the grid counts as unusable
    return bool(landcover[r, c] in UNBUILDABLE)


def _frontage_parcels(
    line: Any, half_width: float, rng: Any, config: ParcelConfig
) -> list[Polygon]:
    out: list[Polygon] = []
    for side in (1.0, -1.0):
        s = 0.0
        while s < line.length:
            width = float(rng.uniform(*config.frontage_m))
            depth = float(rng.uniform(*config.depth_m))
            shear = 0.0
            if config.skew_deg:
                shear = math.tan(math.radians(rng.uniform(-1, 1) * config.skew_deg))
            if s + width > line.length:
                break
            a, b = line.interpolate(s), line.interpolate(s + width)
            tx, ty = b.x - a.x, b.y - a.y
            norm = math.hypot(tx, ty)
            if norm > 0:
                nx, ny = -ty / norm * side, tx / norm * side
                near, far = half_width, half_width + depth
                slide = depth * shear / norm  # far edge slides along the frontage
                out.append(
                    Polygon(
                        [
                            (a.x + nx * near, a.y + ny * near),
                            (b.x + nx * near, b.y + ny * near),
                            (b.x + nx * far + tx * slide, b.y + ny * far + ty * slide),
                            (a.x + nx * far + tx * slide, a.y + ny * far + ty * slide),
                        ]
                    )
                )
            s += width
    return out


def _split_lot(lot: Polygon, rng: Any, config: ParcelConfig, depth: int = 0) -> list[Polygon]:
    """Cut a block piece across its long side until each lot is about one parcel in area."""
    lo = config.frontage_m[0] * config.depth_m[0]
    hi = config.frontage_m[1] * config.depth_m[1]
    target = float(rng.uniform(lo, hi))
    if lot.area <= target or lot.area < 2 * config.min_area_m2 or depth >= 12:
        return [lot]
    (x0, y0), (x1, y1), (x2, y2), _ = list(lot.minimum_rotated_rectangle.exterior.coords)[:4]
    a, b = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
    ux, uy = ((x1 - x0) / a, (y1 - y0) / a) if a >= b else ((x2 - x1) / b, (y2 - y1) / b)
    cx, cy = lot.centroid.x, lot.centroid.y
    t = float(rng.uniform(-0.15, 0.15)) * max(a, b)
    px, py = cx + ux * t, cy + uy * t
    reach = lot.length  # longer than any chord through the lot
    if config.skew_deg:  # slant the cut so neighbouring plots are not parallel
        angle = math.radians(float(rng.uniform(-1, 1)) * config.skew_deg * 0.5)
        c, sn = math.cos(angle), math.sin(angle)
        ux, uy = ux * c - uy * sn, ux * sn + uy * c
    cut = LineString([(px - uy * reach, py + ux * reach), (px + uy * reach, py - ux * reach)])
    pieces = [g for g in split(lot, cut).geoms if isinstance(g, Polygon) and g.area > 1e-6]
    if len(pieces) < 2:
        return [lot]
    pieces.sort(key=lambda g: (round(g.centroid.x, 3), round(g.centroid.y, 3)))
    return [lot2 for g in pieces for lot2 in _split_lot(g, rng, config, depth + 1)]


def _block_parcels(
    roads: gpd.GeoDataFrame, dry: gpd.GeoDataFrame, transform: Affine, shape: tuple[int, ...],
    seed: int, config: ParcelConfig,
) -> list[tuple[Polygon, str, str]]:  # fmt: skip
    """Lots cut from the land between the streets, out to ``block_depth_m`` from a street edge."""
    west, north = transform @ (0, 0)
    east, south = transform @ (shape[1], shape[0])
    region = box(min(west, east), min(south, north), max(west, east), max(south, north))
    carriage = unary_union([r.geometry.buffer(r.width_m / 2) for _, r in roads.iterrows()])
    halo = unary_union(
        [
            r.geometry.buffer(r.width_m / 2 + config.block_depth_m)
            for _, r in dry.sort_values("id").iterrows()
        ]
    )
    band = region.intersection(halo).difference(carriage)
    blocks = [g for g in getattr(band, "geoms", [band]) if isinstance(g, Polygon) and g.area > 1]
    blocks.sort(key=lambda g: (round(g.centroid.x, 3), round(g.centroid.y, 3)))
    ids = list(dry.sort_values("id").id)
    tree = STRtree([r.geometry for _, r in dry.sort_values("id").iterrows()])
    out: list[tuple[Polygon, str, str]] = []
    for block in blocks:
        key = f"{round(block.centroid.x)}:{round(block.centroid.y)}"
        rng = substream(seed, f"blocks:{key}")
        for lot in _split_lot(block, rng, config):
            road_id = ids[int(tree.nearest(lot.centroid))]
            out.append((lot, road_id, "local"))
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
    if config.blocks:
        candidates = _block_parcels(roads, dry_roads, transform, landcover.shape, seed, config)
    else:
        for _, road in dry_roads.sort_values("id").iterrows():
            rng = substream(seed, f"parcels:{road.id}")
            for poly in _frontage_parcels(road.geometry, road.width_m / 2, rng, config):
                candidates.append((poly, road.id, road.road_class))
    polys = [p for p, _, _ in candidates]
    tree = STRtree(polys)
    accepted = [False] * len(candidates)
    for i, (poly, _, _) in enumerate(candidates):
        if poly.area < config.min_area_m2 or not poly.is_valid:
            continue
        if any(
            _cell_is_water(landcover, inverse, x, y)
            for x, y in [*poly.exterior.coords, (poly.centroid.x, poly.centroid.y)]
        ):
            continue
        limit = max(1e-6, config.road_overlap_share * poly.area)
        crosses_road = any(
            poly.intersection(carriageways[j][0]).area > limit for j in road_tree.query(poly)
        )
        if crosses_road:
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
    road_lines: dict[str, Any] | None = None,
    extra: list[dict[str, Any]] | None = None,
) -> gpd.GeoDataFrame:
    """A ``building`` layer with one footprint inside each chosen parcel, facing its road.

    ``extra`` are ready-made features (industrial halls) added to the layer as they are.
    """
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
            rng.uniform(*config.depth_share),
            rng.uniform(*config.width_share),
        )
        lo, hi = config.height_core_m if urban > 0.5 else config.height_edge_m
        height, roof = float(rng.uniform(lo, hi)), ROOF_TYPES[int(rng.integers(0, len(ROOF_TYPES)))]
        tin_roll = float(rng.uniform()) if config.roof_materials else 0.0
        if roll >= chance:
            continue
        shape = _draw_shape(seed, parcel.id, config)
        width_share = float(np.clip(width_share * shape.size, 0.3, 1.0))
        depth_share = float(np.clip(depth_share * shape.size, 0.3, 1.0))
        if road_lines is None:
            parts = _footprint_parts(parcel.geometry, config, width_share, depth_share, shape)
        else:  # block lots are free-form: fit a rectangle facing the street
            parts = _lot_footprint(
                parcel.geometry, road_lines[parcel.road_id], config, width_share, depth_share, shape
            )
        if parts is None:
            continue
        footprint, shed = parts
        feature: dict[str, Any] = {
            "geometry": footprint,
            "provenance": BUILDING_PROVENANCE,
            "height_m": height,
            "roof_type": roof,
            "land_use": "commercial" if urban > 0.5 else "residential",
            "parcel_id": parcel.id,
        }
        if config.roof_materials:
            feature["roof_material"] = "tin" if tin_roll < config.tin_share else "tile"
        features.append(feature)
        if shed is not None:  # a low lean-to against the house, a building of its own
            features.append(
                {**feature, "geometry": shed, "height_m": shape.shed_height, "roof_type": "flat",
                 "land_use": "residential"}
            )  # fmt: skip
    features += extra or []
    if not features:
        raise ValueError("no buildings were placed")
    return make_layer("building", world, features, str(parcels.crs))


@dataclass(frozen=True)
class Shape:
    kind: str = "rect"  # rect, L, T or shed
    size: float = 1.0  # multiplies the width and depth shares
    cut_w: float = 0.4  # share of the width an L or T notch removes
    cut_d: float = 0.4
    flip: bool = False  # which side the L notch or the shed is on
    shed_height: float = 2.8


def _draw_shape(seed: int, parcel_id: str, config: BuildingConfig) -> Shape:
    """The footprint type and size factor of one parcel, from a stream of its own."""
    if config.shape_mix == (0.0, 0.0, 0.0) and config.size_tail <= 0.0:
        return Shape()
    look = substream(seed, f"buildings:shape:{parcel_id}")
    pick = float(look.uniform())
    kind = ("L", "T", "shed", "rect")[
        int(np.searchsorted(np.cumsum(config.shape_mix), pick, side="right"))
    ]
    size = float(np.exp(look.normal(0.0, config.size_tail))) if config.size_tail > 0.0 else 1.0
    return Shape(
        kind, size, float(look.uniform(0.3, 0.5)), float(look.uniform(0.3, 0.5)),
        bool(look.integers(0, 2)), float(look.uniform(*config.shed_height_m)),
    )  # fmt: skip


def _footprint(
    parcel: Polygon, config: BuildingConfig, width_share: float, depth_share: float
) -> Polygon | None:
    """A rectangle inside the parcel, set back from the road and the neighbours."""
    parts = _footprint_parts(parcel, config, width_share, depth_share, Shape())
    return None if parts is None else parts[0]


def _footprint_parts(
    parcel: Polygon, config: BuildingConfig, width_share: float, depth_share: float, shape: Shape
) -> tuple[Polygon, Polygon | None] | None:
    """The house outline in the parcel, and a lean-to shed when the shape has one.

    The outline is a rectangle with an L or T notch cut from its back corners, or a narrower
    rectangle with a low shed on one side, all within the rectangle a plain house would fill.
    """
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

    def poly(points: list[tuple[float, float]]) -> Polygon:
        return Polygon([corner(left + a, near + b) for a, b in points])

    cw, cd = w * shape.cut_w, d * shape.cut_d
    big = w >= 6.0 and d >= 6.0  # a notch or shed needs room to leave two usable parts
    if shape.kind == "L" and big:
        pts = [(0, 0), (w, 0), (w, d - cd), (w - cw, d - cd), (w - cw, d), (0, d)]
        if shape.flip:
            pts = [(w - a, b) for a, b in pts][::-1]
        return poly(pts), None
    if shape.kind == "T" and big:
        notch_w = w * shape.cut_w * 0.7
        pts = [(0, 0), (w, 0), (w, d - cd), (w - notch_w, d - cd), (w - notch_w, d),
               (notch_w, d), (notch_w, d - cd), (0, d - cd)]  # fmt: skip
        return poly(pts), None
    if shape.kind == "shed" and big:
        main_w = w * 0.68
        shed_d = d * 0.55
        x_main, x_shed = (0.0, main_w) if not shape.flip else (w - main_w, 0.0)
        shed_w = w - main_w
        return (
            poly([(x_main, 0), (x_main + main_w, 0), (x_main + main_w, d), (x_main, d)]),
            poly([(x_shed, 0), (x_shed + shed_w, 0), (x_shed + shed_w, shed_d), (x_shed, shed_d)]),
        )
    return poly([(0, 0), (w, 0), (w, d), (0, d)]), None


def _lot_footprint(
    lot: Polygon, street: Any, config: BuildingConfig, width_share: float, depth_share: float,
    shape: Shape | None = None,
) -> tuple[Polygon, Polygon | None] | None:  # fmt: skip
    """A footprint in a free-form lot: its bounding rectangle turned so the street side is first."""
    ring = list(lot.minimum_rotated_rectangle.exterior.coords)[:4]
    near = [street.distance(Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
            for a, b in zip(ring, ring[1:] + ring[:1], strict=True)]  # fmt: skip
    k = int(np.argmin(near))
    rect = Polygon(ring[k:] + ring[:k])
    centre = lot.centroid
    for step in range(0, 13):  # shrink about the centre until the footprint fits the lot
        f = 1.0 - 0.05 * step
        fitted = affinity.scale(rect, f, f, origin=centre)
        parts = _footprint_parts(fitted, config, width_share, depth_share, shape or Shape())
        if parts is not None and all(lot.buffer(1e-6).contains(g) for g in parts if g is not None):
            return parts
    return None


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
