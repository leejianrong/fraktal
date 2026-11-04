"""Fields (V4): plantation blocks with rows and access tracks, as a ``field`` layer.

A field is a rotated rectangle placed off water, roads, parcels, buildings and other fields.
Rows run parallel to one side of the rectangle (``row_angle_deg``), so they always align to the
boundary. Trees are not stored: ``row_points`` derives them from the field when it is rendered.
Placement draws from the sub-stream ``fields:<type>`` only, so nothing else in the world moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
from affine import Affine
from shapely import affinity
from shapely.geometry import LineString, Polygon, box

from fraktal.ids import substream
from fraktal.landcover_gen import WATER
from fraktal.world import make_layer

FIELD_PROVENANCE = "procedural:fields"
OIL_PALM = "oil_palm"


@dataclass(frozen=True)
class FieldConfig:
    palm_count: int = 0  # oil palm blocks to try to place; 0 means the world has no fields
    palm_side_m: tuple[float, float] = (180.0, 420.0)
    palm_spacing_m: float = 8.5  # between palms along a row; rows are 0.87 of this apart
    track_width_m: float = 4.0
    clearance_m: float = 8.0  # kept free around roads, buildings and other fields


def _blocked(local: gpd.GeoDataFrame | None, pad: float) -> list[Any]:
    return [] if local is None else [g.buffer(pad) for g in local.geometry]


def _mostly_dry(poly: Polygon, landcover: Any, transform: Affine, step_m: float) -> bool:
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, step_m), np.arange(miny, maxy, step_m))
    inside = shapely.contains_xy(poly, xs.ravel(), ys.ravel())
    cols, rows = (~transform) @ (xs.ravel()[inside], ys.ravel()[inside])
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    h, w = landcover.shape
    if len(r) == 0 or ((r < 0) | (r >= h) | (c < 0) | (c >= w)).any():
        return False
    return not bool((landcover[r, c] == WATER).any())


def build_fields(
    landcover: Any,
    transform: Affine,
    crs: str,
    seed: int,
    config: FieldConfig,
    obstacles: list[gpd.GeoDataFrame | None],
    world: str = "fields",
) -> gpd.GeoDataFrame | None:
    """A ``field`` layer of oil palm blocks, or None when the config asks for none or none fit."""
    if config.palm_count <= 0:
        return None
    height, width = landcover.shape
    x0, y0 = transform @ (0, height)
    x1, y1 = transform @ (width, 0)
    inner = box(x0, y0, x1, y1).buffer(-config.clearance_m)
    taken = [g for frame in obstacles for g in _blocked(frame, config.clearance_m)]
    rng = substream(seed, f"fields:{OIL_PALM}")
    features: list[dict[str, Any]] = []
    placed: list[Polygon] = []
    for _ in range(40 * config.palm_count):
        if len(placed) == config.palm_count:
            break
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        w, h = (float(rng.uniform(*config.palm_side_m)) for _ in range(2))
        angle = float(rng.uniform(0.0, 180.0))
        poly = affinity.rotate(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), angle)
        pad = poly.buffer(config.clearance_m)
        if not inner.contains(poly):
            continue
        if any(pad.intersects(g) for g in [*taken, *placed]):
            continue
        if not _mostly_dry(poly, landcover, transform, 20.0):
            continue
        placed.append(poly)
        features.append(
            {
                "geometry": poly,
                "provenance": FIELD_PROVENANCE,
                "field_type": OIL_PALM,
                "row_angle_deg": angle % 180.0,
                "row_spacing_m": config.palm_spacing_m,
                "state": "mature",
            }
        )
    if not features:
        return None
    return make_layer("field", world, features, crs)


def row_points(
    poly: Polygon, angle_deg: float, spacing_m: float, leave_track_row: bool = True
) -> Any:
    """Tree positions (N, 2) in a staggered grid whose rows run along ``angle_deg``.

    Rows are ``0.87 * spacing_m`` apart (triangular planting) and trees ``spacing_m`` apart along
    a row, alternate rows shifted by half. Only points inside the polygon, inset by half a
    spacing, are kept. The centre row is left empty for the access track (``access_tracks``).
    """
    cx, cy = poly.centroid.x, poly.centroid.y
    flat = affinity.rotate(poly, -angle_deg, origin=(cx, cy))
    minx, miny, maxx, maxy = flat.buffer(-spacing_m / 2).bounds
    row_gap = 0.87 * spacing_m
    ys = np.arange(math.ceil((miny - cy) / row_gap), math.floor((maxy - cy) / row_gap) + 1)
    pts = []
    for j in ys:
        if leave_track_row and int(j) == 0:
            continue
        shift = 0.5 * spacing_m * (int(j) % 2)
        xs = np.arange(
            math.ceil((minx - cx - shift) / spacing_m),
            1 + math.floor((maxx - cx - shift) / spacing_m),
        )
        for i in xs:
            pts.append((cx + i * spacing_m + shift, cy + j * row_gap))
    if not pts:
        return np.zeros((0, 2))
    arr = np.array(pts)
    arr = arr[shapely.contains_xy(flat.buffer(-spacing_m / 2), arr[:, 0], arr[:, 1])]
    theta = math.radians(angle_deg)
    dx, dy = arr[:, 0] - cx, arr[:, 1] - cy
    cos, sin = math.cos(theta), math.sin(theta)
    out = np.column_stack([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos])
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def access_tracks(fields: gpd.GeoDataFrame, width_m: float) -> list[LineString]:
    """One track along each field's empty centre row, parallel to the rows."""
    tracks: list[LineString] = []
    for _, field in fields.sort_values("id").iterrows():
        poly = field.geometry
        c = poly.centroid
        theta = math.radians(field.row_angle_deg)
        reach = math.hypot(*(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]))
        line = LineString(
            [
                (c.x - math.cos(theta) * reach, c.y - math.sin(theta) * reach),
                (c.x + math.cos(theta) * reach, c.y + math.sin(theta) * reach),
            ]
        ).intersection(poly)
        if isinstance(line, LineString) and line.length > 10 * width_m:
            tracks.append(line)
    return tracks


__all__ = [
    "FIELD_PROVENANCE",
    "OIL_PALM",
    "FieldConfig",
    "access_tracks",
    "build_fields",
    "row_points",
]
