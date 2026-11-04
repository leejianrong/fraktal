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

from fraktal.coast import UNBUILDABLE
from fraktal.ids import substream
from fraktal.landcover_gen import slope_degrees
from fraktal.world import make_layer

FIELD_PROVENANCE = "procedural:fields"
OIL_PALM = "oil_palm"
PADDY = "paddy"
TERRACE = "terrace"
FISHPOND = "fishpond"
PADDY_STATES = ("flooded", "planted", "harvested")
PADDY_STATE_SHARES = (0.35, 0.4, 0.25)


@dataclass(frozen=True)
class FieldConfig:
    palm_count: int = 0  # oil palm blocks to try to place; 0 means the world has no fields
    palm_side_m: tuple[float, float] = (180.0, 420.0)
    palm_spacing_m: float = 8.5  # between palms along a row; rows are 0.87 of this apart
    track_width_m: float = 4.0
    clearance_m: float = 8.0  # kept free around roads, buildings and other fields
    paddy_zones: int = 0  # areas of flat land tiled with bunded paddy blocks
    paddy_zone_radius_m: float = 350.0
    paddy_cell_m: tuple[float, float] = (35.0, 90.0)  # block side lengths on the shared grid
    paddy_max_slope_deg: float = 6.0
    terrace_count: int = 0  # stepped blocks on slopes
    terrace_side_m: tuple[float, float] = (60.0, 140.0)
    terrace_slope_deg: tuple[float, float] = (3.5, 30.0)
    terrace_step_m: float = 3.0  # height of one terrace step
    pond_zones: int = 0  # areas of the coastal zone tiled with fish ponds
    pond_zone_radius_m: float = 450.0
    pond_cell_m: tuple[float, float] = (45.0, 100.0)


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
    return not bool(np.isin(landcover[r, c], UNBUILDABLE).any())


def _flat_enough(poly: Polygon, slope: Any, transform: Affine, lo: float, hi: float) -> bool:
    """True if at least 85% of the sampled cells inside ``poly`` slope within [lo, hi] degrees."""
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, 10.0), np.arange(miny, maxy, 10.0))
    inside = shapely.contains_xy(poly, xs.ravel(), ys.ravel())
    cols, rows = (~transform) @ (xs.ravel()[inside], ys.ravel()[inside])
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    h, w = slope.shape
    if len(r) == 0 or ((r < 0) | (r >= h) | (c < 0) | (c >= w)).any():
        return False
    s = slope[r, c]
    return bool(((s >= lo) & (s <= hi)).mean() >= 0.85)


def build_fields(
    landcover: Any,
    transform: Affine,
    crs: str,
    seed: int,
    config: FieldConfig,
    obstacles: list[gpd.GeoDataFrame | None],
    world: str = "fields",
    dem: Any = None,
    pond_zone: Any = None,
) -> gpd.GeoDataFrame | None:
    """A ``field`` layer (oil palm, paddy, terraces), or None when the config asks for none."""
    wanted = config.palm_count + config.paddy_zones + config.terrace_count + config.pond_zones
    if wanted <= 0:
        return None
    height, width = landcover.shape
    x0, y0 = transform @ (0, height)
    x1, y1 = transform @ (width, 0)
    inner = box(x0, y0, x1, y1).buffer(-config.clearance_m)
    taken = [g for frame in obstacles for g in _blocked(frame, config.clearance_m)]
    placed: list[Polygon] = []
    features: list[dict[str, Any]] = []
    slope = slope_degrees(dem, transform.a) if dem is not None else None

    def free(poly: Polygon, pad: float) -> bool:
        grown = poly.buffer(pad)
        return inner.contains(poly) and not any(grown.intersects(g) for g in [*taken, *placed])

    rng = substream(seed, f"fields:{OIL_PALM}")
    for _ in range(40 * config.palm_count):
        if sum(f["field_type"] == OIL_PALM for f in features) == config.palm_count:
            break
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        w, h = (float(rng.uniform(*config.palm_side_m)) for _ in range(2))
        angle = float(rng.uniform(0.0, 180.0))
        poly = affinity.rotate(box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), angle)
        if not free(poly, config.clearance_m) or not _mostly_dry(poly, landcover, transform, 20.0):
            continue
        placed.append(poly)
        features.append(_feature(poly, OIL_PALM, angle % 180.0, config.palm_spacing_m, "mature"))
    if slope is not None:  # terraces claim the steeper ground first, paddies take the rest
        box_xy = (x0, y0, x1, y1)
        _terraces(features, placed, free, slope, transform, seed, config, box_xy)
        _paddies(features, placed, free, slope, landcover, transform, seed, config, box_xy)
    _ponds(features, placed, free, pond_zone, transform, seed, config, (x0, y0, x1, y1))
    if not features:
        return None
    return make_layer("field", world, features, crs)


def _feature(poly: Polygon, kind: str, angle: float, spacing: float, state: str) -> dict[str, Any]:
    return {
        "geometry": poly,
        "provenance": FIELD_PROVENANCE,
        "field_type": kind,
        "row_angle_deg": angle,
        "row_spacing_m": spacing,
        "state": state,
    }


def _grid_cells(
    rng: Any,
    bounds: tuple[float, float, float, float],
    cell_m: tuple[float, float],
    zones: int,
    radius_m: float,
) -> tuple[list[Polygon], float]:
    """Cells of one rotated grid (touching edge to edge) whose centres lie near a zone centre."""
    x0, y0, x1, y1 = bounds
    angle = float(rng.uniform(0.0, 90.0))
    centres = np.column_stack([rng.uniform(x0, x1, zones), rng.uniform(y0, y1, zones)])
    reach = math.hypot(x1 - x0, y1 - y0)
    n = int(reach / cell_m[0]) + 2
    cols, rows = np.cumsum(rng.uniform(*cell_m, n)), np.cumsum(rng.uniform(*cell_m, n))
    cells: list[Polygon] = []
    for i in range(n - 1):
        for j in range(n - 1):
            cell = box(
                cols[i] - reach / 2,
                rows[j] - reach / 2,
                cols[i + 1] - reach / 2,
                rows[j + 1] - reach / 2,
            )
            cell = affinity.translate(
                affinity.rotate(cell, angle, origin=(0, 0)), (x0 + x1) / 2, (y0 + y1) / 2
            )
            near = np.hypot(centres[:, 0] - cell.centroid.x, centres[:, 1] - cell.centroid.y)
            if (near < radius_m).any():
                cells.append(cell)
    return cells, angle


def _paddies(
    features: list[dict[str, Any]],
    placed: list[Polygon],
    free: Any,
    slope: Any,
    landcover: Any,
    transform: Affine,
    seed: int,
    config: FieldConfig,
    bounds: tuple[float, float, float, float],
) -> None:
    """Tile flat dry land near a few zone centres with bunded blocks on one rotated grid."""
    if config.paddy_zones <= 0:
        return
    rng = substream(seed, f"fields:{PADDY}")
    cells, angle = _grid_cells(
        rng, bounds, config.paddy_cell_m, config.paddy_zones, config.paddy_zone_radius_m
    )
    states = rng.choice(PADDY_STATES, size=len(cells), p=PADDY_STATE_SHARES)
    for cell, state in zip(cells, states, strict=True):
        if not free(cell, -0.05) or not _mostly_dry(cell, landcover, transform, 10.0):
            continue
        if not _flat_enough(cell, slope, transform, 0.0, config.paddy_max_slope_deg):
            continue
        placed.append(cell)
        features.append(_feature(cell, PADDY, angle, 0.0, str(state)))


def _ponds(
    features: list[dict[str, Any]],
    placed: list[Polygon],
    free: Any,
    zone: Any,
    transform: Affine,
    seed: int,
    config: FieldConfig,
    bounds: tuple[float, float, float, float],
) -> None:
    """Fish ponds: grid blocks that sit wholly inside the coastal ``zone`` raster."""
    if config.pond_zones <= 0 or zone is None:
        return
    rng = substream(seed, f"fields:{FISHPOND}")
    cells, angle = _grid_cells(
        rng, bounds, config.pond_cell_m, config.pond_zones, config.pond_zone_radius_m
    )
    for cell in cells:
        if free(cell, -0.05) and _inside_zone(cell, zone, transform):
            placed.append(cell)
            features.append(_feature(cell, FISHPOND, angle, 0.0, "stocked"))


def _inside_zone(poly: Polygon, zone: Any, transform: Affine) -> bool:
    step = abs(transform.a)  # test every cell the block covers, not a coarser sample
    minx, miny, maxx, maxy = poly.bounds
    xs, ys = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    inside = shapely.contains_xy(poly, xs.ravel(), ys.ravel())
    cols, rows = (~transform) @ (xs.ravel()[inside], ys.ravel()[inside])
    r, c = np.floor(rows).astype(int), np.floor(cols).astype(int)
    h, w = zone.shape
    if len(r) == 0 or ((r < 0) | (r >= h) | (c < 0) | (c >= w)).any():
        return False
    return bool(zone[r, c].all())


def _terraces(
    features: list[dict[str, Any]],
    placed: list[Polygon],
    free: Any,
    slope: Any,
    transform: Affine,
    seed: int,
    config: FieldConfig,
    bounds: tuple[float, float, float, float],
) -> None:
    """Blocks on sloping ground whose surface is stepped along the contours when rendered."""
    x0, y0, x1, y1 = bounds
    rng = substream(seed, f"fields:{TERRACE}")
    for _ in range(200 * config.terrace_count):
        if sum(f["field_type"] == TERRACE for f in features) == config.terrace_count:
            break
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        w, h = (float(rng.uniform(*config.terrace_side_m)) for _ in range(2))
        poly = affinity.rotate(
            box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), float(rng.uniform(0, 180))
        )
        if free(poly, config.clearance_m) and _flat_enough(
            poly, slope, transform, *config.terrace_slope_deg
        ):
            placed.append(poly)
            features.append(_feature(poly, TERRACE, 0.0, config.terrace_step_m, "planted"))


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


def level_paddies(dem: Any, transform: Affine, fields: gpd.GeoDataFrame) -> Any:
    """``dem`` with each paddy block levelled to its mean height, as farmers do before flooding."""
    from rasterio.features import rasterize

    out = np.array(dem, copy=True)
    for _, f in fields[fields["field_type"] == PADDY].sort_values("id").iterrows():
        inside = rasterize([(f.geometry, 1)], dem.shape, transform=transform, dtype="uint8") > 0
        if inside.any():
            out[inside] = float(out[inside].mean())
    return out


def access_tracks(fields: gpd.GeoDataFrame, width_m: float) -> list[LineString]:
    """One track along each field's empty centre row, parallel to the rows."""
    tracks: list[LineString] = []
    for _, field in fields[fields["field_type"] == OIL_PALM].sort_values("id").iterrows():
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
    "FISHPOND",
    "OIL_PALM",
    "PADDY",
    "TERRACE",
    "FieldConfig",
    "access_tracks",
    "level_paddies",
    "build_fields",
    "row_points",
]
