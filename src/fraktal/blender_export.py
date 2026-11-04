"""Export one tile of a procedural world for the Blender experiment (ADR-0010, Stage 0).

Writes ``scene.json`` (building footprints and heights, in metres from the tile's south-west
corner, east and north positive) and ``ground.npy`` (an H x W x 3 float image, north up, of the
ground colours with roads painted, the same palette the 2.5D renderer uses). The Blender side
(``scripts/blender/stage0.py``) reads only these two files, so this is the renderer file contract
in miniature: the scene is data, not objects in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from fraktal.objects import place
from fraktal.procedural import read_buildings, read_roads
from fraktal.render import (
    DEFAULT_HEIGHT_M,
    LANDCOVER_RGB,
    TEXTURE_CLASS_AMP,
    RenderConfig,
    _building_rasters,
    _road_raster,
    _warp,
    region_grid,
    texture_field,
)

SCHEMA = 1


def export_scene(
    world_dir: Path,
    out_dir: Path,
    center_lonlat: tuple[float, float],
    size_m: float,
    gsd_m: float,
    sun: tuple[float, float] = (135.0, 50.0),
    seed: int = 0,
    texture: float = 0.7,
    tree_density: float = 1.0,
) -> dict[str, Any]:
    """Write the tile around ``center_lonlat`` and return the scene description.

    The ground carries the 2.5D renderer's fractal texture (``texture`` is its strength, 0 for flat
    colour) and ``trees`` are the 2.5D renderer's scattered trees, same seed and same places.
    """
    px = max(1, round(size_m / gsd_m))
    config = RenderConfig(gsd_m=gsd_m, tile_px=px, size_m=size_m)
    epsg, transform, (_, side) = region_grid(center_lonlat, config)
    west, north = transform.c, transform.f
    south = north - side * gsd_m
    local = read_buildings(world_dir).to_crs(epsg)
    roads = read_roads(world_dir).to_crs(epsg)
    landcover = _warp(world_dir / "landcover.tif", transform, epsg, side, Resampling.nearest)
    _, mask, _ = _building_rasters(local, transform, side)
    surface = landcover
    road = _road_raster(roads, transform, side) if len(roads) else None
    if road is not None:
        road[mask > 0] = 0
        surface = np.where(road > 0, road, landcover).astype("uint8")
    ground = np.zeros((side, side, 3), dtype="float32")
    for code in np.unique(surface):
        ground[surface == code] = np.array(LANDCOVER_RGB.get(int(code), (120, 118, 116))) / 255.0
    if texture > 0:
        amp = np.ones(256, dtype="float32")
        for code, value in TEXTURE_CLASS_AMP.items():
            amp[code] = value
        modulation = texture * texture_field(seed, (side, side), gsd_m) * amp[surface]
        ground = np.clip(ground * (1.0 + modulation)[..., None], 0.0, 1.0).astype("float32")
    trees = []
    if tree_density > 0:
        shapes, heights, _ = place(surface, mask, gsd_m, tree_density, seed)
        for (row, col, radius, *_rest, is_car), height in zip(shapes, heights, strict=True):
            if is_car == 0:  # trees come first and have no car flag
                trees.append(
                    [
                        round((col + 0.5) * gsd_m, 2),
                        round((side - row - 0.5) * gsd_m, 2),
                        round(radius * gsd_m, 2),
                        round(float(height), 2),
                    ]
                )
    buildings = []
    for row in local.itertuples():
        if row.geometry.is_empty or not row.geometry.intersects(
            _box(west, south, west + side * gsd_m, north)
        ):
            continue
        height = DEFAULT_HEIGHT_M if np.isnan(getattr(row, "height_m", np.nan)) else row.height_m
        material = getattr(row, "roof_material", None)
        buildings.append(
            {
                "id": row.id,
                "ring": [
                    [round(x - west, 3), round(y - south, 3)]
                    for x, y in row.geometry.exterior.coords
                ],
                "height_m": float(height),
                "roof_material": None
                if material is None or material != material
                else str(material),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ground.npy", ground)
    scene = {
        "schema": SCHEMA,
        "size_m": side * gsd_m,
        "gsd_m": gsd_m,
        "px": side,
        "sun": {"azimuth_deg": sun[0], "elevation_deg": sun[1]},
        "crs": f"EPSG:{epsg}",
        "origin_sw": [west, south],
        "buildings": buildings,
        "trees": trees,
    }
    (out_dir / "scene.json").write_text(json.dumps(scene) + "\n")
    return scene


def _box(x0: float, y0: float, x1: float, y1: float) -> Any:
    from shapely.geometry import box

    return box(x0, y0, x1, y1)


__all__ = ["SCHEMA", "export_scene"]
