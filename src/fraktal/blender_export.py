"""Export one tile of a procedural world for the Blender experiment (ADR-0010, Stage 0).

Writes ``scene.json`` (building footprints and heights, in metres from the tile's south-west
corner, east and north positive), the building meshes from ``fraktal.building_grammar`` as four
``mesh_*.npy`` arrays, and ``ground.npy`` (an H x W x 3 float image, north up, of the
ground colours with roads painted, the same palette the 2.5D renderer uses). The Blender side
(``src/fraktal/blender_scene.py``) reads only these files and the assets directory, so this is
the renderer file contract in miniature: the scene is data, not objects in memory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from fraktal import building_grammar, trees3d
from fraktal import heights as heights_mod
from fraktal.coast import TIDAL_FLAT
from fraktal.ids import substream
from fraktal.kernels.noise import lattice_shape, value_noise
from fraktal.landcover_gen import WATER
from fraktal.objects import CAR_COLOURS, place
from fraktal.procedural import read_buildings, read_fields, read_roads
from fraktal.render import (
    DEFAULT_HEIGHT_M,
    FISHPOND_WATER,
    LANDCOVER_RGB,
    PADDY_CODES,
    ROAD_ARTERIAL,
    ROAD_LOCAL,
    TEXTURE_CLASS_AMP,
    RenderConfig,
    _building_rasters,
    _field_raster,
    _field_trees,
    _road_raster,
    _warp,
    region_grid,
    texture_field,
)

SCHEMA = 1
WET_CODES = (WATER, PADDY_CODES["flooded"], FISHPOND_WATER)  # standing water: glossy
ROUGH_WATER, ROUGH_MUD, ROUGH_LAND = 0.06, 0.35, 0.95
VEGETATION = (10, 20, 30, 40, 90, 95)  # classes that get patchy hue and fine grain
EDGE_WARP = ((1.5, 0.8), (0.5, 0.3))  # (noise cell, shift) in land-cover cells, bends class edges
EDGE_SOFT = 0.3  # blur sigma between classes, in land-cover cells
HUE_CELL_M, HUE_STRENGTH = 25.0, 0.12  # patches of yellower or bluer green
GRAIN_CELL_M, GRAIN_STRENGTH = 0.7, 0.10  # blades and clumps, finer than the 2.5D texture


def _tile_heights(world_dir: Path, buildings: Any, profile: str, seed: int) -> dict[str, float]:
    """Each building's height: the world's own, or remapped to the biome's storey profile.

    ``profile`` is ``"world"`` (keep), ``"auto"`` (the biome's, from ``world.json``) or a name in
    ``heights.PROFILES``. The remap runs over the whole world, so a building's height does not
    depend on the tile window.
    """
    raw = (
        buildings["height_m"].to_numpy(dtype="float64")
        if "height_m" in buildings
        else np.full(len(buildings), np.nan)
    )
    base = np.where(np.isnan(raw), DEFAULT_HEIGHT_M, raw)
    ids = [str(i) for i in buildings["id"]]
    if profile == "auto":
        biome = json.loads((world_dir / "world.json").read_text()).get("biome", "")
        chosen = heights_mod.profile_for(biome)
    else:
        chosen = None if profile == "world" else profile
    if chosen is not None:
        base = heights_mod.remap(base, ids, chosen, seed)
    return dict(zip(ids, (float(h) for h in base), strict=True))


POLE_SPACING_M = 40.0
POLE_HEIGHT_M = 8.0
POLE_MIN_ROAD_WIDTH_M = 6.0
CAR_LENGTH_M, CAR_WIDTH_M = 4.5, 1.9


def _vehicles(
    surface: Any, road: Any, mask: Any, gsd_m: float, side: int, density: float, seed: int
) -> list[list[float]]:
    """Cars on the roads: x, y, heading (radians, anticlockwise from east), body colour index."""
    if road is None or density <= 0:
        return []
    on_road = np.where(np.isin(road, (ROAD_LOCAL, ROAD_ARTERIAL)), road, 0)
    shapes, _, colours = place(surface, mask, gsd_m, density, seed, on_road)
    out: list[list[float]] = []
    palette = [tuple(c) for c in CAR_COLOURS.tolist()]
    for (row, col, _a, _b, theta, kind), colour in zip(shapes, colours, strict=True):
        if kind != 1:  # trees come first and are ellipses
            continue
        out.append(
            [
                round((col + 0.5) * gsd_m, 2),
                round((side - row - 0.5) * gsd_m, 2),
                round(-float(theta), 4),  # image rows run south, so the angle flips
                float(palette.index(tuple(colour))),
            ]
        )
    return out


def _poles(
    roads: Any, mask: Any, west: float, south: float, gsd_m: float, side: int
) -> list[list[float]]:
    """Roadside poles every 40 m on streets wide enough for traffic, alternating sides."""
    out: list[list[float]] = []
    size = side * gsd_m
    for _, r in roads.sort_values("id").iterrows():
        if r.width_m < POLE_MIN_ROAD_WIDTH_M or r.geometry.length < 2 * POLE_SPACING_M / 2:
            continue
        for k, d in enumerate(np.arange(POLE_SPACING_M / 2, r.geometry.length, POLE_SPACING_M)):
            p = r.geometry.interpolate(float(d))
            a, b = r.geometry.interpolate(max(d - 1.0, 0.0)), r.geometry.interpolate(d + 1.0)
            tx, ty = b.x - a.x, b.y - a.y
            norm = math.hypot(tx, ty)
            if norm == 0:
                continue
            offset = (r.width_m / 2 + 1.2) * (1 if k % 2 == 0 else -1)
            x, y = p.x - ty / norm * offset - west, p.y + tx / norm * offset - south
            col, row = int(x / gsd_m), int(side - 1 - y / gsd_m)
            if 0 <= x < size and 0 <= y < size and mask[row, col] == 0:
                out.append([round(x, 2), round(y, 2), POLE_HEIGHT_M])
    return out


def _class_colours(classes: Any) -> Any:
    """An (H, W, 3) float image of the display colour of each land-cover class."""
    out = np.zeros((*classes.shape, 3), dtype="float32")
    for code in np.unique(classes):
        out[classes == code] = np.array(LANDCOVER_RGB.get(int(code), (120, 118, 116))) / 255.0
    return out


def _smooth(seed: int, name: str, side: int, cell_px: float) -> Any:
    """Value noise in [-1, 1] over a side x side grid with features about ``cell_px`` wide."""
    cell_px = max(cell_px, 1.0)
    shape = (side, side)
    lattice = substream(seed, name).uniform(-1, 1, size=lattice_shape(shape, cell_px))
    return value_noise(lattice, cell_px, shape)


def _bend_edges(land: Any, seed: int, cell_px: float) -> Any:
    """The class map with its block edges pushed about by smooth noise, so they are not squares."""
    side = land.shape[0]
    rows = np.arange(side, dtype="float32")[:, None] + np.zeros((1, side), dtype="float32")
    cols = rows.T.copy()
    for n, (cell, shift) in enumerate(EDGE_WARP):
        rows = rows + shift * cell_px * _smooth(seed, f"ground:warp:r{n}", side, cell * cell_px)
        cols = cols + shift * cell_px * _smooth(seed, f"ground:warp:c{n}", side, cell * cell_px)
    r = np.clip(np.rint(rows), 0, side - 1).astype(int)
    c = np.clip(np.rint(cols), 0, side - 1).astype(int)
    return land[r, c]


def _moving_mean(img: Any, width: int, axis: int) -> Any:
    """Moving average of ``width`` samples along ``axis``, edges repeated."""
    pad = width // 2
    pads = [(0, 0)] * img.ndim
    pads[axis] = (pad, pad)
    padded = np.pad(img, pads, mode="edge")
    csum = np.cumsum(padded, axis=axis, dtype="float64")
    zero = np.zeros_like(np.take(csum, [0], axis=axis))
    csum = np.concatenate([zero, csum], axis=axis)
    n = img.shape[axis]
    hi = np.take(csum, np.arange(width, width + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return ((hi - lo) / width).astype("float32")


def _blur(img: Any, sigma_px: float) -> Any:
    """A close-to-Gaussian blur from three box passes per axis; no SciPy needed."""
    width = int(round(math.sqrt(4 * sigma_px**2 + 1)))
    width += 1 - width % 2
    if width < 3:
        return img
    for axis in (0, 1):
        for _ in range(3):
            img = _moving_mean(img, width, axis)
    return img


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
    vehicle_density: float = 1.0,
    height_profile: str = "auto",
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
    height_of = _tile_heights(world_dir, local, height_profile, seed)
    roads = read_roads(world_dir).to_crs(epsg)
    land_file = world_dir / "landcover.tif"
    landcover = _warp(land_file, transform, epsg, side, Resampling.nearest)
    with rasterio.open(land_file) as src:
        cell_px = max(abs(src.res[0]) / gsd_m, 1.0)  # the world's land-cover cell, in tile pixels
    landcover = _bend_edges(landcover, seed, cell_px)
    soft_land = landcover
    _, mask, _ = _building_rasters(local, transform, side)
    surface = landcover
    road = _road_raster(roads, transform, side) if len(roads) else None
    if road is not None:
        road[mask > 0] = 0
        surface = np.where(road > 0, road, landcover).astype("uint8")
    fields = read_fields(world_dir)
    field_palms = None
    if fields is not None and len(fields):
        dem = _warp(world_dir / "dem.tif", transform, epsg, side, Resampling.bilinear)
        local_fields = fields.to_crs(epsg)
        painted = _field_raster(local_fields, transform, side, dem)
        painted[mask > 0] = 0
        if road is not None:
            painted[road > 0] = 0
        surface = np.where(painted > 0, painted, surface).astype("uint8")
        field_palms = _field_trees(local_fields, transform, side, seed)
    rough = np.full((side, side), ROUGH_LAND, dtype="float32")
    rough[np.isin(surface, WET_CODES)] = ROUGH_WATER
    rough[surface == TIDAL_FLAT] = ROUGH_MUD
    ground = _class_colours(surface)
    softened = _blur(_class_colours(soft_land), EDGE_SOFT * cell_px)
    ground = np.where(
        (surface == soft_land)[..., None], softened, ground
    )  # roads and fields stay sharp
    green = np.isin(surface, VEGETATION)[..., None]
    hue = _smooth(seed, "ground:hue", side, HUE_CELL_M / gsd_m)[..., None]
    ground = np.where(
        green,
        ground * (1.0 + HUE_STRENGTH * hue * np.array([1.0, 0.5, -1.0], dtype="float32")),
        ground,
    )
    grain = _smooth(seed, "ground:grain", side, GRAIN_CELL_M / gsd_m)[..., None]
    ground = np.where(green, ground * (1.0 + GRAIN_STRENGTH * grain), ground)
    if texture > 0:
        amp = np.ones(256, dtype="float32")
        for code, value in TEXTURE_CLASS_AMP.items():
            amp[code] = value
        modulation = texture * texture_field(seed, (side, side), gsd_m) * amp[surface]
        ground = np.clip(ground * (1.0 + modulation)[..., None], 0.0, 1.0).astype("float32")
    biome = json.loads((world_dir / "world.json").read_text()).get("biome", "")
    trees = []
    if tree_density > 0:
        shapes, heights, _ = place(surface, mask, gsd_m, tree_density, seed)
        for (row, col, radius, *_rest, is_car), height in zip(shapes, heights, strict=True):
            if is_car == 0:  # trees come first and have no car flag
                tx, ty = round((col + 0.5) * gsd_m, 2), round((side - row - 0.5) * gsd_m, 2)
                trees.append(
                    [
                        tx,
                        ty,
                        round(radius * gsd_m, 2),
                        round(float(height), 2),
                        trees3d.species_for(biome, tx, ty, float(height), seed),
                    ]
                )
    if field_palms is not None:  # planted rows: every palm of an oil-palm field
        palm_shapes, palm_heights, _ = field_palms
        for (row, col, radius, *_rest), height in zip(palm_shapes, palm_heights, strict=True):
            x, y = round((col + 0.5) * gsd_m, 2), round((side - row - 0.5) * gsd_m, 2)
            if 0 <= x <= side * gsd_m and 0 <= y <= side * gsd_m:
                trees.append([x, y, round(radius * gsd_m, 2), round(float(height), 2), "palm"])
    vehicles = _vehicles(surface, road, mask, gsd_m, side, vehicle_density, seed)
    poles = _poles(roads, mask, west, south, gsd_m, side)
    buildings = []
    for row in local.itertuples():
        if row.geometry.is_empty or not row.geometry.intersects(
            _box(west, south, west + side * gsd_m, north)
        ):
            continue
        height = height_of[row.id]
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
    meshes = []
    owner: list[int] = []  # which entry of `buildings` each mesh belongs to
    for index, b in enumerate(buildings):
        try:
            built = building_grammar.build(
                b["id"], b["ring"], b["height_m"], b["roof_material"], seed
            )
            b["typology"] = built.typology
            meshes.append(built)
            owner.append(index)
        except ValueError:  # a sliver footprint: it stays in `buildings` (labels) but gets no mesh
            continue
    mesh = building_grammar.pack(meshes)
    for key in ("vertices", "face_sizes", "face_verts", "face_material"):
        np.save(out_dir / f"mesh_{key}.npy", mesh[key])
    per_mesh = [len(m.faces) for m in meshes]
    np.save(  # the building each face belongs to, for the instance labels
        out_dir / "mesh_face_building.npy",
        np.repeat(np.asarray(owner, dtype="int32"), per_mesh),
    )
    np.save(out_dir / "ground.npy", ground)
    np.save(out_dir / "ground_rough.npy", rough)
    scene = {
        "schema": SCHEMA,
        "size_m": side * gsd_m,
        "gsd_m": gsd_m,
        "px": side,
        "sun": {"azimuth_deg": sun[0], "elevation_deg": sun[1]},
        "crs": f"EPSG:{epsg}",
        "origin_sw": [west, south],
        "buildings": buildings,
        "materials": mesh["materials"],
        "trees": trees,
        "vehicles": vehicles,
        "poles": poles,
    }
    (out_dir / "scene.json").write_text(json.dumps(scene) + "\n")
    return scene


def _box(x0: float, y0: float, x1: float, y1: float) -> Any:
    from shapely.geometry import box

    return box(x0, y0, x1, y1)


__all__ = ["SCHEMA", "export_scene"]
