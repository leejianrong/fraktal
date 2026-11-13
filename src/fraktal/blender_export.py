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
from rasterio.features import rasterize
from shapely.geometry import LineString

from fraktal import (
    building_grammar,
    cables,
    camera,
    compounds,
    lighting,
    markings,
    props_scene,
    trees3d,
)
from fraktal import heights as heights_mod
from fraktal.coast import TIDAL_FLAT
from fraktal.fields import PADDY
from fraktal.ground_soil import apply_soil, seal_streets, soil_style
from fraktal.ids import substream
from fraktal.kernels.noise import lattice_shape, value_noise
from fraktal.kernels.paddy import paddy_tone
from fraktal.landcover_gen import WATER
from fraktal.objects import CAR_COLOURS, place
from fraktal.procedural import read_buildings, read_fields, read_parcels, read_roads
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
    tile_px_for,
)

SCHEMA = 1
WET_CODES = (WATER, PADDY_CODES["flooded"], FISHPOND_WATER)  # standing water: glossy
ROUGH_WATER, ROUGH_MUD, ROUGH_LAND = 0.06, 0.35, 0.95
VEGETATION = (10, 20, 30, 40, 90, 95)  # classes that get patchy hue and fine grain
EDGE_WARP = ((1.5, 0.8), (0.5, 0.3))  # (noise cell, shift) in land-cover cells, bends class edges
EDGE_SOFT = 0.3  # blur sigma between classes, in land-cover cells
TERRAIN_CELL_M, TERRAIN_MAX_CELLS, TERRAIN_FLAT_M = 1.0, 700, 0.05  # ground mesh cell and limits
HUE_CELL_M, HUE_STRENGTH = 25.0, 0.12  # patches of yellower or bluer green
GRAIN_CELL_M, GRAIN_STRENGTH = 0.7, 0.10  # blades and clumps, finer than the 2.5D texture


def terrain_nodes(dem: Any, gsd_m: float, size_m: float) -> tuple[Any, float, float]:
    """Heights at the nodes of a square ground mesh, relative to the tile's lowest point.

    ``dem`` is the tile's elevation raster, north up, ``gsd_m`` a pixel. Returns ``(nodes, cell_m,
    low_m)``: an (n + 1, n + 1) float32 array with row 0 the southern edge and column 0 the western
    edge (index ``[j, i]`` is at x = i * cell, y = j * cell), the mesh cell, and the lowest
    elevation.
    The mesh is about ``TERRAIN_CELL_M`` fine, coarser for a very large tile.
    """
    n = max(1, min(TERRAIN_MAX_CELLS, round(size_m / TERRAIN_CELL_M)))
    cell = size_m / n
    grid = np.asarray(dem, dtype="float64")[::-1]  # row 0 south
    if not np.isfinite(grid).all():  # nodata outside the world: take the tile's own mean
        grid = np.where(
            np.isfinite(grid), grid, np.nanmean(grid) if np.isfinite(grid).any() else 0.0
        )
    rows, cols = grid.shape
    axis = np.arange(n + 1) * cell
    fy = np.clip(axis / gsd_m - 0.5, 0.0, rows - 1.0)
    fx = np.clip(axis / gsd_m - 0.5, 0.0, cols - 1.0)
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    wy, wx = (fy - y0)[:, None], (fx - x0)[None, :]
    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    nodes = top * (1 - wy) + bottom * wy
    low = float(nodes.min())
    return (nodes - low).astype("float32"), cell, low


def terrain_height(nodes: Any, cell_m: float, x: Any, y: Any) -> Any:
    """Bilinear ground height at tile coordinates from a ``terrain_nodes`` grid."""
    n = nodes.shape[0] - 1
    fx = np.clip(np.asarray(x, dtype="float64") / cell_m, 0.0, n - 1e-9)
    fy = np.clip(np.asarray(y, dtype="float64") / cell_m, 0.0, n - 1e-9)
    i, j = fx.astype(int), fy.astype(int)
    wx, wy = fx - i, fy - j
    return (
        nodes[j, i] * (1 - wx) * (1 - wy) + nodes[j, i + 1] * wx * (1 - wy)
        + nodes[j + 1, i] * (1 - wx) * wy + nodes[j + 1, i + 1] * wx * wy
    )  # fmt: skip


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
MARK_MARGIN_M = 10.0  # roads this far outside the tile still get marks, so edges look continuous
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
    roads: Any, mask: Any, west: float, south: float, gsd_m: float, side: int, biome: str = ""
) -> list[list[float]]:
    """Roadside poles every 40 m (a city's own spacing) on streets wide enough for traffic,
    alternating sides."""
    out: list[list[float]] = []
    size = side * gsd_m
    spacing_m, height_m = cables.pole_plan(biome, POLE_SPACING_M, POLE_HEIGHT_M)
    for _, r in roads.sort_values("id").iterrows():
        if r.width_m < POLE_MIN_ROAD_WIDTH_M or r.geometry.length < spacing_m:
            continue
        for k, d in enumerate(np.arange(spacing_m / 2, r.geometry.length, spacing_m)):
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
                out.append([round(x, 2), round(y, 2), height_m])
    return out


def _paddy_look(
    ground: Any, surface: Any, fields: Any, transform: Any, seed: int, gsd_m: float
) -> Any:
    """``ground`` with each paddy block given its own tone, muted colour and rows (FRK-141)."""
    blocks = fields[fields["field_type"] == PADDY].sort_values("id")
    if blocks.empty:
        return ground
    shapes = [(g, i + 1) for i, g in enumerate(blocks.geometry)]
    cells = rasterize(shapes, surface.shape, transform=transform, dtype="int32")
    cells = np.where(np.isin(surface, list(PADDY_CODES.values())), cells, 0)  # not the bunds
    names = list(PADDY_CODES)
    state = np.array([0] + [names.index(str(v)) for v in blocks["state"]], dtype="int64")
    angle = np.array([0.0] + [math.radians(float(v)) for v in blocks["row_angle_deg"]], "float32")
    jitter = substream(seed, "paddy:look").uniform(-1, 1, size=(len(blocks) + 1, 3))
    return paddy_tone(ground, cells, state, jitter.astype("float32"), angle, gsd_m)


def _street_finder(roads: Any) -> Any:
    """A function from a building centroid to the nearest point on any road, in tile coordinates."""
    from shapely.ops import nearest_points
    from shapely.strtree import STRtree

    lines = [g for g in roads.geometry if not g.is_empty] if len(roads) else []
    tree = STRtree(lines) if lines else None

    def find(centroid: Any, west: float, south: float) -> list[float] | None:
        if tree is None:
            return None
        road = lines[int(tree.nearest(centroid))]
        point = nearest_points(road, centroid)[0]
        return [round(point.x - west, 3), round(point.y - south, 3)]

    return find


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
    prop_density: float = 60.0,
    wall_share: float | None = None,
    wall_textures: bool = True,
    height_profile: str = "auto",
) -> dict[str, Any]:
    """Write the tile around ``center_lonlat`` and return the scene description.

    The ground carries the 2.5D renderer's fractal texture (``texture`` is its strength, 0 for flat
    colour) and ``trees`` are thickets and loners, placed by ``trees3d.scatter``.
    """
    px = tile_px_for(size_m, gsd_m)
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
    dem_file = world_dir / "dem.tif"
    dem = _warp(dem_file, transform, epsg, side, Resampling.bilinear) if dem_file.exists() else None
    if fields is not None and len(fields):
        assert dem is not None, "a world with fields needs dem.tif"
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
    biome = json.loads((world_dir / "world.json").read_text()).get("biome", "")
    ground = apply_soil(ground, surface, seed, gsd_m, soil_style(biome, seed))
    ground = seal_streets(ground, surface, biome, ROAD_LOCAL)
    if fields is not None and len(fields):
        ground = _paddy_look(ground, surface, local_fields, transform, seed, gsd_m)
    if texture > 0:
        amp = np.ones(256, dtype="float32")
        for code, value in TEXTURE_CLASS_AMP.items():
            amp[code] = value
        modulation = texture * texture_field(seed, (side, side), gsd_m) * amp[surface]
        ground = np.clip(ground * (1.0 + modulation)[..., None], 0.0, 1.0).astype("float32")
    walls = compounds.build(
        read_parcels(world_dir).to_crs(epsg), local, roads, (west, south), side * gsd_m, seed,
        compounds.share_for(biome) if wall_share is None else wall_share, biome,
    )  # fmt: skip
    trees: list[list[Any]] = []
    if tree_density > 0:
        segments = [seg for c in walls for seg in c["segments"]]
        trees = trees3d.scatter(surface, mask, gsd_m, tree_density, seed, biome, segments)
        trees += trees3d.yard_trees(walls, seed, biome, compounds.YARD_TREE_SHARE * tree_density)
        trees += trees3d.street_trees(
            [(list(r.xy), r.width_m) for r in _marking_roads(roads, west, south, side * gsd_m)
             if r.road_class in ("arterial", "local")],
            mask, gsd_m, biome, seed, tree_density,
        )  # fmt: skip
    if field_palms is not None:  # planted rows: every palm of an oil-palm field
        palm_shapes, palm_heights, _ = field_palms
        for (row, col, radius, *_rest), height in zip(palm_shapes, palm_heights, strict=True):
            x, y = round((col + 0.5) * gsd_m, 2), round((side - row - 0.5) * gsd_m, 2)
            if 0 <= x <= side * gsd_m and 0 <= y <= side * gsd_m:
                trees.append([x, y, round(radius * gsd_m, 2), round(float(height), 2), "palm"])
    vehicles = _vehicles(surface, road, mask, gsd_m, side, vehicle_density, seed)
    poles = _poles(roads, mask, west, south, gsd_m, side, biome)
    props = props_scene.sample(
        surface,
        mask,
        road,
        gsd_m,
        side,
        prop_density,
        seed,
        walls=[s for c in walls for s in c["segments"]],
    )
    street_scale, street_mix = props_scene.street_plan(biome)
    props += props_scene.sample(
        surface,
        mask,
        road,
        gsd_m,
        side,
        prop_density * props_scene.STREET_DENSITY_RATIO * street_scale,
        seed,
        walls=[s for c in walls for s in c["segments"]],
        mix=street_mix,
        stream="props:street",
        avoid=props,
    )
    props = props_scene.align_to_roads(
        props,
        [
            (LineString(r.xy), r.width_m)
            for r in _marking_roads(roads, west, south, side * gsd_m)
            if r.road_class in ("arterial", "local") and r.width_m >= 6.0
        ],
        mask, gsd_m, side, seed,
    )  # fmt: skip
    yards = _yard_sites(read_parcels(world_dir).to_crs(epsg), local, west, south)
    props += props_scene.yard_props(yards, seed)
    street_of = _street_finder(roads)
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
                "street": street_of(row.geometry.centroid, west, south),
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
                b["id"],
                b["ring"],
                b["height_m"],
                b["roof_material"],
                seed,
                roof_mix=building_grammar.tin_mix(biome),
                street=None if b["street"] is None else (b["street"][0], b["street"][1]),
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
    packed = props_scene.pack(props)
    for key in ("vertices", "faces", "face_material", "face_owner"):
        np.save(out_dir / f"props_{key}.npy", packed[key])
    packed_walls = compounds.pack(walls)
    for key in ("vertices", "faces", "face_material", "face_owner"):
        np.save(out_dir / f"walls_{key}.npy", packed_walls[key])
    surface_parts = markings.build(
        _marking_roads(roads, west, south, side * gsd_m),
        camera.obstacles(buildings, [g for c in walls for g in c["segments"]]),
        biome,
    )
    packed_marks = markings.pack(surface_parts)
    for key in ("vertices", "faces", "face_material", "face_owner"):
        np.save(out_dir / f"markings_{key}.npy", packed_marks[key])
    wires = cables.build(poles, biome, seed)
    packed_cables = cables.pack(wires)
    for key in ("vertices", "faces", "face_material", "face_owner"):
        np.save(out_dir / f"cables_{key}.npy", packed_cables[key])
    np.save(out_dir / "ground.npy", ground)
    np.save(out_dir / "ground_rough.npy", rough)
    terrain = None
    if dem is not None:
        nodes, cell_m, low_m = terrain_nodes(dem, gsd_m, side * gsd_m)
        if float(nodes.max()) > TERRAIN_FLAT_M:  # a flat tile keeps its single-quad ground
            np.save(out_dir / "ground_height.npy", nodes)
            terrain = {
                "file": "ground_height.npy",
                "cell_m": round(cell_m, 4),
                "low_m": round(low_m, 2),
                "relief_m": round(float(nodes.max()), 2),
            }
    scene = {
        "schema": SCHEMA,
        "size_m": side * gsd_m,
        "gsd_m": gsd_m,
        "px": side,
        "sun": {"azimuth_deg": sun[0], "elevation_deg": sun[1]},
        "sky": lighting.draw(seed, sun[1], biome),
        "crs": f"EPSG:{epsg}",
        "origin_sw": [west, south],
        "buildings": buildings,
        "materials": mesh["materials"],
        "trees": trees,
        "vehicles": vehicles,
        "poles": poles,
        "props": props,
        "props_stats": props_scene.repeat_stats(props),
        "prop_materials": packed["materials"],
        "walls": walls,
        "wall_materials": packed_walls["materials"],
        "wall_textures": compounds.textures() if wall_textures else None,
        "markings": {"parts": len(surface_parts), "materials": packed_marks["materials"]},
        "cables": {"wires": len(wires), "materials": packed_cables["materials"]},
    }
    if terrain:
        scene["terrain"] = terrain
    (out_dir / "scene.json").write_text(json.dumps(scene) + "\n")
    return scene


def _marking_roads(roads: Any, west: float, south: float, size_m: float) -> list[markings.RoadLine]:
    """The road lines inside the tile and a margin, in tile metres, for ``markings.build``."""
    m = MARK_MARGIN_M
    window = _box(west - m, south - m, west + size_m + m, south + size_m + m)
    out = []
    for row in roads.itertuples():
        clipped = row.geometry.intersection(window)
        for g in getattr(clipped, "geoms", [clipped]):
            if g.geom_type == "LineString" and g.length > 1.0:
                xy = [(x - west, y - south) for x, y in g.coords]
                out.append(markings.RoadLine(xy, float(row.width_m), str(row.road_class)))
    return out


def _yard_sites(parcels: Any, buildings: Any, west: float, south: float) -> list[dict[str, Any]]:
    """Industrial sites as tile-metre yards: the lot less its halls, pulled in from the fence."""
    from shapely import affinity

    if "land_use" not in parcels:
        return []
    sites = []
    for row in parcels[parcels["land_use"] == "industrial"].itertuples():
        owned = None
        if "parcel_id" in buildings:
            owned = buildings["parcel_id"].astype(str) == str(row.id)
        yard = row.geometry.buffer(-2.5)
        if owned is not None:
            for hall in buildings[owned].geometry:
                yard = yard.difference(hall.buffer(3.0))
        yard = affinity.translate(yard, -west, -south)
        (x0, y0), (x1, y1) = list(row.geometry.exterior.coords)[:2]
        sites.append({"id": row.id, "yard": yard, "yaw": math.atan2(y1 - y0, x1 - x0)})
    return sites


def _box(x0: float, y0: float, x1: float, y1: float) -> Any:
    from shapely.geometry import box

    return box(x0, y0, x1, y1)


__all__ = ["SCHEMA", "export_scene"]
