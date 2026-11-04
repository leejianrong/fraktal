"""Crude renderer (ADR-0004): flat per-class reflectance on a DSM under one sun with hard shadows.

The render grid is a metric (UTM) grid at ``gsd_m`` anchored to a multiple of the pixel size, so
the same world and config always give the same bytes. There is no randomness here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject

from fraktal.fields import OIL_PALM, row_points
from fraktal.ids import substream
from fraktal.kernels.filters import gaussian_blur
from fraktal.kernels.noise import lattice_shape, value_noise
from fraktal.kernels.shadow import shadow_mask
from fraktal.labels import independent_building_mask
from fraktal.objects import roof_pattern, row_objects, scatter
from fraktal.palette import Palette
from fraktal.sensor import add_noise, apply_psf, blank_wedge
from fraktal.terrain import TerrainError

DEFAULT_HEIGHT_M = 6.0
ROAD_LOCAL = 201  # surface codes for painted roads; never written to the land-cover labels
ROAD_ARTERIAL = 202
FIELD_PALM = 203  # oil palm ground
ROAD_TRACK = 204  # plantation access track
AMBIENT = 0.35  # share of direct light that reaches a shadowed pixel
ROOF_RGB = (140.0, 128.0, 115.0)
# Roof materials a building can be made of (concrete, clay tile, corrugated metal, bitumen).
ROOF_MATERIALS: tuple[tuple[float, float, float], ...] = (
    (140.0, 128.0, 115.0),
    (156.0, 92.0, 72.0),
    (186.0, 186.0, 184.0),
    (78.0, 78.0, 84.0),
)
# ESA WorldCover classes -> flat RGB reflectance (0-255).
LANDCOVER_RGB: dict[int, tuple[float, float, float]] = {
    10: (46.0, 94.0, 40.0),  # tree cover
    20: (112.0, 128.0, 70.0),  # shrubland
    30: (140.0, 160.0, 80.0),  # grassland
    40: (170.0, 160.0, 90.0),  # cropland
    50: (120.0, 118.0, 116.0),  # built-up ground
    60: (196.0, 170.0, 130.0),  # bare / sparse vegetation
    70: (240.0, 240.0, 245.0),  # snow and ice
    80: (40.0, 70.0, 110.0),  # permanent water
    90: (70.0, 120.0, 100.0),  # herbaceous wetland
    95: (50.0, 100.0, 70.0),  # mangroves
    100: (150.0, 150.0, 110.0),  # moss and lichen
    ROAD_LOCAL: (146.0, 130.0, 110.0),  # unpaved or dusty street
    ROAD_ARTERIAL: (88.0, 88.0, 92.0),  # asphalt
    FIELD_PALM: (104.0, 108.0, 64.0),  # ground and weeds under oil palm
    ROAD_TRACK: (150.0, 118.0, 86.0),  # laterite access track
}


# Texture octaves as (cell size in metres, weight): fields and ground below the 10 m land cover.
TEXTURE_OCTAVES: tuple[tuple[float, float], ...] = (
    (40.0, 0.3),
    (10.0, 0.3),
    (2.5, 0.25),
    (0.8, 0.15),
)
# How strongly each land-cover class is textured (water is smooth, vegetation and soil are not).
TEXTURE_CLASS_AMP: dict[int, float] = {80: 0.3, 70: 0.3}
ROOF_TEXTURE_AMP = 0.4
BRIGHT_ROOF_GAIN = 1.6  # metal and whitewashed roofs that saturate in real imagery
HAZE_RGB = (200.0, 205.0, 215.0)  # colour of scattered skylight that haze adds

# Per-channel gain that matches the V1 synthetic chips' channel means to the real Khartoum chips
# (R 133 -> 88, G 126 -> 118, B 109 -> 34): the real chips have a strong green cast (FRK-12).
KHARTOUM_TONE: tuple[float, float, float] = (0.66, 0.94, 0.31)


@dataclass(frozen=True)
class RenderConfig:
    gsd_m: float = 0.5
    tile_px: int = 500
    size_m: float = 2000.0
    sun_azimuth_deg: float = 135.0
    sun_elevation_deg: float = 50.0
    seed: int = 0
    variation: float = 0.0  # 0 is the flat V1 look; 0.15 is a typical spread of reflectance
    tone: tuple[float, float, float] = (1.0, 1.0, 1.0)  # per-channel gain on the final image
    haze: float = 0.0  # optical depth of the atmosphere; 0 is clear, 0.2 is hazy
    psf_sigma_px: float = 0.0  # sensor blur (Gaussian sigma in output pixels); 0 is none
    noise_sigma: float = 0.0  # sensor noise (std in digital numbers); 0 is none
    penumbra_m: float = 0.0  # shadow edge softness (blur sigma in metres); 0 is a hard edge
    texture: float = 0.0  # strength of sub-10 m texture; 0 is flat, 0.7 is the synth default
    ambient: float = AMBIENT  # light left inside a shadow; lower is a deeper shadow
    contrast: float = 1.0  # stretch about the channel means; above 1 clips the extremes
    bright_roofs: float = 0.0  # share of buildings with very bright (often clipped) roofs
    blank_prob: float = 0.0  # chance a tile has a no-data region, like a scene footprint edge
    blank_max_fraction: float = 0.3  # largest share of a tile the no-data region can cover
    objects: float = 0.0  # density of trees and vehicles; 0 is none, 1 is the default density
    roof_detail: float = 0.0  # strength of ridge and corrugation stripes on roofs
    palette: Palette | None = None  # class and roof colours fitted from real train chips


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def region_grid(
    center_lonlat: tuple[float, float], config: RenderConfig
) -> tuple[int, Affine, tuple[int, int]]:
    """EPSG code, transform and (tiles_per_side, side_px) of the render grid."""
    lon, lat = center_lonlat
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    tiles = max(1, math.ceil(config.size_m / config.gsd_m / config.tile_px))
    side = tiles * config.tile_px
    half = side * config.gsd_m / 2
    west = math.floor((x - half) / config.gsd_m) * config.gsd_m
    north = math.ceil((y + half) / config.gsd_m) * config.gsd_m
    return epsg, from_origin(west, north, config.gsd_m, config.gsd_m), (tiles, side)


def _warp(path: Path, transform: Affine, epsg: int, side: int, method: Resampling) -> Any:
    dtype = "float32" if method == Resampling.bilinear else "uint8"
    out = np.zeros((side, side), dtype=dtype)
    with rasterio.open(path) as src:
        to_src = Transformer.from_crs(epsg, src.crs, always_xy=True)
        corners = [transform @ (0, 0), transform @ (side, side)]
        (x0, y0), (x1, y1) = (to_src.transform(*c) for c in corners)
        b = src.bounds
        if not (b.left <= min(x0, x1) and max(x0, x1) <= b.right):
            raise TerrainError(f"{path.name} does not cover the render region")
        if not (b.bottom <= min(y0, y1) and max(y0, y1) <= b.top):
            raise TerrainError(f"{path.name} does not cover the render region")
        reproject(
            rasterio.band(src, 1),
            out,
            dst_transform=transform,
            dst_crs=f"EPSG:{epsg}",
            resampling=method,
        )
    return out


def _building_rasters(
    buildings: gpd.GeoDataFrame, transform: Affine, side: int
) -> tuple[Any, Any, Any]:
    """Height raster (m), 0/1 footprint mask and building-index raster (row + 1, 0 is none).

    Painted lowest-first so taller wins.
    """
    heights = buildings["height_m"].fillna(DEFAULT_HEIGHT_M) if "height_m" in buildings else None
    if heights is None:
        heights = np.full(len(buildings), DEFAULT_HEIGHT_M)
    order = np.lexsort((buildings["id"].to_numpy(), np.asarray(heights)))
    shapes = [
        (buildings.geometry.iloc[i], float(np.asarray(heights)[i]))
        for i in order
        if not buildings.geometry.iloc[i].is_empty
    ]
    height = np.zeros((side, side), dtype="float32")
    mask = np.zeros((side, side), dtype="uint8")
    index = np.zeros((side, side), dtype="uint32")
    if shapes:
        height = rasterize(shapes, (side, side), transform=transform, dtype="float32")
        painted = [(buildings.geometry.iloc[i], int(i) + 1) for i in order]
        painted = [(g, v) for g, v in painted if not g.is_empty]
        index = rasterize(painted, (side, side), transform=transform, dtype="uint32")
        footprints = rasterize([(g, 1) for g, _ in shapes], (side, side), transform=transform)
        mask = (footprints > 0).astype("uint8")
    return height, mask, index


def _road_raster(roads: gpd.GeoDataFrame, transform: Affine, side: int) -> Any:
    """0 off roads, ROAD_LOCAL or ROAD_ARTERIAL on them; wider arterials win at junctions."""
    shapes = []
    for _, road in roads.sort_values(["road_class", "id"], ascending=[False, True]).iterrows():
        code = {"arterial": ROAD_ARTERIAL, "track": ROAD_TRACK}.get(road.road_class, ROAD_LOCAL)
        shapes.append((road.geometry.buffer(float(road.width_m) / 2, cap_style="flat"), code))
    shapes.sort(key=lambda s: s[1] not in (ROAD_LOCAL, ROAD_TRACK))  # arterials painted last
    if not shapes:
        return np.zeros((side, side), dtype="uint8")
    return rasterize(shapes, (side, side), transform=transform, dtype="uint8")


TIN_ROOF = (178.0, 176.0, 172.0)  # corrugated metal, often rusting
TILE_ROOF = (156.0, 92.0, 72.0)  # fired clay


def _field_raster(fields: gpd.GeoDataFrame, transform: Affine, side: int) -> Any:
    """0 off fields, else the surface code of each field's ground."""
    shapes = [(f.geometry, FIELD_PALM) for _, f in fields.sort_values("id").iterrows()]
    if not shapes:
        return np.zeros((side, side), dtype="uint8")
    return rasterize(shapes, (side, side), transform=transform, dtype="uint8")


def _field_trees(fields: gpd.GeoDataFrame, transform: Affine, side: int, seed: int) -> Any:
    """Crown shapes, heights and colours for every planted row, or None."""
    inverse = ~transform
    pts = []
    for _, f in fields.sort_values("id").iterrows():
        if f.field_type != OIL_PALM:
            continue
        xy = row_points(f.geometry, float(f.row_angle_deg), float(f.row_spacing_m))
        if len(xy):
            cols, rows = inverse @ (xy[:, 0], xy[:, 1])
            pts.append(np.column_stack([rows, cols]))
    if not pts:
        return None
    return row_objects(np.concatenate(pts), transform.a, seed)


def _roof_base(
    fid: str, palette: Palette | None, material: str | None = None
) -> tuple[float, float, float]:
    """A building's base roof colour: its recorded material, else the palette or default set."""
    if material in ("tin", "tile"):
        rust = int(fid[16:20], 16) / 65535 if material == "tin" else 0.0
        base = TIN_ROOF if material == "tin" else TILE_ROOF
        return (base[0] - 28.0 * rust, base[1] - 60.0 * rust, base[2] - 70.0 * rust)
    if palette is None or not palette.roofs:
        return ROOF_MATERIALS[int(fid[:8], 16) % len(ROOF_MATERIALS)]
    draw = int(fid[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for colour, share in palette.roofs:
        edge += share
        if draw <= edge:
            return colour
    return palette.roofs[-1][0]


def roof_table(
    ids: Any,
    variation: float,
    bright_fraction: float = 0.0,
    palette: Palette | None = None,
    materials: Any = None,
) -> Any:
    """Roof colour per building (row i -> entry i + 1; entry 0 is unused), from each stable id.

    Material and brightness come from the building's id, so a building keeps its roof however
    the rest of the world is generated.
    """
    table = np.zeros((len(ids) + 1, 3), dtype="float32")
    for i, fid in enumerate(ids):
        material = None if materials is None else str(materials[i])
        base = np.array(_roof_base(str(fid), palette, material))
        brightness = 1.0 + variation * (int(str(fid)[8:12], 16) / 65535 * 2 - 1)
        if int(str(fid)[12:16], 16) / 65535 < bright_fraction:
            brightness *= BRIGHT_ROOF_GAIN
        table[i + 1] = base * brightness
    return table


def texture_field(seed: int, shape: tuple[int, int], gsd_m: float) -> Any:
    """Fractal noise in roughly [-1, 1] over the whole render grid, so tiles join up."""
    field = np.zeros(shape, dtype="float32")
    for n, (cell_m, weight) in enumerate(TEXTURE_OCTAVES):
        cell_px = max(cell_m / gsd_m, 1.0)
        lattice = substream(seed, f"texture:{n}").uniform(-1, 1, size=lattice_shape(shape, cell_px))
        field += weight * value_noise(lattice, cell_px, shape)
    return field / sum(w for _, w in TEXTURE_OCTAVES)


def class_gains(seed: int, variation: float) -> Any:
    """Per land-cover class, per channel multiplier for one scene (shape (256, 3))."""
    gains = np.ones((256, 3), dtype="float32")
    if variation <= 0:
        return gains
    for code in LANDCOVER_RGB:
        draw = np.clip(substream(seed, f"reflectance:class:{code}").normal(size=4), -2, 2)
        gains[code] = 1.0 + variation * (draw[3] + 0.5 * draw[:3])
    return gains


def transmittance(optical_depth: float, sun_elevation_deg: float) -> float:
    """Share of surface light that survives the haze: falls as optical depth grows."""
    if optical_depth < 0:
        raise ValueError("optical depth must not be negative")
    return math.exp(-optical_depth / math.sin(math.radians(sun_elevation_deg)))


def apply_haze(radiance: Any, optical_depth: float, sun_elevation_deg: float) -> Any:
    """Blend (H, W, 3) radiance toward the haze colour: ``r * t + haze * (1 - t)``."""
    t = transmittance(optical_depth, sun_elevation_deg)
    return radiance * t + np.array(HAZE_RGB, dtype="float32") * (1.0 - t)


def shade(
    landcover: Any,
    building_mask: Any,
    shadow: Any,
    *,
    gains: Any = None,
    roofs: Any = None,
    tone: tuple[float, float, float] = (1.0, 1.0, 1.0),
    texture: Any = None,
    haze: tuple[float, float] = (0.0, 50.0),
    ambient: float = AMBIENT,
    contrast: float = 1.0,
    scene_objects: Any = None,
    roof_gain: Any = None,
    palette: Palette | None = None,
) -> Any:
    """Per-class reflectance, darkened inside cast shadows. Returns (3, H, W) uint8.

    ``shadow`` may be a soft 0-1 mask. ``haze`` is (optical depth, sun elevation in degrees).
    ``gains`` scales each class per channel and ``roofs`` colours buildings individually (given
    as a (H, W, 3) array); both default to the flat V1 look.
    """
    table = np.zeros((256, 3), dtype="float32")
    for code, rgb in LANDCOVER_RGB.items():
        table[code] = rgb
    if palette is not None:
        for code, rgb in palette.classes.items():
            table[code] = rgb
    table[0] = (255.0, 0.0, 255.0)  # unmapped land cover is loud, not silent
    if gains is not None:
        table = table * gains
    albedo = table[landcover]
    if texture is not None:
        amp = np.ones(256, dtype="float32")
        for code, value in TEXTURE_CLASS_AMP.items():
            amp[code] = value
        modulation = texture * amp[landcover]
        modulation[building_mask > 0] = texture[building_mask > 0] * ROOF_TEXTURE_AMP
        albedo = albedo * (1.0 + modulation)[..., None]
    if roofs is None:
        albedo[building_mask > 0] = ROOF_RGB
    else:
        albedo[building_mask > 0] = roofs[building_mask > 0]
    if roof_gain is not None:
        albedo = albedo * roof_gain[..., None]
    if scene_objects is not None:  # (colour (H, W, 3), coverage (H, W)) of trees and vehicles
        covered = scene_objects[1] > 0
        albedo[covered] = scene_objects[0][covered]
    light = (1.0 - (1.0 - ambient) * np.clip(shadow, 0, 1)).astype("float32")[..., None]
    radiance = albedo * light
    if haze[0] > 0:
        radiance = apply_haze(radiance, haze[0], haze[1])
    graded = radiance * np.array(tone, dtype="float32")
    if contrast != 1.0:  # stretch about each channel's own mean, so dark channels (blue) survive
        pivot = graded.reshape(-1, 3).mean(axis=0)
        graded = (graded - pivot) * contrast + pivot
    rgb = np.clip(np.rint(graded), 0, 255).astype("uint8")
    return np.moveaxis(rgb, -1, 0)


def _write_tile(path: Path, data: Any, transform: Affine, epsg: int) -> None:
    """Write a (bands, H, W) or (H, W) array as a deflate-compressed GeoTIFF."""
    bands = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype=bands.dtype.name,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(bands)


def _validate_tile(
    paths: list[Path],
    mask: Any,
    geometries: list[Any],
    transform: Affine,
    shape: tuple[int, int],
    ignore: Any = None,
) -> str | None:
    """Why a tile must be dropped, or None. Checks alignment, then the independent mask.

    Pixels in ``ignore`` (a no-data region) must hold no label and are not compared.
    """
    seen = set()
    for path in paths:
        with rasterio.open(path) as src:
            seen.add((src.crs.to_string(), tuple(src.transform)[:6], src.height, src.width))
    if len(seen) != 1:
        return "image and masks differ in CRS, transform or size"
    expected = independent_building_mask(geometries, transform, shape)
    differs = expected != mask
    if ignore is not None:
        if mask[ignore].any():
            return "a no-data region carries building labels"
        differs = differs & ~ignore
    wrong = int(differs.sum())
    if wrong:
        return f"building mask differs from the independent rasterisation in {wrong} pixels"
    return None


def render_region(
    terrain_dir: Path,
    buildings: gpd.GeoDataFrame,
    center_lonlat: tuple[float, float],
    out_dir: Path,
    config: RenderConfig | None = None,
    roads: gpd.GeoDataFrame | None = None,
    fields: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    """Render a region. ``roads`` and ``fields`` are painted on the ground; labels ignore them."""
    config = config or RenderConfig()
    epsg, transform, (tiles, side) = region_grid(center_lonlat, config)
    dem = _warp(terrain_dir / "dem.tif", transform, epsg, side, Resampling.bilinear)
    landcover = _warp(terrain_dir / "landcover.tif", transform, epsg, side, Resampling.nearest)
    local = buildings.to_crs(epsg)
    height, mask, index = _building_rasters(local, transform, side)
    surface = landcover
    road = None
    if roads is not None and len(roads):
        road = _road_raster(roads.to_crs(epsg), transform, side)
        road[mask > 0] = 0
        surface = np.where(road > 0, road, landcover).astype("uint8")
    field_trees = None
    if fields is not None and len(fields):
        local_fields = fields.to_crs(epsg)
        painted = _field_raster(local_fields, transform, side)
        painted[mask > 0] = 0
        if road is not None:
            painted[road > 0] = 0
        surface = np.where(painted > 0, painted, surface).astype("uint8")
        field_trees = _field_trees(local_fields, transform, side, config.seed)
    cars_on = (
        None if road is None else np.where(np.isin(road, (ROAD_LOCAL, ROAD_ARTERIAL)), road, 0)
    )
    scene_objects = None
    if config.objects > 0:
        obj_height, obj_colour, obj_covered = scatter(
            surface, mask, config.gsd_m, config.objects, config.seed, cars_on, field_trees
        )
        height = height + obj_height
        scene_objects = (obj_colour, obj_covered)
    roof_gain = None
    if config.roof_detail > 0:
        roof_gain = roof_pattern(index, local["id"].to_numpy(), config.gsd_m, config.roof_detail)
    shadow = shadow_mask(
        dem + height, config.gsd_m, config.sun_azimuth_deg, config.sun_elevation_deg
    )
    if config.penumbra_m > 0:
        shadow = gaussian_blur(shadow.astype("float32"), config.penumbra_m / config.gsd_m)
    haze = (config.haze, config.sun_elevation_deg)
    texture = None
    if config.texture > 0:
        texture = config.texture * texture_field(config.seed, (side, side), config.gsd_m)
    common: dict[str, Any] = {
        "tone": config.tone,
        "texture": texture,
        "haze": haze,
        "ambient": config.ambient,
        "contrast": config.contrast,
        "scene_objects": scene_objects,
        "roof_gain": roof_gain,
        "palette": config.palette,
    }
    if config.variation > 0:
        roofs = roof_table(
            local["id"].to_numpy(),
            config.variation,
            config.bright_roofs,
            config.palette,
            local["roof_material"].to_numpy() if "roof_material" in local else None,
        )[index]
        gains = class_gains(config.seed, config.variation)
        rgb = shade(surface, mask, shadow, gains=gains, roofs=roofs, **common)
    else:
        rgb = shade(surface, mask, shadow, **common)

    rgb = apply_psf(rgb, config.psf_sigma_px)

    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    geometries = list(local.geometry)
    records: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    n = config.tile_px
    for row in range(tiles):
        for col in range(tiles):
            window = (slice(row * n, (row + 1) * n), slice(col * n, (col + 1) * n))
            tile_transform = transform @ Affine.translation(col * n, row * n)
            name = f"r{row:02d}_c{col:02d}"
            tile_mask = np.ascontiguousarray(mask[window])
            tile_rgb = add_noise(
                rgb[:, window[0], window[1]], config.noise_sigma, config.seed, name
            )
            tile_land = np.ascontiguousarray(landcover[window])
            blank = blank_wedge(
                (n, n), config.blank_prob, config.blank_max_fraction, config.seed, name
            )
            if blank.any():  # no data: black image, no labels, unknown land cover
                tile_rgb = tile_rgb.copy()
                tile_rgb[:, blank] = 0
                tile_mask[blank] = 0
                tile_land[blank] = 0
            paths = {
                "": (tile_dir / f"{name}.tif", tile_rgb),
                "_buildings": (tile_dir / f"{name}_buildings.tif", tile_mask),
                "_landcover": (tile_dir / f"{name}_landcover.tif", tile_land),
            }
            for path, data in paths.values():
                _write_tile(path, data, tile_transform, epsg)
            reason = _validate_tile(
                [p for p, _ in paths.values()],
                tile_mask,
                geometries,
                tile_transform,
                (n, n),
                ignore=blank,
            )
            if reason:
                for path, _ in paths.values():
                    path.unlink()
                dropped.append({"tile": name, "reason": reason})
                continue
            records.append(
                {
                    "tile": name,
                    "transform": list(tile_transform)[:6],
                    "building_pixels": int(tile_mask.sum()),
                    "mask_sha256": hashlib.sha256(tile_mask.tobytes()).hexdigest(),
                }
            )
    manifest: dict[str, Any] = {
        "schema": 1,
        "crs": f"EPSG:{epsg}",
        "gsd_m": config.gsd_m,
        "tile_px": config.tile_px,
        "tiles_per_side": tiles,
        "sun": {"azimuth_deg": config.sun_azimuth_deg, "elevation_deg": config.sun_elevation_deg},
        "seed": config.seed,
        "variation": config.variation,
        "tone": list(config.tone),
        "texture": config.texture,
        "haze": config.haze,
        "psf_sigma_px": config.psf_sigma_px,
        "noise_sigma": config.noise_sigma,
        "penumbra_m": config.penumbra_m,
        "ambient": config.ambient,
        "contrast": config.contrast,
        "bright_roofs": config.bright_roofs,
        "blank_prob": config.blank_prob,
        "objects": config.objects,
        "roof_detail": config.roof_detail,
        "palette": config.palette is not None,
        "buildings": len(local),
        "tiles": records,
        "dropped": dropped,
    }
    (out_dir / "render.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
