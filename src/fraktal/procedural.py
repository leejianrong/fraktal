"""Generate a whole world from a seed (V2): terrain, land cover, roads, parcels and buildings.

Where real terrain is supplied it is kept (the DEM gains fine detail below its resolution, and
real land-cover classes win); everything else is procedural. Each layer records where it came
from, and the manifest hashes every file so two runs can be compared byte for byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from fraktal import settlement, world
from fraktal.coast import UNBUILDABLE, CoastConfig, box_any, paint_coast, shape_coast
from fraktal.fields import FieldConfig, access_tracks, build_fields, level_paddies
from fraktal.landcover_gen import generate_landcover
from fraktal.log import emit
from fraktal.render import _warp, utm_epsg
from fraktal.roads import RoadConfig, build_roads
from fraktal.terrain_gen import add_detail, fractal_terrain

GENERATOR_VERSION = "procedural-0.1"
DEFAULT_BIOME = "khartoum"
MANIFEST = "world.json"
# Config fields added after the first worlds: left out of the world id while at their default,
# so those worlds keep their ids.
LATER_FIELDS = (
    ("buildings", "width_share"),
    ("buildings", "depth_share"),
    ("buildings", "roof_materials"),
    ("buildings", "tin_share"),
    ("parcels", "skew_deg"),
    ("parcels", "blocks"),
    ("parcels", "block_depth_m"),
)


@dataclass(frozen=True)
class ProceduralConfig:
    biome: str = DEFAULT_BIOME  # a name from fraktal.biomes; recorded in the manifest
    size_m: float = 2000.0
    gsd_m: float = 5.0
    relief_m: float = 40.0
    aridity: float = 0.5
    roads: RoadConfig = field(default_factory=RoadConfig)
    parcels: settlement.ParcelConfig = field(default_factory=settlement.ParcelConfig)
    buildings: settlement.BuildingConfig = field(default_factory=settlement.BuildingConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    coast: CoastConfig = field(default_factory=CoastConfig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raster(path: Path, data: Any, transform: Any, epsg: int) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype.name,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)


def generate_world(
    out_dir: Path,
    center_lonlat: tuple[float, float],
    seed: int,
    config: ProceduralConfig | None = None,
    real_terrain: Path | None = None,
) -> dict[str, Any]:
    """Write ``dem.tif``, ``landcover.tif``, road, parcel and building layers and ``world.json``."""
    config = config or ProceduralConfig()
    lon, lat = center_lonlat
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    side = max(16, int(round(config.size_m / config.gsd_m)))
    half = side * config.gsd_m / 2
    west = math.floor((x - half) / config.gsd_m) * config.gsd_m
    north = math.ceil((y + half) / config.gsd_m) * config.gsd_m
    transform = from_origin(west, north, config.gsd_m, config.gsd_m)
    crs = f"EPSG:{epsg}"
    emit("generate_start", seed=seed, side_px=side, real_terrain=real_terrain is not None)

    real_land = None
    coast = None
    if real_terrain is not None:
        base = _warp(real_terrain / "dem.tif", transform, epsg, side, Resampling.bilinear)
        real_land = _warp(real_terrain / "landcover.tif", transform, epsg, side, Resampling.nearest)
        dem = add_detail(base, config.gsd_m, seed)
        terrain_origin = "real:copernicus-dem+procedural:detail"
        land_origin = "real:esa-worldcover+procedural:fill"
    else:
        dem = fractal_terrain((side, side), config.gsd_m, seed, relief_m=config.relief_m) + 300.0
        if config.coast.sea_share > 0:
            dem, coast = shape_coast(dem, config.gsd_m, seed, config.coast)
        terrain_origin = land_origin = "procedural:fractal"
    land = generate_landcover(dem, config.gsd_m, seed, real=real_land, aridity=config.aridity)
    if coast is not None:
        land = paint_coast(land, coast)
    emit("generate_stage", stage="terrain_and_landcover_done")
    roads = build_roads(dem, land, config.gsd_m, transform, crs, seed, config.roads, world="roads")
    parcels = settlement.build_parcels(roads, land, transform, seed, config.parcels)
    lines = {r.id: r.geometry for _, r in roads.iterrows()} if config.parcels.blocks else None
    buildings = settlement.build_buildings(
        parcels,
        settlement.arterial_centres(roads),
        seed,
        config.buildings,
        road_lines=lines,
    )
    fields = build_fields(
        land,
        transform,
        crs,
        seed,
        config.fields,
        [roads_wide(roads), parcels, buildings],
        dem=dem,
        pond_zone=None if coast is None else pond_zone(coast, land),
    )
    if fields is not None:
        dem = level_paddies(dem, transform, fields)
        roads = with_tracks(roads, access_tracks(fields, config.fields.track_width_m), config)
    emit(
        "generate_stage",
        stage="settlement_done",
        roads=len(roads),
        buildings=len(buildings),
        fields=0 if fields is None else len(fields),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_raster(out_dir / "dem.tif", dem.astype("float32"), transform, epsg)
    _write_raster(out_dir / "landcover.tif", land, transform, epsg)
    layers = {"road": roads, "parcel": parcels, "building": buildings}
    if fields is not None:
        layers["field"] = fields
    manifest: dict[str, Any] = {
        "schema": 1,
        "world_id": world.world_id(seed, _jsonable(config), GENERATOR_VERSION),
        "generator_version": GENERATOR_VERSION,
        "biome": config.biome,
        "drive_side": config.roads.drive_side,
        "seed": seed,
        "crs": crs,
        "center_lonlat": [lon, lat],
        "bounds": [west, north - side * config.gsd_m, west + side * config.gsd_m, north],
        "gsd_m": config.gsd_m,
        "rasters": {
            "dem": {
                "file": "dem.tif",
                "origin": terrain_origin,
                "sha256": _sha256(out_dir / "dem.tif"),
            },
            "landcover": {
                "file": "landcover.tif",
                "origin": land_origin,
                "sha256": _sha256(out_dir / "landcover.tif"),
            },
        },
        "layers": {},
    }
    for kind, frame in layers.items():
        path = out_dir / f"{kind}s.parquet"
        world.write_layer(frame, kind, path)
        manifest["layers"][kind] = {
            "file": path.name,
            "features": len(frame),
            "provenance": sorted(set(map(str, frame["provenance"]))),
            "sha256": _sha256(path),
        }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    emit("generate_done", out=str(out_dir))
    return manifest


def _jsonable(config: ProceduralConfig) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(config)))
    if out["biome"] == DEFAULT_BIOME:
        del out["biome"]  # keeps the ids of worlds made before biomes existed
    default: dict[str, Any] = json.loads(json.dumps(dataclasses.asdict(ProceduralConfig())))
    for section in ("fields", "coast"):  # whole sections added later: omitted at default
        if out[section] == default[section]:
            del out[section]
    del out["roads"]["drive_side"]  # traffic only: it does not change the world, so not its id
    for section, key in LATER_FIELDS:  # settings added after worlds existed: omitted at default
        if out[section][key] == default[section][key]:
            del out[section][key]
    return out


def pond_zone(coast: Any, land: Any) -> Any:
    """Where fish ponds may go: the coast's pond zone minus any water or wetland, shrunk a cell."""
    blocked = box_any(~coast.pond_zone | np.isin(land, UNBUILDABLE), 1)
    return ~blocked


def roads_wide(roads: Any) -> Any:
    """The road layer with each line widened to its carriageway, for keep-out tests."""
    out = roads.copy()
    out["geometry"] = [r.geometry.buffer(r.width_m / 2) for _, r in roads.iterrows()]
    return out


def with_tracks(roads: Any, tracks: list[Any], config: ProceduralConfig) -> Any:
    """``roads`` plus plantation access tracks (ids of existing roads are unchanged)."""
    if not tracks:
        return roads
    rows = [
        {
            "geometry": r.geometry,
            "provenance": r.provenance,
            "road_class": r.road_class,
            "width_m": r.width_m,
            "bridge": r.bridge,
        }
        for _, r in roads.iterrows()
    ]
    rows += [
        {
            "geometry": t,
            "provenance": "procedural:roads",
            "road_class": "track",
            "width_m": config.fields.track_width_m,
            "bridge": 0.0,
        }
        for t in tracks
    ]
    return world.make_layer("road", "roads", rows, str(roads.crs))


def read_fields(world_dir: Path) -> Any:
    """The field layer of a procedural world directory, or None if it has none."""
    path = world_dir / "fields.parquet"
    return world.read_layer(path, "field") if path.exists() else None


def read_buildings(world_dir: Path) -> Any:
    """The building layer of a procedural world directory."""
    return world.read_layer(world_dir / "buildings.parquet", "building")


def read_roads(world_dir: Path) -> Any:
    """The road layer of a procedural world directory."""
    return world.read_layer(world_dir / "roads.parquet", "road")


__all__ = [
    "DEFAULT_BIOME",
    "GENERATOR_VERSION",
    "MANIFEST",
    "ProceduralConfig",
    "generate_world",
    "read_buildings",
    "read_fields",
    "read_roads",
]
