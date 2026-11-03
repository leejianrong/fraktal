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

import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from fraktal import settlement, world
from fraktal.landcover_gen import generate_landcover
from fraktal.log import emit
from fraktal.render import _warp, utm_epsg
from fraktal.roads import RoadConfig, build_roads
from fraktal.terrain_gen import add_detail, fractal_terrain

GENERATOR_VERSION = "procedural-0.1"
MANIFEST = "world.json"


@dataclass(frozen=True)
class ProceduralConfig:
    size_m: float = 2000.0
    gsd_m: float = 5.0
    relief_m: float = 40.0
    aridity: float = 0.5
    roads: RoadConfig = field(default_factory=RoadConfig)
    parcels: settlement.ParcelConfig = field(default_factory=settlement.ParcelConfig)
    buildings: settlement.BuildingConfig = field(default_factory=settlement.BuildingConfig)


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
    if real_terrain is not None:
        base = _warp(real_terrain / "dem.tif", transform, epsg, side, Resampling.bilinear)
        real_land = _warp(real_terrain / "landcover.tif", transform, epsg, side, Resampling.nearest)
        dem = add_detail(base, config.gsd_m, seed)
        terrain_origin = "real:copernicus-dem+procedural:detail"
        land_origin = "real:esa-worldcover+procedural:fill"
    else:
        dem = fractal_terrain((side, side), config.gsd_m, seed, relief_m=config.relief_m) + 300.0
        terrain_origin = land_origin = "procedural:fractal"
    land = generate_landcover(dem, config.gsd_m, seed, real=real_land, aridity=config.aridity)
    emit("generate_stage", stage="terrain_and_landcover_done")
    roads = build_roads(dem, land, config.gsd_m, transform, crs, seed, config.roads, world="roads")
    parcels = settlement.build_parcels(roads, land, transform, seed, config.parcels)
    buildings = settlement.build_buildings(
        parcels, settlement.arterial_centres(roads), seed, config.buildings
    )
    emit("generate_stage", stage="settlement_done", roads=len(roads), buildings=len(buildings))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_raster(out_dir / "dem.tif", dem.astype("float32"), transform, epsg)
    _write_raster(out_dir / "landcover.tif", land, transform, epsg)
    layers = {"road": roads, "parcel": parcels, "building": buildings}
    manifest: dict[str, Any] = {
        "schema": 1,
        "world_id": world.world_id(seed, _jsonable(config), GENERATOR_VERSION),
        "generator_version": GENERATOR_VERSION,
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
    return out


def read_buildings(world_dir: Path) -> Any:
    """The building layer of a procedural world directory."""
    return world.read_layer(world_dir / "buildings.parquet", "building")


__all__ = ["GENERATOR_VERSION", "MANIFEST", "ProceduralConfig", "generate_world", "read_buildings"]
