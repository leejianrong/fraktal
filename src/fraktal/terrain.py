"""Fetch, clip and cache the DEM and land cover for an AOI (ADR-0002).

Both layers are Cloud-Optimized GeoTIFFs on public S3, so only the AOI window is read. They are
written on the WorldCover 10 m grid, so the two rasters share CRS, transform and shape. A missing
or unreadable source tile is a hard error: a flat or empty DEM must never stand in for it.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import reproject

Bounds = tuple[float, float, float, float]  # left, bottom, right, top, EPSG:4326
UrlFn = Callable[[Bounds], list[str]]
MARKER = ".complete"

DEM_ATTRIBUTION = (
    "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved."
)
WORLDCOVER_ATTRIBUTION = (
    "ESA WorldCover 10 m 2021 v200 © ESA WorldCover project / Contains modified Copernicus "
    "Sentinel data (2021) processed by ESA WorldCover consortium. CC BY 4.0."
)
LAYERS = {
    "dem": {
        "file": "dem.tif",
        "source": "copernicus-dem-glo30",
        "licence": "Copernicus DEM licence (free, attribution required)",
        "attribution": DEM_ATTRIBUTION,
    },
    "landcover": {
        "file": "landcover.tif",
        "source": "esa-worldcover-2021-v200",
        "licence": "CC-BY-4.0",
        "attribution": WORLDCOVER_ATTRIBUTION,
    },
}


class TerrainError(OSError):
    """A source tile is missing or unreadable, or the result has gaps."""


def _tag(value: int, positive: str, negative: str, width: int) -> str:
    return f"{positive if value >= 0 else negative}{abs(value):0{width}d}"


def dem_tile_urls(bounds: Bounds) -> list[str]:
    left, bottom, right, top = bounds
    urls = []
    for lat in range(math.floor(bottom), math.ceil(top)):
        for lon in range(math.floor(left), math.ceil(right)):
            name = (
                f"Copernicus_DSM_COG_10_{_tag(lat, 'N', 'S', 2)}_00_{_tag(lon, 'E', 'W', 3)}_00_DEM"
            )
            urls.append(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif")
    return urls


def worldcover_tile_urls(bounds: Bounds) -> list[str]:
    left, bottom, right, top = bounds
    urls = []
    for lat in range(math.floor(bottom / 3) * 3, math.ceil(top / 3) * 3, 3):
        for lon in range(math.floor(left / 3) * 3, math.ceil(right / 3) * 3, 3):
            tile = f"{_tag(lat, 'N', 'S', 2)}{_tag(lon, 'E', 'W', 3)}"
            urls.append(
                "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            )
    return urls


def _open_all(urls: Sequence[str]) -> list[Any]:
    opened: list[Any] = []
    try:
        for url in urls:
            try:
                opened.append(rasterio.open(url))
            except RasterioIOError as err:
                raise TerrainError(f"source tile missing or unreadable: {url} ({err})") from err
    except BaseException:
        for src in opened:
            src.close()
        raise
    return opened


def _snap_out(bounds: Bounds, origin: tuple[float, float], res: tuple[float, float]) -> Bounds:
    left, bottom, right, top = bounds
    ox, oy = origin
    rx, ry = res
    eps = 1e-9
    return (
        math.floor((left - ox) / rx + eps) * rx + ox,
        oy - math.ceil((oy - bottom) / ry - eps) * ry,
        math.ceil((right - ox) / rx - eps) * rx + ox,
        oy - math.floor((oy - top) / ry + eps) * ry,
    )


def _write_cog(
    path: Path, array: np.ndarray[Any, Any], transform: Affine, tags: dict[str, str]
) -> None:
    profile = {
        "driver": "GTiff",
        "count": 1,
        "height": array.shape[0],
        "width": array.shape[1],
        "dtype": array.dtype.name,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    tmp = path.with_name(path.name + ".tmp")
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(array, 1)
            dst.update_tags(**tags)
        with mem.open() as src:
            rasterio.shutil.copy(src, tmp, driver="COG", compress="deflate")
    tmp.rename(path)


@dataclass(frozen=True)
class TerrainResult:
    status: str  # "cached" or "fetched"
    path: Path
    shape: tuple[int, int]
    bounds: Bounds


def ingest_terrain(
    data_dir: Path,
    aoi: Bounds,
    *,
    dem_urls: UrlFn = dem_tile_urls,
    worldcover_urls: UrlFn = worldcover_tile_urls,
) -> TerrainResult:
    target = data_dir / "terrain" / "khartoum"
    marker = target / MARKER
    if marker.is_file():
        info = json.loads(marker.read_text())
        if info["aoi"] == list(aoi):
            return TerrainResult("cached", target, tuple(info["shape"]), tuple(info["bounds"]))

    wc_sources = _open_all(worldcover_urls(aoi))
    dem_sources: list[Any] = []
    try:
        dem_sources = _open_all(dem_urls(aoi))
        first = wc_sources[0]
        res = first.res
        bounds = _snap_out(aoi, (first.transform.c, first.transform.f), res)
        landcover, transform = merge(wc_sources, bounds=bounds, res=res, nodata=0)
        landcover = landcover[0]
        if (landcover == 0).any():
            raise TerrainError("land cover has gaps inside the AOI")
        dem_native, dem_transform = merge(dem_sources, bounds=bounds, nodata=np.nan)
        dem_nodata = dem_sources[0].nodata
        dem_native = dem_native[0].astype("float32")
        if dem_nodata is not None:
            dem_native[dem_native == dem_nodata] = np.nan
        if np.isnan(dem_native).any():
            raise TerrainError("DEM has gaps inside the AOI")
        dem = np.full(landcover.shape, np.nan, dtype="float32")
        reproject(
            dem_native,
            dem,
            src_transform=dem_transform,
            src_crs="EPSG:4326",
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )
        if np.isnan(dem).any():
            raise TerrainError("DEM does not cover the whole AOI")
    finally:
        for src in [*wc_sources, *dem_sources]:
            src.close()

    staging = data_dir / "terrain" / "khartoum.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    layers = {"dem": dem, "landcover": landcover}
    for name, array in layers.items():
        meta = LAYERS[name]
        tags = {k: meta[k] for k in ("source", "licence", "attribution")}
        _write_cog(staging / meta["file"], array, transform, tags)
    provenance = {
        "aoi": list(aoi),
        "bounds": list(bounds),
        "shape": list(landcover.shape),
        "crs": "EPSG:4326",
        "layers": LAYERS,
    }
    (staging / MARKER).write_text(json.dumps(provenance, indent=2))
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return TerrainResult("fetched", target, landcover.shape, bounds)
