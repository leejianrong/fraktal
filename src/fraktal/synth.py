"""Synthetic pretraining chips: render regions of the skeleton world and cut them into chips.

A chip is kept only if it lies inside the labelled training tiles and clear of every held-out test
tile. Elsewhere real buildings exist but are not in the world, so the mask would wrongly call them
background.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_origin
from shapely.geometry import box
from shapely.strtree import STRtree

from fraktal.ids import substream
from fraktal.log import emit
from fraktal.procedural import ProceduralConfig, generate_world, read_buildings, read_roads
from fraktal.randomise import Ranges, sample_config
from fraktal.render import RenderConfig, render_region
from fraktal.split import read_manifest
from fraktal.terrain import TerrainError


def pick_centers(
    manifest_path: Path, seed: int, size_m: float = 2000.0
) -> list[tuple[float, float]]:
    """Region centres, best-labelled first, at least ``size_m`` apart.

    SpaceNet labels only cover its tiles, so a region is worth rendering in proportion to how
    much of it lies inside training tiles. Ties break on a seeded order.
    """
    manifest = read_manifest(manifest_path)
    train = set(manifest["train"])
    tiles = [t for t in manifest.get("tile_bounds", []) if t["id"] in train]
    if not tiles:
        raise ValueError("split manifest has no tile_bounds; re-run `fraktal split`")
    boxes = [box(*t["bounds"]) for t in tiles]
    bounds = np.array([t["bounds"] for t in tiles])
    centres = np.column_stack(
        [(bounds[:, 0] + bounds[:, 2]) / 2, (bounds[:, 1] + bounds[:, 3]) / 2]
    )
    lat = float(np.mean(centres[:, 1]))
    half_lat = size_m / 2 / 111_320.0
    half_lon = half_lat / math.cos(math.radians(lat))
    tree = STRtree(boxes)
    rank = substream(seed, "synthetic-regions").permutation(len(tiles))
    scored = []
    for i, (x, y) in enumerate(centres):
        window = box(x - half_lon, y - half_lat, x + half_lon, y + half_lat)
        covered = sum(window.intersection(boxes[j]).area for j in tree.query(window))
        scored.append((-round(covered / window.area, 3), int(rank[i]), float(x), float(y)))
    scored.sort()
    chosen: list[tuple[float, float]] = []
    for _score, _tie, x, y in scored:
        far = all(abs(x - cx) > 2 * half_lon or abs(y - cy) > 2 * half_lat for cx, cy in chosen)
        if far:
            chosen.append((x, y))
    return chosen


PARAM_KEYS = (
    "seed",
    "variation",
    "tone",
    "texture",
    "haze",
    "psf_sigma_px",
    "noise_sigma",
    "penumbra_m",
    "sun",
)


def manifest_params(manifest: dict[str, Any]) -> dict[str, Any]:
    """The realism parameters a region was rendered with, from its render manifest."""
    return {k: manifest[k] for k in PARAM_KEYS if k in manifest}


def _png(path: Path, rgb: np.ndarray[Any, Any]) -> None:
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=rgb.shape[2],
            height=rgb.shape[1],
            count=3,
            dtype="uint8",
            transform=from_origin(1, 1, 1, 1),  # PNG has no georeferencing; silence the warning
        ) as dst:
            dst.write(rgb)
        with mem.open() as src:
            rasterio.shutil.copy(src, path, driver="PNG")


def synthesize(
    terrain_dir: Path,
    layer: gpd.GeoDataFrame | None,
    centers: list[tuple[float, float]],
    regions: int,
    test_footprints: list[Any],
    train_union: Any,
    out_dir: Path,
    config: RenderConfig,
    samples: int = 6,
    min_coverage: float = 0.95,
    ranges: Ranges | None = None,
    seed: int = 0,
    procedural: ProceduralConfig | None = None,
    max_chips: int | None = None,
) -> dict[str, Any]:
    """Render ``regions`` regions and keep the chips clear of the held-out test tiles.

    With ``procedural`` set, every region gets its own generated world (buildings, roads, parcels)
    over the real terrain, so labels are complete everywhere and the training-tile coverage filter
    is skipped. ``max_chips`` thins the chips evenly per region to keep sets comparable.
    """
    if layer is None and procedural is None:
        raise ValueError("synthesize needs a building layer or a procedural config")
    chips_dir = out_dir / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    dropped: list[dict[str, str]] = []
    rendered = 0
    region_configs: list[dict[str, Any]] = []
    for center in centers:
        if rendered == regions:
            break
        region_dir = out_dir / f"region_{rendered:02d}"
        emit("synth_region_start", region=rendered, of=regions, center=list(center))
        try:
            scene = sample_config(config, ranges, seed, rendered) if ranges else config
            if procedural is not None:
                world_dir = out_dir / "worlds" / f"world_{rendered:02d}"
                generate_world(world_dir, center, seed * 1000 + rendered, procedural, terrain_dir)
                manifest = render_region(
                    world_dir,
                    read_buildings(world_dir),
                    center,
                    region_dir,
                    scene,
                    roads=read_roads(world_dir),
                )
            else:
                assert layer is not None
                manifest = render_region(terrain_dir, layer, center, region_dir, scene)
        except TerrainError:
            continue  # region sticks out of the terrain AOI; try the next candidate
        region_configs.append(manifest_params(manifest))
        rendered += 1
        emit("synth_region_done", region=rendered - 1, chips_so_far=len(ids))
        recs = manifest["tiles"]
        if max_chips is not None and len(recs) > math.ceil(max_chips / regions):
            pick = np.linspace(0, len(recs) - 1, math.ceil(max_chips / regions)).round()
            recs = [recs[int(i)] for i in pick]
        for rec in recs:
            tile_transform = Affine(*rec["transform"])
            n = config.tile_px
            footprint = box(
                tile_transform.c,
                tile_transform.f + tile_transform.e * n,
                tile_transform.c + tile_transform.a * n,
                tile_transform.f,
            )
            chip_id = f"r{rendered - 1:02d}_{rec['tile']}"
            if any(footprint.intersects(t) and not footprint.touches(t) for t in test_footprints):
                dropped.append({"chip": chip_id, "reason": "overlaps a held-out test tile"})
                continue
            if (
                procedural is None
                and footprint.intersection(train_union).area < min_coverage * footprint.area
            ):
                dropped.append({"chip": chip_id, "reason": "outside labelled training tiles"})
                continue
            with rasterio.open(region_dir / "tiles" / f"{rec['tile']}.tif") as src:
                image = src.read()
            with rasterio.open(region_dir / "tiles" / f"{rec['tile']}_buildings.tif") as src:
                mask = src.read(1)
            np.save(chips_dir / f"{chip_id}_img.npy", image)
            np.save(chips_dir / f"{chip_id}_mask.npy", mask)
            ids.append(chip_id)
    if rendered < regions:
        raise TerrainError(f"only {rendered} of {regions} regions fit inside the terrain AOI")
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    for chip_id in ids[:: max(1, len(ids) // samples)][:samples]:
        image = np.load(chips_dir / f"{chip_id}_img.npy")
        mask = np.load(chips_dir / f"{chip_id}_mask.npy")
        side_by_side = np.concatenate([image, np.repeat((mask * 255)[None], 3, axis=0)], axis=2)
        _png(sample_dir / f"{chip_id}.png", side_by_side)
    meta = {
        "schema": 1,
        "source": "synthetic",
        "tiles": sorted(ids),
        "regions": regions,
        "region_params": region_configs,
        "randomised": ranges is not None,
        "procedural": procedural is not None,
        "dropped": dropped,
        "gsd_m": config.gsd_m,
        "tile_px": config.tile_px,
    }
    (chips_dir / "chips.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
