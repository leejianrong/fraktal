"""Train and held-out test split by tile, plus the leak check (ADR-0006).

The split is a pure function of the sorted tile IDs and a seed, so it is reproducible. Test tile
footprints are stored in the manifest so a pretraining world can be checked against them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from fraktal.ids import substream

TILE_RE = re.compile(r"_(img\d+)\.tif$")
IMAGERY_DIR = "AOI_5_Khartoum_Train/RGB-PanSharpen"
SCHEMA = 1


class LeakError(Exception):
    """A pretraining world overlaps a held-out test tile."""


@dataclass(frozen=True)
class Tile:
    id: str
    bounds: tuple[float, float, float, float]  # left, bottom, right, top


def list_tiles(root: Path) -> tuple[list[Tile], str]:
    """Tiles under an extracted Khartoum directory, sorted by ID, and their CRS."""
    tiles: list[Tile] = []
    crs: set[str] = set()
    for path in sorted((root / IMAGERY_DIR).glob("*.tif")):
        match = TILE_RE.search(path.name)
        if match is None:
            continue
        with rasterio.open(path) as src:
            b = src.bounds
            crs.add(str(src.crs))
        tiles.append(Tile(match.group(1), (b.left, b.bottom, b.right, b.top)))
    if not tiles:
        raise FileNotFoundError(f"no tiles under {root / IMAGERY_DIR}; run `fraktal ingest`")
    if len(crs) != 1:
        raise ValueError(f"tiles use mixed CRS: {sorted(crs)}")
    return sorted(tiles, key=lambda t: t.id), crs.pop()


def split_tiles(
    tiles: Sequence[Tile], seed: int, test_fraction: float
) -> tuple[list[Tile], list[Tile]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    ordered = sorted(tiles, key=lambda t: t.id)
    if len({t.id for t in ordered}) != len(ordered):
        raise ValueError("duplicate tile IDs")
    if len(ordered) < 2:
        raise ValueError("need at least two tiles to split")
    order = substream(seed, "split").permutation(len(ordered))
    n_test = min(max(1, round(len(ordered) * test_fraction)), len(ordered) - 1)
    test_idx = set(order[:n_test].tolist())
    test = [t for i, t in enumerate(ordered) if i in test_idx]
    train = [t for i, t in enumerate(ordered) if i not in test_idx]
    return train, test


def build_manifest(
    tiles: Sequence[Tile], crs: str, seed: int, test_fraction: float
) -> dict[str, Any]:
    train, test = split_tiles(tiles, seed, test_fraction)
    all_tiles = sorted(tiles, key=lambda t: t.id)
    return {
        "schema": SCHEMA,
        "dataset": "spacenet2-khartoum",
        "seed": seed,
        "test_fraction": test_fraction,
        "crs": crs,
        "counts": {"train": len(train), "test": len(test), "total": len(train) + len(test)},
        "train": [t.id for t in train],
        "tile_bounds": [{"id": t.id, "bounds": list(t.bounds)} for t in all_tiles],
        "test": [{"id": t.id, "bounds": list(t.bounds)} for t in test],
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unsupported split manifest schema: {manifest.get('schema')!r}")
    return manifest


def assert_no_test_leak(geometries: Iterable[BaseGeometry], manifest: dict[str, Any]) -> None:
    """Raise ``LeakError`` if any geometry overlaps a held-out test tile.

    Geometries must be in the manifest's CRS. Touching a tile edge is not a leak.
    """
    test = manifest["test"]
    footprints = [box(*t["bounds"]) for t in test]
    tree = STRtree(footprints)
    leaks: set[str] = set()
    for geom in geometries:
        for idx in np.sort(tree.query(geom, predicate="intersects")):
            if not geom.touches(footprints[idx]):
                leaks.add(test[idx]["id"])
    if leaks:
        raise LeakError(f"world overlaps held-out test tiles: {', '.join(sorted(leaks))}")
