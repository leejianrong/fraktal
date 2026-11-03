"""Training chips from the extracted SpaceNet 2 Khartoum tiles.

Each tile becomes a uint8 RGB array and a 0/1 building mask rasterised onto the tile's own grid,
stored as ``.npy`` so training can memory-map them. The cache is also the small thing to ship to
a GPU machine instead of the 5 GB raw archive.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from fraktal.log import emit
from fraktal.split import IMAGERY_DIR, read_manifest

LABEL_DIR = "AOI_5_Khartoum_Train/geojson/buildings"
SCHEMA = 1
STRETCH_SAMPLE = 64


def _tile_path(root: Path, tile_id: str) -> Path:
    return root / IMAGERY_DIR / f"RGB-PanSharpen_AOI_5_Khartoum_{tile_id}.tif"


def stretch_range(root: Path, tile_ids: Sequence[str]) -> tuple[float, float]:
    """One global (low, high) for uint16 to uint8, from the 2nd and 98th percentile of a sample."""
    step = max(1, len(tile_ids) // STRETCH_SAMPLE)
    values = []
    for tile_id in sorted(tile_ids)[::step]:
        with rasterio.open(_tile_path(root, tile_id)) as src:
            data = src.read()
        values.append(data[data > 0][::17])
    flat = np.concatenate(values)
    low, high = np.percentile(flat, [2, 98])
    return float(low), float(high)


def _to_uint8(data: np.ndarray[Any, Any], low: float, high: float) -> np.ndarray[Any, Any]:
    scaled = (data.astype(np.float32) - low) / max(high - low, 1.0)
    return np.clip(np.rint(scaled * 255), 0, 255).astype(np.uint8)


def build_chips(data_dir: Path, manifest_path: Path, out_dir: Path | None = None) -> Path:
    """Write chips for every tile in the split manifest. Idempotent for an unchanged split."""
    manifest = read_manifest(manifest_path)
    root = data_dir / "spacenet2" / "khartoum"
    out = out_dir or data_dir / "spacenet2" / "chips"
    train: list[str] = list(manifest["train"])
    test: list[str] = [t["id"] for t in manifest["test"]]
    meta_path = out / "chips.json"
    key = {"schema": SCHEMA, "seed": manifest["seed"], "train": train, "test": test}
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text())
        if all(existing.get(k) == v for k, v in key.items()):
            return out
    out.mkdir(parents=True, exist_ok=True)
    low, high = stretch_range(root, [*train, *test])
    everything = [*train, *test]
    emit("chips_start", tiles=len(everything))
    for n, tile_id in enumerate(everything):
        if n and n % 200 == 0:
            emit("chips_progress", done=n, of=len(everything))
        with rasterio.open(_tile_path(root, tile_id)) as src:
            image = _to_uint8(src.read(), low, high)
            transform, shape = src.transform, src.shape
        labels = root / LABEL_DIR / f"buildings_AOI_5_Khartoum_{tile_id}.geojson"
        mask = np.zeros(shape, dtype=np.uint8)
        if labels.is_file():
            frame = gpd.read_file(labels)
            shapes = [(g, 1) for g in frame.geometry if g is not None and not g.is_empty]
            if shapes:
                mask = rasterize(shapes, shape, transform=transform, dtype="uint8")
        np.save(out / f"{tile_id}_img.npy", image)
        np.save(out / f"{tile_id}_mask.npy", mask)
    meta_path.write_text(json.dumps({**key, "stretch": [low, high]}, indent=2) + "\n")
    return out


class ChipTiles:
    """A split's chips as a ``benchmark.TileSource``, memory-mapped from disk."""

    def __init__(self, chips_dir: Path, tile_ids: Sequence[str]) -> None:
        self.dir = chips_dir
        self.ids = sorted(tile_ids)

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        tile = self.ids[index]
        image = np.load(self.dir / f"{tile}_img.npy", mmap_mode="r")
        mask = np.load(self.dir / f"{tile}_mask.npy", mmap_mode="r")
        return np.array(image), np.array(mask)


def load_splits(chips_dir: Path) -> tuple[ChipTiles, ChipTiles]:
    meta = json.loads((chips_dir / "chips.json").read_text())
    return ChipTiles(chips_dir, meta["train"]), ChipTiles(chips_dir, meta["test"])


def load_all(chips_dir: Path) -> ChipTiles:
    """Every chip in a directory (synthetic chips have no train/test split)."""
    meta = json.loads((chips_dir / "chips.json").read_text())
    return ChipTiles(chips_dir, meta["tiles"])
