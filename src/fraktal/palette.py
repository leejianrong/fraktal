"""Fit a render palette from the real train chips (FRK-23).

Class colours come from where WorldCover classes overlap real imagery away from buildings, and
roof colours from clustering the pixels under the building labels. Only train tiles are read, so
nothing from the held-out test tiles reaches a synthetic world (ADR-0006).

Palette colours are albedo before the renderer's tone gain: chip colour divided by ``tone``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from fraktal.chips import ChipTiles
from fraktal.ids import substream
from fraktal.log import emit

Array = np.ndarray[Any, Any]
SCHEMA = 1
MIN_CLASS_PIXELS = 2000
ROOF_CLUSTERS = 4
ROOF_SAMPLE = 200_000


class PaletteError(ValueError):
    pass


@dataclass(frozen=True)
class Palette:
    classes: dict[int, tuple[float, float, float]]
    roofs: tuple[tuple[tuple[float, float, float], float], ...]  # (albedo, share), shares sum to 1
    tone: tuple[float, float, float]


def kmeans(points: Array, k: int, seed: int, iterations: int = 20) -> tuple[Array, Array]:
    """Lloyd's algorithm with a seeded start. Returns centres (k, d) and member counts (k,)."""
    if len(points) < k:
        raise PaletteError(f"need at least {k} points to cluster, got {len(points)}")
    rng = substream(seed, "palette:roofs")
    centres = points[rng.choice(len(points), size=k, replace=False)].astype("float64")
    for _ in range(iterations):
        distance = ((points[:, None, :] - centres[None]) ** 2).sum(axis=2)
        label = distance.argmin(axis=1)
        for j in range(k):
            members = points[label == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    label = ((points[:, None, :] - centres[None]) ** 2).sum(axis=2).argmin(axis=1)
    return centres, np.bincount(label, minlength=k)


def _landcover_on(tile_path: Path, landcover_path: Path) -> Array:
    with rasterio.open(tile_path) as tile, rasterio.open(landcover_path) as land:
        out = np.zeros((tile.height, tile.width), dtype="uint8")
        reproject(
            rasterio.band(land, 1),
            out,
            dst_transform=tile.transform,
            dst_crs=tile.crs,
            resampling=Resampling.nearest,
        )
    return out


def fit(
    train: ChipTiles,
    tile_paths: dict[str, Path],
    landcover_path: Path,
    tone: tuple[float, float, float],
    seed: int = 0,
    max_tiles: int = 150,
) -> Palette:
    """Palette from (a deterministic sample of) the train chips. ``train`` must be train only."""
    step = max(1, len(train) // max_tiles)
    sums: dict[int, Array] = {}
    counts: dict[int, int] = {}
    roof_pixels: list[Array] = []
    emit("palette_start", tiles=len(train), sampled=len(train.ids[::step]))
    for index in range(0, len(train), step):
        image, mask = train.get(index)
        land = _landcover_on(tile_paths[train.ids[index]], landcover_path)
        valid = (image > 0).any(axis=0)  # black is no data
        flat = image.reshape(3, -1).T.astype("float64")
        for code in np.unique(land[valid & (mask == 0)]):
            if code == 0:
                continue
            sel = ((land == code) & valid & (mask == 0)).ravel()
            sums[int(code)] = sums.get(int(code), 0) + flat[sel].sum(axis=0)
            counts[int(code)] = counts.get(int(code), 0) + int(sel.sum())
        roof = (valid & (mask > 0)).ravel()
        if roof.any():
            roof_pixels.append(flat[roof][::7])
    classes = {
        code: tuple(float(v) for v in sums[code] / counts[code] / np.array(tone))
        for code in sums
        if counts[code] >= MIN_CLASS_PIXELS
    }
    pixels = np.concatenate(roof_pixels) if roof_pixels else np.zeros((0, 3))
    if len(pixels) > ROOF_SAMPLE:
        pixels = pixels[substream(seed, "palette:sample").choice(len(pixels), ROOF_SAMPLE, False)]
    centres, members = kmeans(pixels, ROOF_CLUSTERS, seed)
    order = np.argsort(centres.sum(axis=1))
    roofs = tuple(
        (tuple(float(v) for v in centres[j] / np.array(tone)), float(members[j] / members.sum()))
        for j in order
        if members[j] > 0
    )
    emit("palette_done", classes=len(classes), roof_pixels=len(pixels))
    return Palette(classes=classes, roofs=roofs, tone=tone)  # type: ignore[arg-type]


def write(palette: Palette, path: Path, tiles: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": SCHEMA,
        "source": "train chips only",
        "tone": list(palette.tone),
        "classes": {str(k): list(v) for k, v in sorted(palette.classes.items())},
        "roofs": [{"albedo": list(c), "share": s} for c, s in palette.roofs],
        "train_tiles": sorted(tiles),
    }
    path.write_text(json.dumps(body, indent=2) + "\n")


def read(path: Path) -> Palette:
    body = json.loads(path.read_text())
    if body.get("schema") != SCHEMA:
        raise PaletteError(f"{path} is not a schema {SCHEMA} palette")
    return Palette(
        classes={int(k): tuple(v) for k, v in body["classes"].items()},
        roofs=tuple((tuple(r["albedo"]), float(r["share"])) for r in body["roofs"]),
        tone=tuple(body["tone"]),
    )
