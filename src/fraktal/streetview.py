"""Eye-level review: how far synthetic street frames are from real street photos (FRK-144).

Evaluation and looking only (ADR-0012): the photos under ``data/eval/ground-reference`` are never
trained on, conditioned on, committed or redistributed. These are coarse, GPU-free statistics that
steer work (ADR-0009); they do not gate it. A frame can match every number and still look wrong, so
the side-by-side sheet is the main check and the table says where to look first.

Each image gives one value per statistic. A city's photos and its synthetic frames are compared as
two samples per statistic, and the report always carries both sample sizes: a few photos cannot
support a claim about a city, so a gap with fewer than ``LOW_N`` photos is marked ``low_n``.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

Array = np.ndarray[Any, Any]

MAX_SIDE = 480  # images are block-averaged down to at most this many pixels on the long side
LOW_N = 8  # fewer photos than this: the gap is a hint, not a measurement
MIN_SD = 0.02  # the photo spread is never taken as less than this (every statistic is 0 to 1)
FLAG_Z = 2.0  # a synthetic mean further than this many photo standard deviations is flagged
STATS = (
    "sky_share",  # top half: blue or bright low-saturation pixels
    "green_share",  # excess-green pixels (vegetation)
    "warm_share",  # red-over-blue, saturated (soil, brick, rust)
    "brightness",  # mean of the brightest channel, 0 to 1
    "contrast",  # standard deviation of grey, 0 to 1
    "saturation",  # mean saturation, 0 to 1
    "edge_energy",  # mean absolute grey step between neighbours, 0 to 1
    "vertical_structure",  # share of strong edges that run vertically: walls, poles, trunks
    "road_brightness",  # mean grey of the bottom fifth, middle 60% of the width
)


@dataclass(frozen=True)
class Photo:
    path: Path
    source: str  # "commons", "mapillary" or "synthetic"
    place: str
    image: Array  # (H, W, 3) uint8, already reduced to at most MAX_SIDE


def _read(path: Path) -> Array:
    """An image as (H, W, 3) uint8, block-averaged to at most ``MAX_SIDE`` on the long side."""
    try:
        with rasterio.open(path) as src:
            raw = src.read()
    except rasterio.errors.RasterioError as err:
        raise ValueError(f"cannot read {path}: {err}") from err
    if raw.shape[0] == 1:
        raw = np.repeat(raw, 3, axis=0)
    img = np.moveaxis(raw[:3], 0, -1)
    k = -(-max(img.shape[:2]) // MAX_SIDE)
    if k > 1:
        h, w = img.shape[0] // k * k, img.shape[1] // k * k
        img = (
            img[:h, :w]
            .reshape(h // k, k, w // k, k, 3)
            .astype("float32")
            .mean(axis=(1, 3))
            .round()
            .astype("uint8")
        )
    return np.ascontiguousarray(img)


def load_reference(root: Path, city: str = "*") -> list[Photo]:
    """The photos of every place matching the glob ``city`` (``nairobi-*``), in name order."""
    photos = []
    for path in sorted(root.glob("*.jpg")):
        source, place = path.name.split("__")[:2]
        if fnmatch.fnmatch(place, city):
            photos.append(Photo(path, source, place, _read(path)))
    return photos


def load_frames(directory: Path, limit: int = 40) -> list[Photo]:
    """Up to ``limit`` PNG frames of a render, spread evenly along the sequence."""
    files = sorted(directory.glob("*.png"))
    if len(files) > limit:
        files = [files[round(k * (len(files) - 1) / (limit - 1))] for k in range(limit)]
    return [Photo(f, "synthetic", directory.name, _read(f)) for f in files]


def statistics(image: Array) -> dict[str, float]:
    """The ``STATS`` of one (H, W, 3) uint8 image."""
    rgb = image.astype("float32")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    top, bottom = rgb.max(axis=2), rgb.min(axis=2)
    sat = (top - bottom) / (top + 1e-6)
    grey = rgb.mean(axis=2)
    h, w = grey.shape
    sky = ((b > r + 10) & (b > 120)) | ((top > 175) & (sat < 0.15))
    gx = np.abs(np.diff(grey, axis=1))  # a step along x is a vertical edge
    gy = np.abs(np.diff(grey, axis=0))
    vertical, horizontal = float((gx > 20).sum()), float((gy > 20).sum())
    road = grey[int(h * 0.8) :, int(w * 0.2) : int(w * 0.8)]
    return {
        "sky_share": float(sky[: h // 2].mean()),
        "green_share": float(((2 * g - r - b > 25) & (g > 60)).mean()),
        "warm_share": float(((r > b + 25) & (sat > 0.2)).mean()),
        "brightness": float(top.mean() / 255.0),
        "contrast": float(grey.std() / 255.0),
        "saturation": float(sat.mean()),
        "edge_energy": float((gx.mean() + gy.mean()) / 2 / 255.0),
        "vertical_structure": vertical / max(vertical + horizontal, 1.0),
        "road_brightness": float(road.mean() / 255.0),
    }


def wasserstein(a: Sequence[float] | Array, b: Sequence[float] | Array) -> float:
    """Wasserstein-1 distance between two 1D samples (mean absolute quantile difference)."""
    q = np.linspace(0.0, 1.0, 101)
    return float(np.abs(np.quantile(a, q) - np.quantile(b, q)).mean())


def compare(
    real: Sequence[Photo],
    synthetic: Sequence[Photo],
    extra: Callable[[Sequence[Photo]], list[dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Per statistic (``STATS`` and whatever ``extra`` adds): both samples' mean and spread, the
    distance in photo standard deviations (at least ``MIN_SD``), and whether the synthetic mean
    is further than ``FLAG_Z`` of them."""
    if not real or not synthetic:
        raise ValueError("need at least one photo and one synthetic frame")
    ra = [statistics(p.image) for p in real]
    sa = [statistics(p.image) for p in synthetic]
    if extra:  # more per-image statistics, such as class shares from a segmenter
        ra = [{**a, **e} for a, e in zip(ra, extra(real), strict=True)]
        sa = [{**a, **e} for a, e in zip(sa, extra(synthetic), strict=True)]
    rows: dict[str, Any] = {}
    for name in ra[0]:
        rv, sv = np.array([s[name] for s in ra]), np.array([s[name] for s in sa])
        spread = max(float(rv.std()), MIN_SD)
        z = (float(sv.mean()) - float(rv.mean())) / spread
        rows[name] = {
            "photo_mean": round(float(rv.mean()), 4),
            "photo_sd": round(float(rv.std()), 4),
            "synthetic_mean": round(float(sv.mean()), 4),
            "synthetic_sd": round(float(sv.std()), 4),
            "w1_over_sd": round(wasserstein(rv, sv) / spread, 3),
            "z": round(z, 2),
            "flag": abs(z) > FLAG_Z,
        }
    return {
        "n_photos": len(real),
        "n_synthetic": len(synthetic),
        "low_n": len(real) < LOW_N,
        "stats": rows,
    }


def sheet(real: Sequence[Photo], synthetic: Sequence[Photo], path: Path, cell: int = 240) -> None:
    """A PNG: the synthetic frames in the top rows, the photos below, ``cell`` px per image,
    cropped to a square window. Row order is written to ``path.with_suffix('.json')``."""

    def tile(img: Array) -> Array:
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = img[y0 : y0 + side, x0 : x0 + side].astype("float32")
        k = max(1, side // cell)
        n = side // k * k
        small = crop[:n, :n].reshape(n // k, k, n // k, k, 3).mean(axis=(1, 3))
        out = np.zeros((cell, cell, 3), dtype="uint8")
        m = min(cell, small.shape[0])
        out[:m, :m] = small[:m, :m].round().astype("uint8")
        return out

    cols = 6
    groups = [list(synthetic), list(real)]
    rows: list[Array] = []
    names: list[list[str]] = []
    for group in groups:
        for start in range(0, len(group), cols):
            chunk = group[start : start + cols]
            tiles = [tile(p.image) for p in chunk]
            tiles += [np.zeros_like(tiles[0])] * (cols - len(tiles))
            rows.append(np.concatenate(tiles, axis=1))
            names.append([f"{p.source}:{p.path.name}" for p in chunk])
    image = np.concatenate(rows, axis=0)
    with rasterio.open(
        path, "w", driver="PNG", height=image.shape[0], width=image.shape[1], count=3,
        dtype="uint8",
    ) as dst:  # fmt: skip
        dst.write(np.moveaxis(image, -1, 0))
    path.with_suffix(".json").write_text(json.dumps({"cell_px": cell, "rows": names}) + "\n")


def table(report: dict[str, Any]) -> str:
    """The report as aligned text, flagged statistics marked with ``*``."""
    lines = [
        f"{report['n_photos']} photos, {report['n_synthetic']} synthetic frames"
        + (" (low n: a hint, not a measurement)" if report["low_n"] else ""),
        f"{'statistic':<20}{'photos':>14}{'synthetic':>14}{'z':>8}",
    ]
    for name, r in report["stats"].items():
        mark = "*" if r["flag"] else " "
        lines.append(
            f"{name:<20}{r['photo_mean']:>9.3f}±{r['photo_sd']:<4.2f}"
            f"{r['synthetic_mean']:>9.3f}±{r['synthetic_sd']:<4.2f}{r['z']:>7.1f}{mark}"
        )
    return "\n".join(lines)


__all__ = [
    "STATS", "Photo", "compare", "load_frames", "load_reference", "sheet", "statistics", "table",
    "wasserstein",
]  # fmt: skip
