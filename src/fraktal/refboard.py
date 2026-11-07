"""Looking at real reference chips (ADR-0012; FRK-100, FRK-101). Evaluation and looking only.

The benchmark chips for SpaceNet 2 Khartoum are stretched to uint8 with one range for all three
bands (``chips.stretch_range``). The raw blue band is far lower than green, so that keeps a strong
green cast, and the Khartoum tone and palette were fitted to it. They stay as they are until the
next training milestone (ADR-0012 addendum). To *look* at a SpaceNet tile, ``spacenet_view``
stretches each band by its own range instead, which gives a natural colour image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from fraktal.chips import tile_path

SAMPLE_TILES = 64
LOW_PCT, HIGH_PCT = 2.0, 98.0


def band_ranges(root: Path, tile_ids: Sequence[str]) -> np.ndarray[Any, Any]:
    """A (3, 2) array: the 2nd and 98th percentile of each band over a sample of tiles.

    No-data (zero in any band) is ignored. ``root`` is ``<data>/spacenet2/khartoum``.
    """
    step = max(1, len(tile_ids) // SAMPLE_TILES)
    parts = []
    for tile_id in sorted(tile_ids)[::step]:
        with rasterio.open(tile_path(root, tile_id)) as src:
            data = src.read().astype("float32")
        valid = (data > 0).all(axis=0)
        parts.append(data[:, valid][:, ::37])
    flat = np.concatenate(parts, axis=1)
    return np.stack([np.percentile(band, [LOW_PCT, HIGH_PCT]) for band in flat]).astype("float32")


def spacenet_view(root: Path, tile_id: str, ranges: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The tile as a (3, H, W) uint8 image, each band stretched by its own range."""
    with rasterio.open(tile_path(root, tile_id)) as src:
        data = src.read().astype("float32")
    low = ranges[:, 0][:, None, None]
    high = ranges[:, 1][:, None, None]
    scaled = (data - low) / np.maximum(high - low, 1.0)
    out: np.ndarray[Any, Any] = np.clip(np.rint(scaled * 255), 0, 255).astype(np.uint8)
    return out


__all__ = ["band_ranges", "spacenet_view"]
