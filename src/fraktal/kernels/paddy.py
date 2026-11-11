"""Paddy block look (ADR-0005): arrays in, an array out.

The class colours paint every paddy block one flat tone, so a paddy tile reads as neon squares.
Real blocks differ one from another (water depth, crop age, soil), are duller than the class
colour and show transplanting rows or stubble lines. ``paddy_tone`` mutes each block, gives it a
tone of its own and draws rows along its own angle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]

FLOODED, PLANTED, HARVESTED = 0, 1, 2
MUTE = np.array([0.75, 0.45, 0.55], dtype="float32")  # how far each state moves to its tint
# What each state tends to: muddy water, young rice, straw. Scaled to keep the pixel's luma.
_TINT = np.array([[1.15, 1.05, 0.75], [0.85, 1.10, 0.70], [1.18, 1.00, 0.68]], dtype="float32")
LUMA = np.array([0.299, 0.587, 0.114], dtype="float32")
TINT = _TINT / (_TINT @ LUMA)[:, None]
GAIN = np.array([0.80, 0.88, 0.80], dtype="float32")  # darkening: the sky fill brightens fields
ROW_AMP = np.array([0.0, 0.16, 0.10], dtype="float32")  # stripe depth: water has no rows
ROW_M = 0.9  # transplanting row spacing
BRIGHTNESS, WARMTH = 0.12, 0.07  # per-block tone spread


def paddy_tone(
    ground: Array, cells: Array, state: Array, jitter: Array, angle: Array, gsd_m: float
) -> Array:
    """``ground`` (H, W, 3, 0..1) with every paddy block restyled.

    ``cells`` is an (H, W) int array, 0 off blocks and ``i`` for block ``i``. ``state``, ``angle``
    (radians) and ``jitter`` (n + 1, 3, uniform in [-1, 1]: brightness, warmth, row phase) are
    indexed by block, entry 0 unused. Pixels off the blocks come back unchanged.
    """
    if ground.shape[:2] != cells.shape:
        raise ValueError("ground and cells must share their height and width")
    inside = cells > 0
    if not inside.any():
        return ground.copy()
    s = state[cells]
    j = jitter[cells]
    out = ground.astype("float32", copy=True)
    luma = out @ LUMA
    out = out + (luma[..., None] * TINT[s] - out) * MUTE[s][..., None]
    out *= (GAIN[s] * (1.0 + BRIGHTNESS * j[..., 0]))[..., None]
    out[..., 0] *= 1.0 + WARMTH * j[..., 1]
    out[..., 2] *= 1.0 - WARMTH * j[..., 1]
    rows, cols = np.indices(cells.shape, dtype="float32")
    a = angle[cells]
    along = (cols * np.cos(a) + rows * np.sin(a)) * gsd_m / ROW_M + j[..., 2]
    fade = float(np.clip((ROW_M / gsd_m - 1.5) / 1.5, 0.0, 1.0))  # full at 3 px a row, gone at 1.5
    stripe = 0.5 + 0.5 * np.cos(2.0 * np.pi * along)
    out *= (1.0 - ROW_AMP[s] * fade * stripe)[..., None]
    return np.where(inside[..., None], np.clip(out, 0.0, 1.0), ground).astype("float32")
