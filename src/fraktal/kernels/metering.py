"""Auto-exposure for a strip of eye-level frames (FRK-171): one gamma for the whole strip.

Arrays and plain numbers in, arrays out (ADR-0005). Brightness is the mean of the brightest
channel, the number ``fraktal streets`` reports. The gamma pulls the strip's mean part of the way
to a target; a gamma below 1 lifts the shadows and mid-tones and leaves white at white, so nothing
new clips.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LEVELS = 256


def brightness_histogram(rgb: Any) -> Any:
    """256-bin count of the brightest channel of a (3, H, W) uint8 frame."""
    return np.bincount(np.asarray(rgb).max(axis=0).ravel(), minlength=LEVELS).astype("float64")


def mean_after(hist: Any, gamma: float) -> float:
    """Mean brightness (0 to 1) of the pixels in ``hist`` after ``x ** gamma``."""
    x = (np.arange(LEVELS) / (LEVELS - 1)) ** gamma
    return float((hist * x).sum() / max(float(hist.sum()), 1.0))


def solve_gamma(hist: Any, target: float, strength: float, lo: float, hi: float) -> float:
    """The gamma, within ``[lo, hi]``, that moves the mean ``strength`` of the way to ``target``.

    Strength 1 meets the target and 0 changes nothing. The mean falls as gamma rises, so bisection
    finds it; a strip already on the goal gets 1.0.
    """
    goal = mean_after(hist, 1.0) + strength * (target - mean_after(hist, 1.0))
    a, b = lo, hi
    for _ in range(40):
        mid = 0.5 * (a + b)
        if mean_after(hist, mid) > goal:
            a = mid
        else:
            b = mid
    return float(min(hi, max(lo, 0.5 * (a + b))))


def apply_gamma(rgb: Any, gamma: float) -> Any:
    """``x ** gamma`` on a (3, H, W) uint8 frame, the same curve on every channel."""
    lut = np.round(255.0 * (np.arange(LEVELS) / (LEVELS - 1)) ** gamma).astype("uint8")
    out: Any = lut[np.asarray(rgb)]
    return out
