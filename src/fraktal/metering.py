"""Auto-exposure over a strip's frames (FRK-171), the camera metering the scene the light made.

Seeds change the light and the layout (how much sky is in view), and Manila-Quezon strips came out
between 0.31 and 0.46 bright. A real camera meters, so every eye-level strip is pulled part of the
way to ``lighting.METER_TARGET``: ``lighting.METER_STRENGTH`` of the gap, with one gamma for the
whole strip so frames keep their relative light. Only the beauty frames change; labels and depth do
not. Run it before the eye-level sensor chain, which then starts from the metered frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fraktal import lighting
from fraktal.kernels import metering as k


def apply(out_dir: Path) -> dict[str, Any]:
    """Meter ``out_dir/frames`` in place; return the note for ``render.json``."""
    import rasterio

    files = sorted((out_dir / "frames").glob("frame_*.png"))
    if not files:
        raise ValueError(f"no frames in {out_dir / 'frames'}")
    frames = []
    hist = np.zeros(k.LEVELS)
    for path in files:
        with rasterio.open(path) as src:
            rgb = src.read()[:3].astype("uint8")
        frames.append(rgb)
        hist += k.brightness_histogram(rgb)
    before = k.mean_after(hist, 1.0)
    gamma = k.solve_gamma(
        hist, lighting.METER_TARGET, lighting.METER_STRENGTH, *lighting.METER_GAMMA
    )
    for path, rgb in zip(files, frames, strict=True):
        out = k.apply_gamma(rgb, gamma)
        with rasterio.open(
            path, "w", driver="PNG", width=out.shape[2], height=out.shape[1], count=3, dtype="uint8"
        ) as dst:
            dst.write(out)
    note = {
        "gamma": round(gamma, 4),
        "brightness_before": round(before, 4),
        "brightness_after": round(k.mean_after(hist, gamma), 4),
        "target": lighting.METER_TARGET,
        "strength": lighting.METER_STRENGTH,
    }
    (out_dir / "metering.json").write_text(json.dumps(note, indent=2, sort_keys=True))
    return note


__all__ = ["apply"]
