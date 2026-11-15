"""Camera chain for eye-level strips (FRK-165): what a phone or dashcam does to a clean render.

The strip's frames come from one camera, so the rig (exposure, softness, fringe, vignette, noise
level, compression, bonnet) is drawn once per strip from ``fraktal.ids.substream``. Only the noise
differs from frame to frame, drawn from a per-frame stream. Everything is photometric except the
bonnet: it hides pixels, so the labels (and depth, when present) are blanked under it. That is
the one place a label changes, and ``apply`` returns the mask so a test can show it.

The arithmetic is in ``fraktal.kernels.sensor``; this module draws parameters and moves files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels import sensor as k

RAW_DIR = "frames_raw"


def draw_rig(seed: int, dashcam: bool = False) -> dict[str, Any]:
    """The strip's camera, drawn once. Ranges are typical of phones and dashcams, not fitted."""
    rng = substream(seed, "eyesensor:rig")
    rig: dict[str, Any] = {
        "ev": round(float(rng.uniform(0.2, 0.45)), 3),
        "shoulder": round(float(rng.uniform(0.15, 0.4)), 3),
        "contrast": round(float(rng.uniform(1.0, 1.12)), 3),
        "saturation": round(float(rng.uniform(0.75, 0.9)), 3),
        "psf_sigma_px": round(float(rng.uniform(0.3, 0.6)), 3),
        "ca_px": round(float(rng.uniform(0.4, 1.2)), 3),
        "vignette": round(float(rng.uniform(0.12, 0.3)), 3),
        "read_sigma": round(float(rng.uniform(1.0, 2.5)), 3),
        "shot_gain": round(float(rng.uniform(0.15, 0.45)), 3),
        "jpeg_quality": int(rng.integers(55, 85)),
        "dashcam": None,
    }
    if dashcam:
        rig["dashcam"] = {
            "height_share": round(float(rng.uniform(0.10, 0.16)), 3),
            "curve": round(float(rng.uniform(0.05, 0.25)), 3),
            "tone": [round(float(v), 3) for v in rng.uniform(0.7, 1.0, 3)],
        }
        rig["jpeg_quality"] = int(rng.integers(40, 65))  # dashcams compress harder
    return rig


def process(rgb: Any, rig: dict[str, Any], seed: int, frame: int) -> tuple[Any, Any]:
    """One (3, H, W) uint8 frame through the rig: ``(image, hidden mask (H, W) bool)``."""
    out = k.expose(rgb, rig["ev"], rig["shoulder"])
    out = k.sensor_chain(out, contrast=rig["contrast"], psf_sigma_px=rig["psf_sigma_px"])
    out = k.desaturate(out, rig["saturation"])
    out = k.add_sensor_noise(
        out, rig["read_sigma"], rig["shot_gain"], substream(seed, f"eyesensor:noise:{frame}")
    )
    out = k.lens_effects(out, vignette=rig["vignette"], ca_px=rig["ca_px"], jpeg_quality=0)
    mask = np.zeros(out.shape[1:], dtype=bool)
    if rig["dashcam"]:
        d = rig["dashcam"]
        layer, mask = k.bonnet(
            out.shape[1], out.shape[2], d["height_share"], d["curve"], tuple(d["tone"])
        )
        out = np.where(mask[None], layer, out)
    return k.lens_effects(out, jpeg_quality=rig["jpeg_quality"]), mask


def apply(out_dir: Path, seed: int, dashcam: bool = False) -> dict[str, Any]:
    """Run the rig over ``out_dir/frames``: the raw frames move to ``frames_raw``, the processed
    ones take their place, and under a bonnet the labels (and raw depth) are zeroed.

    Returns the rig, which the caller records in ``render.json``. Idempotent only by re-rendering:
    a second call would process the processed frames, so it refuses.
    """
    import rasterio

    frames = out_dir / "frames"
    raw = out_dir / RAW_DIR
    if raw.exists():
        raise ValueError(f"{raw} exists: the eye-level chain was already applied")
    files = sorted(frames.glob("frame_*.png"))
    if not files:
        raise ValueError(f"no frames in {frames}")
    rig = draw_rig(seed, dashcam)
    raw.mkdir()
    for i, path in enumerate(files):
        with rasterio.open(path) as src:
            rgb = src.read()[:3].astype("uint8")
        path.rename(raw / path.name)
        out, mask = process(rgb, rig, seed, i)
        with rasterio.open(
            path, "w", driver="PNG", width=out.shape[2], height=out.shape[1], count=3, dtype="uint8"
        ) as dst:
            dst.write(out)
        if mask.any():
            for sub in ("labels", "depth_raw"):
                _blank(out_dir / sub / path.name, mask)
    (out_dir / "eye_sensor.json").write_text(json.dumps(rig, indent=2, sort_keys=True))
    return rig


def _blank(path: Path, mask: Any) -> None:
    """Zero the pixels under ``mask`` in a PNG (labels: class none; depth: no reading)."""
    import rasterio

    if not path.is_file():
        return
    with rasterio.open(path) as src:
        data, profile = src.read(), src.profile
    data[:, mask] = 0
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


__all__ = ["RAW_DIR", "apply", "draw_rig", "process"]
