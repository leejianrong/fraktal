"""Sensor effects shared by the 2.5D and 3D renderers: haze, tone, contrast, blur and noise.

Arrays and plain parameters in, arrays out (ADR-0005). The 2.5D renderer calls ``grade`` on its
float radiance before rounding; the 3D path calls ``sensor_chain`` on a finished 8-bit render, so
both look through the same lens. Randomness is the caller's: noise takes a generator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraktal.kernels.filters import gaussian_blur

HAZE_RGB = (200.0, 205.0, 215.0)  # colour of scattered skylight that haze adds


def transmittance(optical_depth: float, sun_elevation_deg: float) -> float:
    """Share of surface light that survives the haze: falls as optical depth grows."""
    if optical_depth < 0:
        raise ValueError("optical depth must not be negative")
    return math.exp(-optical_depth / math.sin(math.radians(sun_elevation_deg)))


def apply_haze(radiance: Any, optical_depth: float, sun_elevation_deg: float) -> Any:
    """Blend (H, W, 3) radiance toward the haze colour: ``r * t + haze * (1 - t)``."""
    t = transmittance(optical_depth, sun_elevation_deg)
    return radiance * t + np.array(HAZE_RGB, dtype="float32") * (1.0 - t)


def grade(
    radiance: Any,
    haze: tuple[float, float] = (0.0, 50.0),
    tone: tuple[float, float, float] = (1.0, 1.0, 1.0),
    contrast: float = 1.0,
) -> Any:
    """Haze, per-channel tone and contrast on (H, W, 3) float radiance; no rounding yet.

    ``haze`` is (optical depth, sun elevation in degrees). Contrast stretches about each
    channel's own mean, so dark channels (blue) survive.
    """
    out = apply_haze(radiance, haze[0], haze[1]) if haze[0] > 0 else radiance
    graded = out * np.array(tone, dtype="float32")
    if contrast != 1.0:
        pivot = graded.reshape(-1, 3).mean(axis=0)
        graded = (graded - pivot) * contrast + pivot
    return graded


def apply_psf(rgb: Any, sigma_px: float) -> Any:
    """Gaussian point-spread blur of a (3, H, W) uint8 image, one band at a time."""
    if sigma_px <= 0:
        return rgb
    out = np.empty_like(rgb)
    for band in range(rgb.shape[0]):
        blurred = gaussian_blur(rgb[band].astype("float32"), sigma_px)
        out[band] = np.clip(np.rint(blurred), 0, 255).astype("uint8")
    return out


def add_gaussian_noise(tile: Any, sigma: float, rng: Any) -> Any:
    """Add zero-mean Gaussian noise (std ``sigma`` in digital numbers) to a (3, H, W) uint8 tile."""
    if sigma <= 0:
        return tile
    noise = rng.standard_normal(tile.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(np.rint(tile.astype("float32") + noise), 0, 255).astype("uint8")


def _radius(h: int, w: int) -> Any:
    """Distance from the centre, 1 at the corners, as an (H, W) array."""
    y, x = np.indices((h, w), dtype="float32")
    cy, cx = (h - 1) / 2, (w - 1) / 2
    return np.hypot(y - cy, x - cx) / max(float(np.hypot(cy, cx)), 1e-6)


def _scaled(channel: Any, factor: float) -> Any:
    """``channel`` resampled so the picture is ``factor`` times larger about the centre."""
    h, w = channel.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y, x = np.indices((h, w), dtype="float32")
    sy, sx = np.clip(cy + (y - cy) / factor, 0, h - 1), np.clip(cx + (x - cx) / factor, 0, w - 1)
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = sy - y0, sx - x0
    top = channel[y0, x0] * (1 - fx) + channel[y0, x1] * fx
    bottom = channel[y1, x0] * (1 - fx) + channel[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def lens_effects(rgb: Any, vignette: float = 0.0, ca_px: float = 0.0, jpeg_quality: int = 0) -> Any:
    """Drone-lens effects on a (3, H, W) uint8 image; each one is off at 0.

    ``ca_px`` is the red-blue fringe at the corner, in pixels (red is magnified, blue shrunk);
    ``vignette`` darkens the corners by that share; ``jpeg_quality`` (1 to 95) is a JPEG round trip.
    """
    out = rgb
    if ca_px > 0:
        half_diag = float(np.hypot(*out.shape[1:])) / 2
        k = ca_px / max(half_diag, 1.0)
        planes = out.astype("float32")
        planes[0], planes[2] = _scaled(planes[0], 1 + k), _scaled(planes[2], 1 - k)
        out = np.clip(np.rint(planes), 0, 255).astype("uint8")
    if vignette > 0:
        falloff = 1.0 - vignette * _radius(*out.shape[1:]) ** 2
        out = np.clip(np.rint(out.astype("float32") * falloff), 0, 255).astype("uint8")
    if jpeg_quality > 0:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.moveaxis(out, 0, -1)).save(buf, "JPEG", quality=int(jpeg_quality))
        out = np.moveaxis(np.asarray(Image.open(buf).convert("RGB")), -1, 0).copy()
    return out


def sensor_chain(
    rgb: Any,
    *,
    haze: tuple[float, float] = (0.0, 50.0),
    tone: tuple[float, float, float] = (1.0, 1.0, 1.0),
    contrast: float = 1.0,
    psf_sigma_px: float = 0.0,
) -> Any:
    """Grade then blur a finished (3, H, W) uint8 render: the 3D path's sensor, noise aside."""
    radiance = np.moveaxis(rgb.astype("float32"), 0, -1)
    graded = np.clip(np.rint(grade(radiance, haze, tone, contrast)), 0, 255).astype("uint8")
    return apply_psf(np.moveaxis(graded, -1, 0), psf_sigma_px)


__all__ = [
    "HAZE_RGB",
    "add_gaussian_noise",
    "apply_haze",
    "apply_psf",
    "grade",
    "lens_effects",
    "sensor_chain",
    "transmittance",
]
