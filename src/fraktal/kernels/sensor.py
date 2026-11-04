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
    "sensor_chain",
    "transmittance",
]
