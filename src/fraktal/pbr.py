"""Normal and roughness maps derived from a tileable colour map (FRK-126).

A colour map already carries relief for the materials that need it most: the ribs of corrugated tin,
plaster pitting, tile joints. The height is the colour's lightness with its slow variation removed;
the normal comes from the height's slope; the roughness from how bright the surface is. Every step
wraps at the image edge (FFT blur, rolled gradients), so a tileable map gives a tileable result.
This is a procedural stand-in, not a learned estimator; comparing with a model is left open.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]


def lightness(rgb: Array) -> Array:
    """Rec. 709 luma of an (H, W, 3) image in 0..1 (accepts uint8)."""
    img = np.asarray(rgb, dtype="float64")
    if img.max(initial=0.0) > 1.0:
        img = img / 255.0
    return np.asarray(img[..., :3] @ np.array([0.2126, 0.7152, 0.0722]))


def blur_wrap(img: Array, sigma_px: float) -> Array:
    """A Gaussian blur through the FFT, so the image wraps at its edges."""
    if sigma_px <= 0:
        return img
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    kernel = np.exp(-2.0 * (np.pi * sigma_px) ** 2 * (fx**2 + fy**2))
    return np.asarray(np.fft.ifft2(np.fft.fft2(img) * kernel).real)


def height_map(rgb: Array, detail_sigma_px: float = 24.0) -> Array:
    """Relief in 0..1: the lightness minus its slow variation (the lighting and colour drift)."""
    light = lightness(rgb)
    relief = light - blur_wrap(light, detail_sigma_px)
    spread = float(np.percentile(np.abs(relief), 98)) or 1.0
    return np.asarray(np.clip(0.5 + 0.5 * relief / spread, 0.0, 1.0))


def normal_map(height: Array, strength: float = 4.0) -> Array:
    """A tangent-space normal map, OpenGL convention (+Y up), as uint8 (H, W, 3), wrapping."""
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    n = np.stack([-dx * strength, dy * strength, np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.asarray(np.rint((n * 0.5 + 0.5) * 255.0), dtype="uint8")


def roughness_map(rgb: Array, height: Array, base: float = 0.7, spread: float = 0.25) -> Array:
    """Roughness as uint8 (H, W): ``base`` shifted up in dark, deep spots and down on bright, raised
    ones. Clipped to 0.05..1 so nothing is a perfect mirror."""
    light = lightness(rgb)
    centred = 0.5 * (light - float(np.median(light))) / (float(light.std()) or 1.0)
    value = base - spread * np.clip(centred, -1.0, 1.0) - 0.5 * spread * (height - 0.5)
    return np.asarray(np.rint(np.clip(value, 0.05, 1.0) * 255.0), dtype="uint8")


def derive(
    rgb: Array, strength: float = 4.0, detail_sigma_px: float = 24.0, rough_base: float = 0.7
) -> tuple[Array, Array]:
    """``(normal, roughness)`` for a colour map."""
    height = height_map(rgb, detail_sigma_px)
    return normal_map(height, strength), roughness_map(rgb, height, rough_base)


def seam_score(img: Array) -> float:
    """How visible the wrap seam is: the step across the seam over the typical step inside, 1 means
    none. Taken on a grey image, along both axes."""
    grey = np.asarray(img, dtype="float64")
    if grey.ndim == 3:
        grey = grey.mean(axis=-1)
    inside = np.concatenate(
        [np.abs(np.diff(grey, axis=0)).ravel(), np.abs(np.diff(grey, axis=1)).ravel()]
    )
    seam = np.concatenate([np.abs(grey[0] - grey[-1]), np.abs(grey[:, 0] - grey[:, -1])])
    return float(seam.mean() / (inside.mean() or 1.0))
