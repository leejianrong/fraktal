"""Separable Gaussian blur (ADR-0005): arrays in, an array out. Used for penumbra and sensor PSF."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def gaussian_kernel(sigma_px: float) -> np.ndarray[Any, Any]:
    """Normalised 1D Gaussian weights out to 3 sigma (a single 1.0 when sigma is 0)."""
    if sigma_px < 0:
        raise ValueError("sigma_px must not be negative")
    if sigma_px == 0:
        return np.ones(1, dtype=np.float32)
    radius = max(1, math.ceil(3 * sigma_px))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (x / sigma_px) ** 2)
    kernel: np.ndarray[Any, Any] = (weights / weights.sum()).astype(np.float32)
    return kernel


def gaussian_blur(image: np.ndarray[Any, Any], sigma_px: float) -> np.ndarray[Any, Any]:
    """Blur the last two axes of ``image`` (float32), reflecting at the edges."""
    if image.ndim < 2:
        raise ValueError("image must be at least 2D")
    kernel = gaussian_kernel(sigma_px)
    out = np.asarray(image, dtype=np.float32)
    if kernel.size == 1:
        return out.copy()
    radius = kernel.size // 2
    for axis in (-2, -1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for k, w in enumerate(kernel):
            index = [slice(None)] * out.ndim
            index[axis] = slice(k, k + n)
            acc += w * padded[tuple(index)]
        out = acc
    return out
