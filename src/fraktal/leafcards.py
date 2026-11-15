"""A leaf-cluster texture with alpha, drawn in code (FRK-169).

Leaf cards are small quads dressed with this image so a tree crown has a ragged, see-through
silhouette instead of a smooth blob. The image is grey (the shader tints it per tree), 4 channels,
north up. Pure numpy and deterministic through ``fraktal.ids.substream``, so the texture is ours
(licence ``owned``) and can be regenerated bit for bit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraktal.ids import substream

Array = np.ndarray[Any, Any]

LEAVES = 34  # leaves in one cluster
SUPERSAMPLE = 3  # coverage is drawn at this many times the size, then averaged, for soft edges


def leaf_cluster(seed: int = 0, size: int = 256) -> Array:
    """An ``(size, size, 4)`` uint8 RGBA cluster of leaves; RGB is grey light, A is coverage."""
    rng = substream(seed, "leafcards:cluster")
    n = size * SUPERSAMPLE
    ys, xs = np.mgrid[0:n, 0:n].astype("float32")
    xs, ys = xs / n - 0.5, ys / n - 0.5  # -0.5 to 0.5
    alpha = np.zeros((n, n), dtype="float32")
    light = np.zeros((n, n), dtype="float32")
    for _ in range(LEAVES):
        radius = 0.26 * math.sqrt(float(rng.uniform(0.05, 1.0)))  # a ragged disc of leaves
        theta = float(rng.uniform(0, 2 * math.pi))
        cx, cy = radius * math.cos(theta), radius * math.sin(theta)
        length = float(rng.uniform(0.13, 0.22))
        half_w = length * float(rng.uniform(0.28, 0.4))
        angle = float(rng.uniform(0, math.pi))
        shade = float(rng.uniform(0.55, 1.0))
        c, s = math.cos(angle), math.sin(angle)
        u = (xs - cx) * c + (ys - cy) * s  # along the leaf
        v = -(xs - cx) * s + (ys - cy) * c  # across it
        t = u / length
        inside = (np.abs(t) <= 1.0) & (np.abs(v) <= half_w * (1.0 - t * t))  # a pointed ellipse
        rib = np.abs(v) < half_w * 0.08  # a darker midrib
        value = np.where(rib, shade * 0.6, shade * (0.85 + 0.15 * (1.0 - np.abs(t))))
        light = np.where(inside, value, light)
        alpha = np.where(inside, 1.0, alpha)
    cov = alpha.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE).mean(axis=(1, 3))
    lit = light.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE).sum(axis=(1, 3)) / np.maximum(
        alpha.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE).sum(axis=(1, 3)), 1.0
    )
    grey = np.clip(np.rint(lit * 255.0), 0, 255).astype("uint8")
    out = np.empty((size, size, 4), dtype="uint8")
    out[..., :3] = grey[..., None]
    out[..., 3] = np.clip(np.rint(cov * 255.0), 0, 255).astype("uint8")
    return out


__all__ = ["LEAVES", "leaf_cluster"]
