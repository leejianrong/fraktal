"""Optical flow from camera motion and depth (ADR-0005, ADR-0011 phase 3).

Arrays in, arrays out. For every pixel of frame A the depth gives a world point; that point is
moved by ``motion`` (metres, zero for the static scene) and projected into the camera of frame B.
The flow is where the pixel lands in B minus where it was in A, in pixels, x to the right and y
down. Pixels with no surface (depth ``inf``) or whose point falls behind camera B get ``nan``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def flow(
    depth_m: np.ndarray[Any, Any],
    a_to_world: np.ndarray[Any, Any],
    b_to_world: np.ndarray[Any, Any],
    k: dict[str, float],
    motion: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """Flow (H, W, 2) from frame A to frame B; see the module notes."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = np.where(np.isfinite(depth_m), depth_m, np.nan)
    cam = np.stack([(u - k["cx"]) / k["fx"] * z, (v - k["cy"]) / k["fy"] * z, z], axis=-1)
    world = cam @ a_to_world[:3, :3].T + a_to_world[:3, 3]
    if motion is not None:
        world = world + motion
    back = (world - b_to_world[:3, 3]) @ b_to_world[
        :3, :3
    ]  # into camera B (rotation is orthonormal)
    with np.errstate(invalid="ignore", divide="ignore"):
        zb = np.where(back[..., 2] > 0, back[..., 2], np.nan)
        ub = k["fx"] * back[..., 0] / zb + k["cx"]
        vb = k["fy"] * back[..., 1] / zb + k["cy"]
    return np.stack([ub - u, vb - v], axis=-1).astype("float32")


__all__ = ["flow"]
