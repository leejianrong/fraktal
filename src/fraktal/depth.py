"""Depth images and camera geometry for the 3D path (ADR-0011 phase 1, FRK-79).

Depth is the distance along the camera's optical axis (the "z depth" of a pinhole camera) to the
first surface, stored as a 16-bit PNG in units of ``UNIT_M`` (5 cm), so it reaches about 3.3 km.
Zero means no surface (the sky). Blender writes the value split over two 8-bit channels, as the
labels do, and ``from_encoded`` joins them.

Camera geometry follows the OpenCV convention: camera x to the right, y down, z forward.
``camera_to_world`` maps camera coordinates into the tile frame (metres east, north, up), so a
world point is ``camera_to_world @ [x, y, z, 1]``. ``pixel_to_ray`` and ``project`` go between
pixels and rays with the intrinsics from ``intrinsics``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

UNIT_M = 0.05
MAX_RAW = 65535


class DepthError(ValueError):
    pass


def intrinsics(fov_deg: float, px: int) -> dict[str, float]:
    """Pinhole intrinsics for a square image; ``fov_deg`` is the full field of view across it."""
    f = (px / 2) / math.tan(math.radians(fov_deg) / 2)
    return {"fx": round(f, 4), "fy": round(f, 4), "cx": px / 2, "cy": px / 2, "width": px,
            "height": px, "fov_deg": fov_deg}  # fmt: skip


def camera_to_world(pose: dict[str, Any]) -> np.ndarray:
    """The 4x4 matrix for a pose of ``fraktal.camera`` (heading clockwise, off-nadir from down)."""
    n, h = math.radians(pose["off_nadir_deg"]), math.radians(pose["heading_deg"])
    rx = np.array([[1, 0, 0], [0, math.cos(n), -math.sin(n)], [0, math.sin(n), math.cos(n)]])
    rz = np.array([[math.cos(-h), -math.sin(-h), 0], [math.sin(-h), math.cos(-h), 0], [0, 0, 1]])
    rotation = rz @ rx @ np.diag([1.0, -1.0, -1.0])  # Blender looks down -Z with +Y up
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = (pose["x"], pose["y"], pose["z"])
    return out


def pixel_to_ray(u: float, v: float, k: dict[str, float]) -> np.ndarray:
    """The camera-frame direction (z = 1) through pixel centre ``(u, v)``."""
    return np.array([(u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], 1.0])


def project(point: Any, matrix: np.ndarray, k: dict[str, float]) -> tuple[float, float, float]:
    """``(u, v, depth)`` of a world point, or a ``DepthError`` when it is behind the camera."""
    p = np.linalg.inv(matrix) @ np.append(np.asarray(point, dtype=float), 1.0)
    if p[2] <= 0:
        raise DepthError("the point is behind the camera")
    return float(k["fx"] * p[0] / p[2] + k["cx"]), float(k["fy"] * p[1] / p[2] + k["cy"]), p[2]


def from_encoded(rgb: Any) -> np.ndarray:
    """A uint16 depth image from Blender's (H, W, 3) or (3, H, W) uint8 two-channel encoding."""
    image = np.asarray(rgb)
    if image.ndim != 3:
        raise DepthError(f"expected a 3-channel image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    joined = (image[..., 0].astype("uint32") << 8) | image[..., 1].astype("uint32")
    return np.asarray(joined.astype("uint16"))


def to_metres(raw: Any) -> np.ndarray:
    """Depth in metres from the 16-bit image; no-surface pixels become ``inf``."""
    raw = np.asarray(raw)
    return np.asarray(np.where(raw == 0, np.inf, raw.astype("float64") * UNIT_M))


def meta() -> dict[str, Any]:
    return {
        "encoding": "16-bit PNG, z depth along the optical axis; metres = value * unit_m; "
        "0 = no surface",
        "unit_m": UNIT_M,
        "max_m": round(MAX_RAW * UNIT_M, 1),
    }


__all__ = ["DepthError", "camera_to_world", "from_encoded", "intrinsics", "meta", "pixel_to_ray",
           "project", "to_metres"]  # fmt: skip
