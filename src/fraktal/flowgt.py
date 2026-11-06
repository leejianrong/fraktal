"""Optical flow ground truth for a rendered camera sequence (ADR-0011 phase 3, FRK-89).

``write_sequence`` joins the depth images, the camera poses and (when there are moving agents)
the label images into one flow field per pair of consecutive frames, ``flow/frame_NNNN.npy``,
from frame N to frame N+1: float16, shape (H, W, 2), pixels, x to the right and y down, ``nan``
where nothing is seen. The static scene moves only through the camera; a moving agent's pixels
also move by that agent's simulated displacement between the two frames (translation only, so a
turning car's flow is approximate at its ends). The last frame has no successor, so no file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from fraktal import depth as depths
from fraktal import labels3d
from fraktal.agents import Tracks
from fraktal.kernels import flow as flow_kernel


def _read(path: Path) -> Any:
    with rasterio.open(path) as src:
        return src.read()


def _motion(
    label_file: Path, frame: int, groups: list[tuple[Tracks, str, int]]
) -> np.ndarray[Any, Any]:
    """World displacement (H, W, 3) from ``frame`` to ``frame + 1`` for the agents' pixels."""
    semantic, instance = labels3d.decode(_read(label_file)[:3])
    out = np.zeros((*semantic.shape, 3))
    for tracks, klass, first in groups:
        k = instance.astype("int64") - first - 1
        mask = (semantic == labels3d.CLASSES[klass]) & (k >= 0) & (k < len(tracks.ids))
        kk = k[mask]
        out[mask, 0] = tracks.x[kk, frame + 1] - tracks.x[kk, frame]
        out[mask, 1] = tracks.y[kk, frame + 1] - tracks.y[kk, frame]
        if tracks.z is not None:
            out[mask, 2] = tracks.z[kk, frame + 1] - tracks.z[kk, frame]
    return out


def write_sequence(
    out_dir: Path,
    poses: dict[str, Any],
    depth_files: list[Path],
    label_files: list[Path] | None = None,
    groups: list[tuple[Tracks, str, int]] | None = None,
) -> int:
    """Write the flow files under ``out_dir/flow`` and return how many there are."""
    frames = poses["frames"]
    if len(depth_files) != len(frames):
        raise ValueError(f"{len(depth_files)} depth images for {len(frames)} poses")
    k = poses["intrinsics"]
    target = out_dir / "flow"
    target.mkdir(parents=True, exist_ok=True)
    for i in range(len(frames) - 1):
        z = depths.to_metres(_read(depth_files[i])[0])
        motion = _motion(label_files[i], i, groups) if label_files and groups else None
        a = np.asarray(frames[i]["camera_to_world"])
        b = np.asarray(frames[i + 1]["camera_to_world"])
        field = flow_kernel.flow(z, a, b, k, motion)
        np.save(target / f"frame_{i:04d}.npy", field.astype("float16"))
    return max(len(frames) - 1, 0)


def meta() -> dict[str, Any]:
    return {
        "encoding": "flow/frame_NNNN.npy, float16 (H, W, 2) pixels, x right, y down, from frame "
        "N to N+1; nan where no surface is seen or the point leaves the view behind the camera",
        "agents": "moving agents add their simulated displacement (translation only)",
    }


__all__ = ["meta", "write_sequence"]
