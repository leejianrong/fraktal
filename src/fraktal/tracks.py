"""Ground-truth tracks of moving cars from the label images (ADR-0011 phase 2, FRK-85).

For every rendered frame the label image says exactly which pixels belong to which car (class
vehicle, instance ``first_instance + k + 1`` for the k-th car). ``from_labels`` turns
that into one record per car: for every frame where it is visible, the box around its pixels,
the pixel count, and its simulated position, heading and speed. A car hidden behind a building has
no record for that frame, so occlusion is exact. Boxes are in image pixels, x to the right, y down,
with ``bbox = [x_min, y_min, x_max, y_max]`` inclusive.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from fraktal import labels3d
from fraktal.agents import Tracks


def _read(path: Path) -> Any:
    with rasterio.open(path) as src:
        return src.read()[:3]


def from_labels(
    label_files: list[Path], tracks: Tracks, first_instance: int, klass: str = "vehicle"
) -> dict[str, Any]:
    """Per-agent detections over ``label_files`` (one per track column, in order).

    ``klass`` is the label class of these agents: "vehicle" for cars, "person" for people.
    """
    if len(label_files) != tracks.x.shape[1]:
        raise ValueError(f"{len(label_files)} label images for {tracks.x.shape[1]} track frames")
    n = len(tracks.ids)
    detections: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    vehicle = labels3d.CLASSES[klass]
    for frame, path in enumerate(label_files):
        semantic, instance = labels3d.decode(_read(path))
        rows, cols = np.nonzero(semantic == vehicle)
        car = instance[rows, cols].astype("int64") - first_instance - 1
        keep = (car >= 0) & (car < n)
        rows, cols, car = rows[keep], cols[keep], car[keep]
        if not len(car):
            continue
        count = np.bincount(car, minlength=n)
        lo_x, hi_x = np.full(n, np.iinfo(np.int64).max), np.full(n, -1)
        lo_y, hi_y = np.full(n, np.iinfo(np.int64).max), np.full(n, -1)
        np.minimum.at(lo_x, car, cols)
        np.maximum.at(hi_x, car, cols)
        np.minimum.at(lo_y, car, rows)
        np.maximum.at(hi_y, car, rows)
        for k in np.nonzero(count)[0]:
            detections[int(k)].append(
                {
                    "frame": frame,
                    "t": round(float(tracks.t[frame]), 4),
                    "bbox": [int(lo_x[k]), int(lo_y[k]), int(hi_x[k]), int(hi_y[k])],
                    "pixels": int(count[k]),
                    "x": round(float(tracks.x[k, frame]), 3),
                    "y": round(float(tracks.y[k, frame]), 3),
                    "heading_deg": round(math.degrees(float(tracks.heading[k, frame])), 2),
                    "speed": round(float(tracks.speed[k, frame]), 3),
                    **({"z": round(float(tracks.z[k, frame]), 3)} if tracks.z is not None else {}),
                }
            )
    return {
        "schema": 1,
        "frames": len(label_files),
        "first_instance": first_instance,
        "note": "bbox is [x_min, y_min, x_max, y_max] in pixels around the visible pixels; "
        "a frame with no record means the car is not visible in it; x, y in metres east and "
        "north of the tile's south-west corner, heading in degrees anticlockwise from east",
        "tracks": [
            {"id": tracks.ids[k], "instance": first_instance + k + 1, "kind": tracks.kind,
             "detections": detections[k]}
            for k in range(n)
        ],
    }  # fmt: skip


__all__ = ["from_labels"]
