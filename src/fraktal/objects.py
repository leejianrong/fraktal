"""Place trees and vehicles, and pattern roofs, deterministically from named sub-streams.

Objects exist in the render only: they join the surface model (so trees and cars cast shadows)
and the shading, but they are not world features and never touch the building labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.filters import gaussian_blur
from fraktal.kernels.objects import paint_objects

TREES_PER_HA = 25.0
CARS_PER_HA = 6.0
TREE_CLASSES = (10, 20, 30, 40, 50, 60)  # trees stand on vegetation and bare or built-up ground
CAR_CLASSES = (50, 60)  # vehicles stand on built-up and bare ground
CAR_COLOURS = np.array(
    [(235, 235, 235), (40, 40, 45), (150, 30, 30), (170, 175, 180), (30, 60, 120)], dtype="float32"
)
TREE_RGB = (38.0, 78.0, 34.0)
PALM_RGB = (52.0, 92.0, 38.0)


def _pick(
    landcover: Any, building_mask: Any, rows: Any, cols: Any, classes: tuple[int, ...]
) -> Any:
    ok = np.isin(landcover[rows, cols], classes) & (building_mask[rows, cols] == 0)
    return np.nonzero(ok)[0]


CARS_PER_ROAD_HA = 14.0  # vehicles per hectare of road surface when roads are painted


def road_heading(road: Any, gsd_m: float) -> Any:
    """Direction of the road at each pixel, radians, from the gradient of the blurred road mask."""
    blurred = gaussian_blur((road > 0).astype("float32"), max(2.0, 4.0 / gsd_m))
    grad_row, grad_col = np.gradient(blurred)
    return np.arctan2(grad_col, -grad_row)  # perpendicular to the gradient, as (col, row) angle


def place(
    landcover: Any,
    building_mask: Any,
    gsd_m: float,
    density: float,
    seed: int,
    road: Any = None,
) -> tuple[Any, Any, Any]:
    """Shapes, heights and colours of trees then cars, for ``paint_objects``.

    With a ``road`` raster (non-zero on roads) vehicles stand on the road, along its direction;
    without one they keep to built-up and bare ground as before.
    """
    rows_px, cols_px = landcover.shape
    hectares = rows_px * cols_px * gsd_m**2 / 1e4
    parts: list[tuple[Any, Any, Any]] = []
    for name, per_ha, classes in (
        ("trees", TREES_PER_HA, TREE_CLASSES),
        ("cars", CARS_PER_HA, CAR_CLASSES),
    ):
        rng = substream(seed, f"objects:{name}")
        on_road = name == "cars" and road is not None
        if on_road:
            road_rows, road_cols = np.nonzero(road)
            n = int(CARS_PER_ROAD_HA * density * len(road_rows) * gsd_m**2 / 1e4)
            pick = rng.integers(0, max(1, len(road_rows)), n)
            rows, cols = road_rows[pick], road_cols[pick]
            keep = np.nonzero(building_mask[rows, cols] == 0)[0]
        else:
            n = int(per_ha * density * hectares)
            rows = rng.integers(0, rows_px, n)
            cols = rng.integers(0, cols_px, n)
            keep = _pick(landcover, building_mask, rows, cols, classes)
        rows, cols = rows[keep], cols[keep]
        k = len(keep)
        if name == "trees":
            radius = rng.uniform(1.5, 4.0, n)[keep] / gsd_m
            shapes = np.column_stack([rows, cols, radius, radius, np.zeros(k), np.zeros(k)])
            heights = rng.uniform(4.0, 9.0, n)[keep]
            shade = rng.uniform(0.7, 1.3, n)[keep, None]
            colours = np.array(TREE_RGB, dtype="float32") * shade
        else:
            length = np.full(k, 2.25 / gsd_m)
            width = np.full(k, 0.95 / gsd_m)
            angle = rng.uniform(0, np.pi, n)[keep]
            if on_road:
                heading = road_heading(road, gsd_m)
                angle = heading[rows, cols] + rng.normal(0.0, 0.05, n)[keep]
            shapes = np.column_stack([rows, cols, length, width, angle, np.ones(k)])
            heights = np.full(k, 1.5)
            colours = CAR_COLOURS[rng.integers(0, len(CAR_COLOURS), n)[keep]]
        parts.append((shapes, heights, colours))
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


def scatter(
    landcover: Any,
    building_mask: Any,
    gsd_m: float,
    density: float,
    seed: int,
    road: Any = None,
    extra: tuple[Any, Any, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Height (m), colour and coverage rasters for the trees and cars of one region.

    ``extra`` is more (shapes, heights, colours), such as planted rows, painted last.
    """
    shapes, heights, colours = place(landcover, building_mask, gsd_m, density, seed, road)
    if extra is not None and len(extra[0]):
        shapes = np.concatenate([shapes, extra[0]])
        heights = np.concatenate([heights, extra[1]])
        colours = np.concatenate([colours, extra[2]])
    return paint_objects(landcover.shape, shapes, heights, colours, building_mask)


def row_objects(points_px: Any, gsd_m: float, seed: int) -> tuple[Any, Any, Any]:
    """Palm crowns at pixel positions (N, 2 as row, col): shapes, heights and colours."""
    n = len(points_px)
    rng = substream(seed, "objects:rows")
    radius = rng.uniform(2.3, 3.0, n) / gsd_m
    zeros = np.zeros(n)
    shapes = np.column_stack([points_px[:, 0], points_px[:, 1], radius, radius, zeros, zeros])
    heights = rng.uniform(8.0, 13.0, n)
    colours = np.array(PALM_RGB, dtype="float32") * rng.uniform(0.8, 1.2, n)[:, None]
    return shapes, heights, colours


def roof_pattern(index: Any, ids: Any, gsd_m: float, amplitude: float) -> Any:
    """Multiplier (H, W) that stripes each roof like ridges or corrugation, 1.0 off roofs.

    Direction, period and phase come from the building's stable id, so a roof keeps its pattern
    wherever it is rendered.
    """
    out = np.ones(index.shape, dtype="float32")
    if amplitude <= 0 or len(ids) == 0:
        return out
    keys = np.array([int(str(i)[-8:], 16) for i in ids], dtype="float64")
    angle = (keys % 1000) / 1000 * np.pi
    period_m = 1.0 + (keys // 1000 % 1000) / 1000 * 1.5
    phase = (keys // 1_000_000 % 1000) / 1000 * 2 * np.pi
    rows, cols = np.nonzero(index)
    b = index[rows, cols].astype(np.int64) - 1
    wave = np.sin(
        2 * np.pi * (np.cos(angle[b]) * cols + np.sin(angle[b]) * rows) * gsd_m / period_m[b]
        + phase[b]
    )
    out[rows, cols] = 1.0 + amplitude * np.clip(wave * 3.0, -1.0, 1.0)
    return out
