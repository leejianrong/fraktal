"""An independent rasterisation of building footprints, used to validate rendered masks (R5).

The renderer paints masks with GDAL's scan-line rasteriser. This module recomputes the same
mask differently: a pixel is a building pixel if its centre lies inside a footprint, tested with
shapely over each footprint's bounding window. Two code paths agreeing is the evidence that
labels match pixels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import shapely
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry


def independent_building_mask(
    geometries: Sequence[BaseGeometry],
    transform: Affine,
    shape: tuple[int, int],
    touching: bool = False,
) -> np.ndarray[Any, Any]:
    """uint8 mask for a north-up grid (``transform.b == transform.d == 0``).

    A pixel is set when its centre lies strictly inside a footprint; with ``touching`` also when
    the centre is exactly on an edge. Rasterisers differ on which edge pixels they paint, so a
    label is right when it lies between the two masks.
    """
    if transform.b != 0 or transform.d != 0 or transform.e >= 0:
        raise ValueError("only north-up grids are supported")
    rows, cols = shape
    mask = np.zeros(shape, dtype=np.uint8)
    for geom in geometries:
        if geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        c0 = max(0, math.floor((minx - transform.c) / transform.a))
        c1 = min(cols, math.ceil((maxx - transform.c) / transform.a))
        r0 = max(0, math.floor((maxy - transform.f) / transform.e))
        r1 = min(rows, math.ceil((miny - transform.f) / transform.e))
        if c0 >= c1 or r0 >= r1:
            continue
        xs = transform.c + (np.arange(c0, c1) + 0.5) * transform.a
        ys = transform.f + (np.arange(r0, r1) + 0.5) * transform.e
        grid_x, grid_y = np.meshgrid(xs, ys)
        inside = (shapely.intersects_xy if touching else shapely.contains_xy)(geom, grid_x, grid_y)
        mask[r0:r1, c0:c1] |= inside.astype(np.uint8)
    return mask
