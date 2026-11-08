"""Which kind of tree stands where, for the 3D path (FRK-66).

Three shapes: ``broadleaf`` (a round crown on a trunk), ``palm`` (a bare trunk and a star of
fronds) and ``shrub`` (a low wide mound). A tree under 3.5 m is always a shrub; the others are drawn
from a per-biome mix with a sub-stream of the tree's own position, so a tree keeps its species when
the tile window or the tree count changes. The mixes are assumptions from general knowledge of each
setting, not counts from real imagery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.ids import substream

Array = np.ndarray[Any, Any]

SPECIES = ("broadleaf", "palm", "shrub")
SHRUB_BELOW_M = 3.5
# Shares of (broadleaf, palm, shrub) among trees of 3.5 m and over, then the shrub share added
# by chance on top of the small trees that are always shrubs.
MIX: dict[str, tuple[float, float, float]] = {
    "khartoum": (0.85, 0.05, 0.10),  # neem and acacia, the odd date palm
    "dense_urban": (0.55, 0.35, 0.10),
    "informal_settlement": (0.45, 0.35, 0.20),
    "oil_palm_plantation": (0.35, 0.55, 0.10),
    "paddy_terraces": (0.50, 0.40, 0.10),
    "mangrove_coast": (0.55, 0.35, 0.10),
}
DEFAULT_MIX = (0.6, 0.3, 0.1)


def species_for(biome: str, x: float, y: float, height_m: float, seed: int) -> str:
    """The species of the tree at (x, y) metres in the tile frame."""
    if height_m < SHRUB_BELOW_M:
        return "shrub"
    roll = float(substream(seed, f"tree:{round(x, 1)}:{round(y, 1)}").uniform())
    broadleaf, palm, _ = MIX.get(biome, DEFAULT_MIX)
    if roll < broadleaf:
        return "broadleaf"
    return "palm" if roll < broadleaf + palm else "shrub"


# Scrub placement (FRK-111). Real dry-land cover is mostly small shrubs in thickets with a few big
# shade trees, not evenly spread round crowns. Numbers are set by eye against the reference board.
PER_HA = {  # trees and shrubs per hectare before ``density``
    "khartoum": 70.0,
    "dense_urban": 30.0,
    "informal_settlement": 40.0,
    "oil_palm_plantation": 25.0,
    "paddy_terraces": 40.0,
    "mangrove_coast": 90.0,
}
DEFAULT_PER_HA = 40.0
CLUSTER_SHARE = 0.6  # of the trees stand in thickets, the rest alone
THICKET_M = (3.0, 7.0, 18.0)  # spread (min, median, max) of a thicket, metres
MEAN_PER_THICKET = 14.0
RADIUS_MEDIAN_M, RADIUS_SIGMA = 1.3, 0.45
RADIUS_MIN_M, RADIUS_MAX_M = 0.5, 4.5
BIG_TREE_SHARE = 0.04  # shade trees: crowns of 3 to 5 m
HEDGE_SHARE = 0.2  # of the compounds get a hedge along one wall
HEDGE_SPACING_M = 1.2
HEDGE_OFFSET_M = 0.9
TREE_CLASSES = (10, 20, 30, 40, 50, 60, 95)  # trees stand on vegetation and bare or built-up ground


def _wall_distance(x: Array, y: Array, walls: Array) -> Array:
    """Distance from each point to the nearest wall segment (``walls`` is n x 4: x0 y0 x1 y1)."""
    if len(walls) == 0:
        return np.full(len(x), np.inf)
    ax, ay, bx, by = (walls[:, i][None, :] for i in range(4))
    dx, dy = bx - ax, by - ay
    t = ((x[:, None] - ax) * dx + (y[:, None] - ay) * dy) / np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    return np.asarray(np.hypot(x[:, None] - (ax + t * dx), y[:, None] - (ay + t * dy)).min(axis=1))


def scatter(
    surface: Array,
    mask: Array,
    gsd_m: float,
    density: float,
    seed: int,
    biome: str,
    walls: list[list[float]] | None = None,
) -> list[list[Any]]:
    """Trees as ``[x, y, radius_m, height_m, species]`` in tile metres (y up from the south edge).

    Thickets (a Gaussian cloud round a parent) plus loners, sizes from a heavy-tailed mix with a few
    big shade trees, none on a building or within its crown of a wall. Hedges of small shrubs follow
    one wall of some compounds.
    """
    side = surface.shape[0]
    size_m = side * gsd_m
    rng = substream(seed, "trees:scrub")
    n = int(PER_HA.get(biome, DEFAULT_PER_HA) * density * size_m**2 / 1e4)
    if n <= 0:
        return []
    n_cluster = int(n * CLUSTER_SHARE)
    parents = max(1, round(n_cluster / MEAN_PER_THICKET))
    margin = THICKET_M[2]
    px = rng.uniform(-margin, size_m + margin, parents)
    py = rng.uniform(-margin, size_m + margin, parents)
    spread = np.clip(
        THICKET_M[1] * np.exp(rng.normal(0.0, 0.5, parents)), THICKET_M[0], THICKET_M[2]
    )
    weight = np.exp(rng.normal(0.0, 0.6, parents))
    owner = rng.choice(parents, n_cluster, p=weight / weight.sum())
    cx = px[owner] + rng.normal(0.0, 1.0, n_cluster) * spread[owner] / 2
    cy = py[owner] + rng.normal(0.0, 1.0, n_cluster) * spread[owner] / 2
    x = np.concatenate([cx, rng.uniform(0.0, size_m, n - n_cluster)])
    y = np.concatenate([cy, rng.uniform(0.0, size_m, n - n_cluster)])
    radius = np.clip(
        RADIUS_MEDIAN_M * np.exp(rng.normal(0.0, RADIUS_SIGMA, n)), RADIUS_MIN_M, RADIUS_MAX_M
    )
    big = rng.uniform(size=n) < BIG_TREE_SHARE
    radius = np.where(big, rng.uniform(3.0, 5.0, n), radius)
    height = 1.5 + 1.9 * radius**0.9 * np.exp(rng.normal(0.0, 0.15, n))
    wall_arr = np.asarray(walls if walls else [], dtype="float64").reshape(-1, 4)
    hedge = _hedges(wall_arr, mask, gsd_m, size_m, rng)
    x, y = np.concatenate([x, hedge[:, 0]]), np.concatenate([y, hedge[:, 1]])
    radius = np.concatenate([radius, hedge[:, 2]])
    height = np.concatenate([height, hedge[:, 3]])
    keep = (x >= 0) & (x < size_m) & (y >= 0) & (y < size_m)
    x, y, radius, height = x[keep], y[keep], radius[keep], height[keep]
    # the cell under the centre and under four points on the crown must be free of buildings
    # and a land class trees can stand on
    ok = np.ones(len(x), dtype=bool)
    for ox, oy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        col = np.clip(((x + ox * radius * 0.7) / gsd_m).astype(int), 0, side - 1)
        row = np.clip(((size_m - y - oy * radius * 0.7) / gsd_m).astype(int), 0, side - 1)
        ok &= mask[row, col] == 0
        if ox == 0 and oy == 0:
            ok &= np.isin(surface[row, col], TREE_CLASSES)
    ok &= (_wall_distance(x, y, wall_arr) > radius * 0.6 + 0.25) | (radius < 0.3)
    trees: list[list[Any]] = []
    for tx, ty, r, h in zip(x[ok], y[ok], radius[ok], height[ok], strict=True):
        tx, ty = round(float(tx), 2), round(float(ty), 2)
        trees.append(
            [
                tx,
                ty,
                round(float(r), 2),
                round(float(h), 2),
                species_for(biome, tx, ty, float(h), seed),
            ]
        )
    return trees


def _hedges(walls: Array, mask: Array, gsd_m: float, size_m: float, rng: Any) -> Array:
    """Rows of ``x y radius height``: small shrubs spaced along one wall of some compounds.

    A hedge keeps to the side of its wall whose midpoint is free of buildings. Long segments only.
    """
    rows: list[list[float]] = []
    side = mask.shape[0]
    for x0, y0, x1, y1 in walls:
        length = float(np.hypot(x1 - x0, y1 - y0))
        roll, flip = float(rng.uniform()), float(rng.choice([-1.0, 1.0]))
        if length < 6.0 or roll >= HEDGE_SHARE:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        nx, ny = -uy * flip * HEDGE_OFFSET_M, ux * flip * HEDGE_OFFSET_M
        mx, my = (x0 + x1) / 2 + nx, (y0 + y1) / 2 + ny
        col = min(max(int(mx / gsd_m), 0), side - 1)
        row = min(max(int((size_m - my) / gsd_m), 0), side - 1)
        if mask[row, col] > 0:
            nx, ny = -nx, -ny
        for d in np.arange(0.8, length - 0.8, HEDGE_SPACING_M):
            rows.append(
                [
                    x0 + ux * d + nx,
                    y0 + uy * d + ny,
                    float(rng.uniform(0.55, 0.9)),
                    float(rng.uniform(1.2, 1.9)),
                ]
            )
    return np.asarray(rows, dtype="float64").reshape(-1, 4)


__all__ = ["DEFAULT_MIX", "MIX", "SPECIES", "SHRUB_BELOW_M", "scatter", "species_for"]
