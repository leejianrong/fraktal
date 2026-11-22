"""Procedural terrain: fractal height fields, and fine detail on top of a coarse real DEM.

Randomness comes only from named sub-streams (ADR-0003); the smooth noise is the
``value_noise`` kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.noise import lattice_shape, value_noise


def octave_noise(
    shape: tuple[int, int],
    gsd_m: float,
    seed: int,
    stream: str,
    largest_cell_m: float,
    smallest_cell_m: float,
    persistence: float,
) -> np.ndarray[Any, Any]:
    """Sum of noise octaves from ``largest_cell_m`` down to ``smallest_cell_m``, scaled to a
    peak-to-peak amplitude of about one."""
    if not 0 < persistence < 1:
        raise ValueError("persistence must be between 0 and 1")
    if smallest_cell_m <= 0 or largest_cell_m < smallest_cell_m:
        raise ValueError("cell sizes must satisfy 0 < smallest <= largest")
    field = np.zeros(shape, dtype=np.float32)
    total, cell_m, n, amp = 0.0, largest_cell_m, 0, 1.0
    while cell_m >= smallest_cell_m:
        cell_px = max(cell_m / gsd_m, 1.0)
        lattice = substream(seed, f"{stream}:{n}").uniform(
            -1, 1, size=lattice_shape(shape, cell_px)
        )
        field += np.float32(amp) * value_noise(lattice, cell_px, shape)
        total += amp
        cell_m, n, amp = cell_m / 2, n + 1, amp * persistence
    return field / np.float32(total)


def fractal_terrain(
    shape: tuple[int, int],
    gsd_m: float,
    seed: int,
    relief_m: float = 40.0,
    largest_cell_m: float = 800.0,
    smallest_cell_m: float = 4.0,
    persistence: float = 0.5,
) -> np.ndarray[Any, Any]:
    """A height field (metres, float32) with about ``relief_m`` of relief, from a seed alone."""
    field = octave_noise(
        shape, gsd_m, seed, "terrain", largest_cell_m, smallest_cell_m, persistence
    )
    out: np.ndarray[Any, Any] = field * np.float32(relief_m / 2)
    return out


def add_detail(
    dem: np.ndarray[Any, Any],
    gsd_m: float,
    seed: int,
    amplitude_m: float = 1.5,
    largest_cell_m: float = 30.0,
    smallest_cell_m: float = 2.0,
    persistence: float = 0.55,
) -> np.ndarray[Any, Any]:
    """``dem`` plus fine relief below the real data's resolution (a 30 m DEM has none finer)."""
    detail = octave_noise(
        dem.shape, gsd_m, seed, "terrain-detail", largest_cell_m, smallest_cell_m, persistence
    )
    out: np.ndarray[Any, Any] = dem.astype(np.float32) + detail * np.float32(amplitude_m / 2)
    return out


@dataclass(frozen=True)
class TerrainConfig:
    style: str = "rolling"  # "rolling" (the default fractal) or "arid" (ridges, mesas, gullies)
    gully_depth_share: float = 0.08  # arid: gully depth as a share of the relief


STYLES = ("rolling", "arid")


def arid_terrain(
    shape: tuple[int, int],
    gsd_m: float,
    seed: int,
    relief_m: float = 80.0,
    gully_depth_share: float = 0.08,
) -> np.ndarray[Any, Any]:
    """Dry badland relief (FRK-187): sharp ridge crests over broad swells, cut by steep gullies.

    Ridges are one minus the absolute value of a noise field, which folds it into crest lines. The
    gullies are narrow valleys where a second field crosses zero, deepest on the higher ground,
    so slopes run from flat wadi floors to steep scarps.
    """
    broad = octave_noise(shape, gsd_m, seed, "arid:broad", 900.0, 60.0, 0.5)
    ridge = 1.0 - 2.0 * np.abs(octave_noise(shape, gsd_m, seed, "arid:ridge", 700.0, 30.0, 0.55))
    height = 0.55 * np.clip(ridge, 0.0, 1.0) ** 1.6 + 0.45 * (broad + 0.5)
    fold = np.abs(octave_noise(shape, gsd_m, seed, "arid:gully", 140.0, 5.0, 0.55))
    gully = np.exp(-((fold / 0.035) ** 2))  # 1 on the gully axis, 0 a few metres away
    height = height - gully_depth_share * gully * (0.4 + 0.6 * np.clip(height, 0.0, 1.0))
    out: np.ndarray[Any, Any] = (height * np.float32(relief_m)).astype(np.float32)
    return out
