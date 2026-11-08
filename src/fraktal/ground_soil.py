"""Bare-soil ground for the 3D path (FRK-108).

Real tropical ground is mostly red or ochre soil with crisp-edged plots, not an even green. This
turns a share of the grass, shrub and crop ground into soil patches with fine variation, and
mutes the rest of the green. Colours are set by eye from the reference board, in 0..1 reflectance;
nothing is fitted to real pixels (the board is evaluation only, ADR-0012). Randomness only through
``substream`` (ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.noise import lattice_shape, value_noise

Array = np.ndarray[Any, Any]

SHRUB, GRASS, CROP = 20, 30, 40
SOIL_CLASSES = (SHRUB, GRASS, CROP)  # ground that can turn to bare soil; trees and water do not


@dataclass(frozen=True)
class SoilStyle:
    soil_a: tuple[float, float, float]  # the two soil colours the patches blend between
    soil_b: tuple[float, float, float]
    share: float  # fraction of the soil-capable ground that becomes soil
    green_mute: float = 0.5  # how far the remaining green moves toward dry olive, 0..1


LATERITE = ((158 / 255, 92 / 255, 64 / 255), (184 / 255, 116 / 255, 80 / 255))
OCHRE = ((190 / 255, 150 / 255, 104 / 255), (208 / 255, 172 / 255, 124 / 255))

# dry Sahelian red-brown earth
TAN = ((150 / 255, 108 / 255, 86 / 255), (172 / 255, 128 / 255, 102 / 255))

BIOME_SOIL: dict[str, SoilStyle] = {
    "khartoum": SoilStyle(*TAN, share=0.8, green_mute=0.65),
    "dense_urban": SoilStyle(*LATERITE, share=0.5),
    "informal_settlement": SoilStyle(*LATERITE, share=0.55),
    "oil_palm_plantation": SoilStyle(*LATERITE, share=0.3),
    "paddy_terraces": SoilStyle(*OCHRE, share=0.3),
    "mangrove_coast": SoilStyle(*OCHRE, share=0.25),
}
DEFAULT_SOIL = SoilStyle(*OCHRE, share=0.4)

DRY_OLIVE = np.array([0.50, 0.48, 0.30], dtype="float32")
PATCH_CELLS_M = ((70.0, 0.7), (22.0, 0.25), (7.0, 0.05))  # (noise cell, weight) of the patch field
EDGE = 0.035  # width of the soil edge in field units: crisp, not a blur


def _noise(seed: int, name: str, side: int, cell_m: float, gsd_m: float) -> Array:
    cell_px = max(cell_m / gsd_m, 1.0)
    shape = (side, side)
    lattice = substream(seed, name).uniform(-1, 1, size=lattice_shape(shape, cell_px))
    return np.asarray(value_noise(lattice, cell_px, shape), dtype="float32")


def soil_style(biome: str) -> SoilStyle:
    return BIOME_SOIL.get(biome, DEFAULT_SOIL)


def apply_soil(ground: Array, surface: Array, seed: int, gsd_m: float, style: SoilStyle) -> Array:
    """Return ``ground`` with soil patches on the soil-capable classes of ``surface``."""
    side = surface.shape[0]
    capable = np.isin(surface, SOIL_CLASSES)
    if not capable.any() or style.share <= 0:
        return ground
    field = np.zeros((side, side), dtype="float32")
    for n, (cell, weight) in enumerate(PATCH_CELLS_M):
        field += weight * _noise(seed, f"soil:patch{n}", side, cell, gsd_m)
    threshold = np.quantile(field[capable], 1.0 - style.share)
    alpha = np.clip((field - threshold) / EDGE + 0.5, 0.0, 1.0)
    alpha = np.where(capable, alpha, 0.0).astype("float32")[..., None]
    mix = (0.5 + 0.5 * _noise(seed, "soil:mix", side, 15.0, gsd_m))[..., None]
    soil = (
        np.array(style.soil_a, dtype="float32") * (1 - mix)
        + np.array(style.soil_b, "float32") * mix
    )
    tone = 1.0 + 0.12 * _noise(seed, "soil:tone", side, 3.0, gsd_m)[..., None]
    tone = tone * (1.0 + 0.06 * _noise(seed, "soil:grain", side, 0.6, gsd_m)[..., None])
    soil = np.clip(soil * tone, 0.0, 1.0)
    muted = ground * (1 - style.green_mute) + DRY_OLIVE * style.green_mute
    green = np.where(capable[..., None], muted, ground)
    return np.asarray(np.clip(green * (1 - alpha) + soil * alpha, 0.0, 1.0), dtype="float32")
