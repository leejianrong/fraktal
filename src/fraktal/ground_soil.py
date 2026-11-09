"""Bare-soil ground for the 3D path (FRK-108).

Real tropical ground is mostly red or ochre soil with crisp-edged plots, not an even green. This
turns a share of the grass, shrub and crop ground into soil patches with fine variation, and
mutes the rest of the green. Colours are set by eye from the reference board, in 0..1 reflectance;
nothing is fitted to real pixels (the board is evaluation only, ADR-0012). Randomness only through
``substream`` (ADR-0003).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.noise import lattice_shape, value_noise

Array = np.ndarray[Any, Any]

TREE_COVER, SHRUB, GRASS, CROP, BARE = 10, 20, 30, 40, 60
SOIL_CLASSES = (SHRUB, GRASS, CROP)  # ground that can turn to bare soil; trees and water do not


@dataclass(frozen=True)
class SoilStyle:
    soil_a: tuple[float, float, float]  # the two soil colours the patches blend between
    soil_b: tuple[float, float, float]
    share: float  # fraction of the soil-capable ground that becomes soil
    green_mute: float = 0.5  # how far the remaining green moves toward dry olive, 0..1
    bare_is_soil: bool = (
        False  # bare ground (class 60) takes the soil colour too, not the class cream
    )
    mute_tree_cover: bool = False  # tree-cover ground (class 10) is muted to dry olive too
    speckle: float = 0.0  # share of the ground under fine dark scrub tufts, 0 for none


LATERITE = ((140 / 255, 96 / 255, 74 / 255), (164 / 255, 116 / 255, 88 / 255))
OCHRE = ((190 / 255, 150 / 255, 104 / 255), (208 / 255, 172 / 255, 124 / 255))

# dry Sahelian red-brown earth
TAN = ((150 / 255, 108 / 255, 86 / 255), (172 / 255, 128 / 255, 102 / 255))

BIOME_SOIL: dict[str, SoilStyle] = {
    "khartoum": SoilStyle(
        *TAN, share=0.9, green_mute=0.8, bare_is_soil=True, mute_tree_cover=True, speckle=0.22
    ),
    "dense_urban": SoilStyle(
        *LATERITE, share=0.75, green_mute=0.9, bare_is_soil=True, mute_tree_cover=True, speckle=0.18
    ),
    "informal_settlement": SoilStyle(
        *LATERITE, share=0.85, green_mute=0.9, bare_is_soil=True, mute_tree_cover=True, speckle=0.2
    ),
    "oil_palm_plantation": SoilStyle(*LATERITE, share=0.3),
    "paddy_terraces": SoilStyle(*OCHRE, share=0.3),
    "mangrove_coast": SoilStyle(*OCHRE, share=0.25),
}
DEFAULT_SOIL = SoilStyle(*OCHRE, share=0.4)

# Other earth tones a tile may draw instead of the biome's base one (FRK-134), set by eye.
PALE_SAND = ((184 / 255, 152 / 255, 118 / 255), (204 / 255, 174 / 255, 138 / 255))
DEEP_RED = ((128 / 255, 80 / 255, 62 / 255), (150 / 255, 98 / 255, 76 / 255))
GREY_BROWN = ((130 / 255, 112 / 255, 96 / 255), (150 / 255, 130 / 255, 112 / 255))
TONES = (None, PALE_SAND, DEEP_RED, GREY_BROWN)  # None keeps the base colours of the biome
TONE_MIX: dict[str, tuple[float, float, float, float]] = {  # the shares of TONES, by biome
    "khartoum": (0.5, 0.25, 0.10, 0.15),
    "dense_urban": (0.5, 0.15, 0.2, 0.15),
    "informal_settlement": (0.5, 0.1, 0.3, 0.1),
}
DEFAULT_TONE_MIX = (0.7, 0.1, 0.1, 0.1)

DRY_OLIVE = np.array([0.46, 0.43, 0.30], dtype="float32")
TUFT = np.array([0.30, 0.31, 0.21], dtype="float32")  # dark dry scrub, set by eye
PATCH_CELLS_M = ((70.0, 0.45), (22.0, 0.3), (7.0, 0.15), (2.5, 0.1))  # ragged edges, not blobs
TUFT_CELLS_M = ((1.6, 0.6), (0.7, 0.4))
TUFT_ZONE_M = 45.0  # (noise cell, weight) of the patch field
EDGE = 0.035  # width of the soil edge in field units: crisp, not a blur


def _noise(seed: int, name: str, side: int, cell_m: float, gsd_m: float) -> Array:
    cell_px = max(cell_m / gsd_m, 1.0)
    shape = (side, side)
    lattice = substream(seed, name).uniform(-1, 1, size=lattice_shape(shape, cell_px))
    return np.asarray(value_noise(lattice, cell_px, shape), dtype="float32")


def soil_style(biome: str, seed: int | None = None) -> SoilStyle:
    """The biome's soil; with a ``seed`` the tile may draw another earth tone (own stream)."""
    style = BIOME_SOIL.get(biome, DEFAULT_SOIL)
    if seed is None:
        return style
    mix = TONE_MIX.get(biome, DEFAULT_TONE_MIX)
    roll = float(substream(seed, "soil:tone").uniform())
    tone = TONES[int(np.searchsorted(np.cumsum(mix), roll))]
    return style if tone is None else replace(style, soil_a=tone[0], soil_b=tone[1])


def apply_soil(ground: Array, surface: Array, seed: int, gsd_m: float, style: SoilStyle) -> Array:
    """Return ``ground`` with soil patches on the soil-capable classes of ``surface``."""
    side = surface.shape[0]
    capable = np.isin(surface, SOIL_CLASSES + ((BARE,) if style.bare_is_soil else ()))
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
    mute = capable | (style.mute_tree_cover and (surface == TREE_COVER))
    green = np.where(mute[..., None], muted, ground)
    out = np.clip(green * (1 - alpha) + soil * alpha, 0.0, 1.0)
    if style.speckle > 0:
        # fine dark tufts, thick in some zones and absent in others, as on dry scrub land
        tufts = sum(
            w * _noise(seed, f"soil:tuft{n}", side, c, gsd_m)
            for n, (c, w) in enumerate(TUFT_CELLS_M)
        )
        zone = 0.5 + 0.5 * _noise(seed, "soil:zone", side, TUFT_ZONE_M, gsd_m)
        cover = style.speckle * (0.3 + 1.4 * zone)
        hit = np.clip((tufts - (1.0 - 2.0 * cover) * 0.6) / 0.12, 0.0, 1.0)
        hit = np.where(capable, hit, 0.0).astype("float32")[..., None] * 0.55
        out = out * (1 - hit) + TUFT * hit
    return np.asarray(np.clip(out, 0.0, 1.0), dtype="float32")
