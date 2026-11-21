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

MUD = ((98 / 255, 80 / 255, 60 / 255), (120 / 255, 98 / 255, 74 / 255))  # wet paddy earth
MUD_DARK = ((84 / 255, 72 / 255, 58 / 255), (104 / 255, 90 / 255, 72 / 255))  # tidal mud

# dry Sahelian red-brown earth
TAN = ((150 / 255, 108 / 255, 86 / 255), (172 / 255, 128 / 255, 102 / 255))

PALE = ((158 / 255, 126 / 255, 94 / 255), (176 / 255, 144 / 255, 108 / 255))  # dry wadi sand

BIOME_SOIL: dict[str, SoilStyle] = {
    "arid_badlands": SoilStyle(*PALE, share=0.95, green_mute=0.9, bare_is_soil=True, speckle=0.18),
    "khartoum": SoilStyle(
        *TAN, share=0.9, green_mute=0.8, bare_is_soil=True, mute_tree_cover=True, speckle=0.22
    ),
    "khartoum_city": SoilStyle(
        *TAN, share=0.9, green_mute=0.8, bare_is_soil=True, mute_tree_cover=True, speckle=0.22
    ),
    "manila_quezon": SoilStyle(  # damp earth under canopy: less soil, the green stays green
        *LATERITE, share=0.5, green_mute=0.5, bare_is_soil=True, speckle=0.1
    ),
    "nairobi_centre": SoilStyle(
        *LATERITE, share=0.5, green_mute=0.6, bare_is_soil=True, speckle=0.1
    ),
    "dense_urban": SoilStyle(
        *LATERITE, share=0.75, green_mute=0.9, bare_is_soil=True, mute_tree_cover=True, speckle=0.18
    ),
    "informal_settlement": SoilStyle(
        *LATERITE, share=0.85, green_mute=0.9, bare_is_soil=True, mute_tree_cover=True, speckle=0.2
    ),
    "oil_palm_plantation": SoilStyle(  # earth between the palm rows and tracks, a little scrub
        *LATERITE, share=0.8, green_mute=0.85, bare_is_soil=True, speckle=0.08
    ),
    "paddy_terraces": SoilStyle(*MUD, share=0.6, green_mute=0.8, bare_is_soil=True),
    "mangrove_coast": SoilStyle(*MUD_DARK, share=0.5, green_mute=0.8, bare_is_soil=True),
}
DEFAULT_SOIL = SoilStyle(*OCHRE, share=0.4)

# City biomes whose local streets are sealed (FRK-152). The base road colour is a pale dusty street
# set against overhead chips; at eye level it renders about twice as bright as the photos' asphalt
# (road brightness 0.81 against 0.44 in Manila-Quezon), so these streets take an asphalt grey.
SEALED_STREETS = (
    "khartoum_city",
    "manila_quezon",
    "nairobi_centre",
    "planned_suburb",
    "planned_cul_de_sac",
)
SEALED_LOCAL_RGB = (84.0, 82.0, 82.0)

# Biomes with exposed rock (FRK-187): steep ground turns to banded rock; flat ground stays soil.
ROCKY_BIOMES = ("arid_badlands",)
ROCK_RGB = (122.0, 96.0, 78.0)
ROCK_FROM_DEG, ROCK_FULL_DEG = 7.0, 24.0
STRATA_M = 4.5  # height of one rock band
WADI_RGB = (216.0, 198.0, 166.0)  # pale dry sand on the floor of a gully
WADI_SCALE_M = 18.0  # a cell this far below the ground around it is on a channel floor

# Other earth tones a tile may draw instead of the biome's base one (FRK-134), set by eye.
# Keep these close to TAN: the sky fill makes anything lighter clip white or turn grey-lavender
PALE_SAND = ((150 / 255, 116 / 255, 90 / 255), (168 / 255, 134 / 255, 104 / 255))
DEEP_RED = ((128 / 255, 80 / 255, 62 / 255), (150 / 255, 98 / 255, 76 / 255))
GREY_BROWN = ((118 / 255, 94 / 255, 76 / 255), (136 / 255, 110 / 255, 90 / 255))
TONES = (None, PALE_SAND, DEEP_RED, GREY_BROWN)  # None keeps the base colours of the biome
TONE_MIX: dict[str, tuple[float, float, float, float]] = {  # the shares of TONES, by biome
    "khartoum": (0.5, 0.25, 0.10, 0.15),
    "dense_urban": (0.5, 0.15, 0.2, 0.15),
    "informal_settlement": (0.5, 0.1, 0.3, 0.1),
    "paddy_terraces": (1.0, 0.0, 0.0, 0.0),  # mud stays mud
    "mangrove_coast": (1.0, 0.0, 0.0, 0.0),
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


def seal_streets(ground: Any, surface: Any, biome: str, local_code: int) -> Any:
    """``ground`` with the local-road cells recoloured asphalt in a biome of ``SEALED_STREETS``."""
    if biome not in SEALED_STREETS:
        return ground
    asphalt = np.array(SEALED_LOCAL_RGB, dtype="float32") / 255.0
    return np.where((surface == local_code)[..., None], asphalt, ground).astype("float32")


# Streets with a kerb and a pavement (FRK-188): a pale strip beside the carriageway, set against it
# by a thin lighter kerb line. Only the planned subdivisions, where the reference tiles show them.
PAVED_STREETS = ("planned_suburb", "planned_cul_de_sac", "planned_patchwork")
PAVEMENT_RGB = (186.0, 182.0, 174.0)
KERB_RGB = (222.0, 219.0, 212.0)
PAVEMENT_M, KERB_M = 1.8, 0.5


def pavements(
    ground: Any, surface: Any, building: Any, biome: str, gsd_m: float, road_codes: Any
) -> Any:
    """``ground`` with a kerb line and pavement beside each street in a biome of ``PAVED_STREETS``.

    ``building`` marks footprint cells, which stay as they are; ``road_codes`` are the surface
    codes of carriageways.
    """
    if biome not in PAVED_STREETS:
        return ground
    from fraktal.coast import box_any

    road = np.isin(surface, road_codes)
    verge = box_any(road, max(1, round(PAVEMENT_M / gsd_m))) & ~road & ~(building > 0)
    kerb = box_any(road, max(1, round(KERB_M / gsd_m))) & ~road & verge
    out = np.where(verge[..., None], np.array(PAVEMENT_RGB, dtype="float32") / 255.0, ground)
    out = np.where(kerb[..., None], np.array(KERB_RGB, dtype="float32") / 255.0, out)
    return out.astype("float32")


def rock_relief(
    ground: Any, dem: Any, gsd_m: float, biome: str, bare: Any, blur: Any = None
) -> Any:
    """``ground`` with steep bare cells turned to banded rock in a biome of ``ROCKY_BIOMES``.

    Channel floors (wadis) turn pale sand. The share of rock rises with slope; the bands follow
    height, so contours read on a scarp.
    ``bare`` marks the cells that may turn to rock (bare ground, not vegetation or roads).
    """
    if biome not in ROCKY_BIOMES:
        return ground
    dy, dx = np.gradient(np.asarray(dem, dtype="float64"), gsd_m)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    share = np.clip((slope - ROCK_FROM_DEG) / (ROCK_FULL_DEG - ROCK_FROM_DEG), 0.0, 1.0)
    if blur is not None:  # the DEM cell is metres wide: soften the stair-steps in the slope
        share = blur(share, 4.0 / gsd_m)
    bands = 1.0 + 0.09 * np.sin(2.0 * np.pi * np.asarray(dem, dtype="float64") / STRATA_M)
    rock = (np.array(ROCK_RGB, dtype="float64") / 255.0) * bands[..., None]
    mixed = ground * (1.0 - share[..., None]) + rock * share[..., None]
    if (
        blur is not None
    ):  # wadis: cells below the surrounding ground, on gentle slope, fill with sand
        height = np.asarray(dem, dtype="float64")
        below = blur(height, WADI_SCALE_M / gsd_m) - height
        sand = np.clip((below - 0.5) / 1.5, 0.0, 1.0) * (1.0 - share)
        pale = np.array(WADI_RGB, dtype="float64") / 255.0
        mixed = mixed * (1.0 - sand[..., None]) + pale * sand[..., None]
    return np.where(bare[..., None], mixed, ground).astype("float32")


# Open water (FRK-187): colour by distance from the shore, a foam line, wet sand, and ripples.
SHALLOW_RGB = (74.0, 146.0, 142.0)
DEEP_RGB = (24.0, 56.0, 100.0)
FOAM_RGB = (226.0, 234.0, 230.0)
WET_SAND_RGB = (122.0, 106.0, 84.0)
SHORE_BLUR_M = 14.0  # width over which the depth colour and the wet band fade
RIPPLE_M = 7.0


def water_look(
    ground: Any, water: Any, gsd_m: float, blur: Any, ripple: Any, tide: bool = True
) -> Any:
    """``ground`` with open water shaded by depth, a foam line at the shore and wet sand beside it.

    ``water`` marks the water cells. ``blur(image, sigma_px)`` is a Gaussian-like blur and
    ``ripple`` a noise field in [-1, 1] that breaks the surface into wave streaks. The blurred
    water mask is about one half at the shore, so it stands in for distance from the shore.
    """
    if not water.any():
        return ground
    near = blur(water.astype("float32"), SHORE_BLUR_M / gsd_m)
    depth = np.clip((near - 0.5) / 0.3, 0.0, 1.0)[..., None]  # a narrow river never gets deeper
    shallow = np.array(SHALLOW_RGB, dtype="float32") / 255.0
    deep = np.array(DEEP_RGB, dtype="float32") / 255.0
    sea = shallow * (1.0 - depth) + deep * depth
    sea = sea * (1.0 + 0.07 * ripple[..., None])
    foam = np.exp(-(((near - 0.5) / 0.05) ** 2))[..., None] * 0.55
    sea = sea * (1.0 - foam) + (np.array(FOAM_RGB, dtype="float32") / 255.0) * foam
    out = np.where(water[..., None], sea, ground)
    wet = np.clip((near - 0.12) / 0.38, 0.0, 1.0)[..., None] * 0.7  # land within a few metres
    sand = np.array(WET_SAND_RGB, dtype="float32") / 255.0
    out = np.where(water[..., None], out, ground * (1.0 - wet) + sand * wet)
    return out.astype("float32")
