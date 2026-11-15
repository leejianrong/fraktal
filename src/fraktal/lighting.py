"""The light of one scene: sun colour, dusty sky and shadow-lifting fill (FRK-113).

Pure numbers, no Blender: ``draw`` returns the settings ``blender_scene.build_light`` turns into a
sun lamp and a world shader. Everything is drawn from one named stream per scene, so the same
seed gives the same light and a different seed a different day. Colours are set by eye against the
real reference board (warm hard light, pale dusty horizon), never fitted to real pixels.
"""

from __future__ import annotations

import math
from typing import Any

from fraktal.ids import substream

SUN_LOW = (1.0, 0.62, 0.38)  # a low sun: orange
SUN_HIGH = (1.0, 0.93, 0.82)  # a high sun: warm white
ZENITH = (0.30, 0.46, 0.80)  # a clear zenith
DUST = (0.86, 0.78, 0.64)  # the horizon with dust in the air
SUN_STRENGTH = 2.3
FILL_RANGE = (1.3, 1.9)  # sky fill: shadows keep texture because the sky lights them
DUST_RANGE = (0.2, 0.8)  # how much of the horizon is dust rather than blue sky
GLOW_RANGE = (0.2, 0.7)  # the warm bloom around the sun
# Eye-level views (FRK-152). The light above is set for ground seen from overhead; from the street
# the same light renders 20 to 50 percent brighter than the photos, and the sky clips to white
# (Manila-Quezon: building 0.53 against 0.43, sky 1.00 against 0.88). These scale the sun, the
# sky's light, and how bright the sky looks when seen directly (a multiplier on ``fill``), for
# perspective frames near the horizon only.
EYE_LEVEL = {"sun": 0.8, "fill": 0.8, "sky_seen": 0.6}
# A city can set its own light (FRK-155). Hazy: (least dust, most dust, how far the zenith pales).
# Khartoum's photos have a pale grey-yellow sky with no blue in it. Nairobi's streets are 18 to 40 m
# canyons, so its sun and sky are not turned down as far (0.35 against 0.49 in the photos).
HAZY: dict[str, tuple[float, float, float]] = {"khartoum_city": (0.85, 1.0, 0.8)}
EYE_LEVEL_BY_BIOME: dict[str, dict[str, float]] = {
    "nairobi_centre": {"sun": 1.1, "fill": 1.3, "sky_seen": 0.6},
    "manila_quezon": {"sun": 0.9, "fill": 0.92, "sky_seen": 0.6},
}
# Light variety for street views (FRK-168), from its own stream so the light above is unchanged.
# The sun's height offset in degrees: a lower sun is warmer with longer shadows.
SUN_OFFSET_DEG = (-5.0, 8.0)
CLOUD_COVER = (0.0, 0.3)  # share of the sky with cloud; under a cloud the sun is dimmed
CLOUD_SCALE_M = (150.0, 450.0)  # width of one cloud shadow on the ground
CLOUD_SUN_BLOCKED = 0.4  # share of the sun a cloud takes
HAZE_DENSITY = (0.0, 0.0003)  # per metre: scattering in the air, a hint of dust
# Khartoum's dust: 0.0012 per metre is about 800 m of visibility.
HAZY_DENSITY: dict[str, tuple[float, float]] = {"khartoum_city": (0.0004, 0.0012)}
SUN_GAIN_MAX = 1.5  # most the sun is strengthened to make up for a lower sun
EYE_LEVEL_MIN_OFF_NADIR_DEG = 75.0  # a camera at least this far from straight down is eye level


def _mix(a: tuple[float, ...], b: tuple[float, ...], t: float) -> list[float]:
    return [round(x + (y - x) * t, 4) for x, y in zip(a, b, strict=True)]


def _warmth(sun_elevation_deg: float, jitter: float) -> float:
    low = max(0.0, min(1.0, (60.0 - sun_elevation_deg) / 45.0))  # 0 at 60 degrees, 1 at 15
    return float(min(1.0, max(0.0, low + jitter)))


def variety(seed: int, sun_elevation_deg: float, biome: str = "") -> dict[str, Any]:
    """What a street view adds to the light: a sun height offset and its colour, cloud shadows and
    haze in the air. ``blender_scene`` applies it to eye-level frames only."""
    rng = substream(seed, "lighting:variety")
    offset = float(rng.uniform(*SUN_OFFSET_DEG))
    el = max(8.0, sun_elevation_deg + offset)
    warm = _warmth(el, float(rng.uniform(-0.15, 0.15)))
    cover = float(rng.uniform(*CLOUD_COVER))
    # Ground gets sin(elevation) of the sun's light, so a lower sun is strengthened to keep the
    # street's exposure; what changes between seeds is the colour and the shadow length.
    gain = min(SUN_GAIN_MAX, math.sin(math.radians(sun_elevation_deg)) / math.sin(math.radians(el)))
    return {
        "sun_offset_deg": round(el - sun_elevation_deg, 3),
        "sun_gain": round(gain, 3),
        "sun_colour": _mix(SUN_HIGH, SUN_LOW, warm),
        "cloud": {
            "cover": round(cover, 3),
            "scale_m": round(float(rng.uniform(*CLOUD_SCALE_M)), 1),
            "offset_m": [round(float(v), 1) for v in rng.uniform(0.0, 2000.0, 2)],
            "blocked": CLOUD_SUN_BLOCKED,
        },
        "haze_density": round(float(rng.uniform(*HAZY_DENSITY.get(biome, HAZE_DENSITY))), 6),
    }


def draw(seed: int, sun_elevation_deg: float, biome: str = "") -> dict[str, Any]:
    """Sun colour and strength, sky colours, dust and fill for a scene."""
    rng = substream(seed, "lighting")
    warm = _warmth(sun_elevation_deg, float(rng.uniform(-0.15, 0.15)))
    lo, hi, pale = HAZY.get(biome, (*DUST_RANGE, 0.35))
    dust = float(rng.uniform(lo, hi))
    return {
        "sun_colour": _mix(SUN_HIGH, SUN_LOW, warm),
        "sun_strength": round(SUN_STRENGTH * float(rng.uniform(0.92, 1.08)), 3),
        "zenith": _mix(ZENITH, DUST, dust * pale),
        "horizon": _mix((0.62, 0.72, 0.9), DUST, dust),
        "glow": _mix((1.0, 0.85, 0.62), SUN_LOW, warm),
        "glow_amount": round(float(rng.uniform(*GLOW_RANGE)) * (0.6 + 0.8 * dust), 3),
        "fill": round(float(rng.uniform(*FILL_RANGE)), 3),
        "eye_level": dict(EYE_LEVEL_BY_BIOME.get(biome, EYE_LEVEL)),
        "variety": variety(seed, sun_elevation_deg, biome),
    }


__all__ = ["EYE_LEVEL", "EYE_LEVEL_BY_BIOME", "HAZY", "EYE_LEVEL_MIN_OFF_NADIR_DEG", "draw"]
