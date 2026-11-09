"""The light of one scene: sun colour, dusty sky and shadow-lifting fill (FRK-113).

Pure numbers, no Blender: ``draw`` returns the settings ``blender_scene.build_light`` turns into a
sun lamp and a world shader. Everything is drawn from one named stream per scene, so the same
seed gives the same light and a different seed a different day. Colours are set by eye against the
real reference board (warm hard light, pale dusty horizon), never fitted to real pixels.
"""

from __future__ import annotations

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


def _mix(a: tuple[float, ...], b: tuple[float, ...], t: float) -> list[float]:
    return [round(x + (y - x) * t, 4) for x, y in zip(a, b, strict=True)]


def draw(seed: int, sun_elevation_deg: float) -> dict[str, Any]:
    """Sun colour and strength, sky colours, dust and fill for a scene."""
    rng = substream(seed, "lighting")
    low = max(0.0, min(1.0, (60.0 - sun_elevation_deg) / 45.0))  # 0 at 60 degrees, 1 at 15
    warm = float(min(1.0, max(0.0, low + rng.uniform(-0.15, 0.15))))
    dust = float(rng.uniform(*DUST_RANGE))
    return {
        "sun_colour": _mix(SUN_HIGH, SUN_LOW, warm),
        "sun_strength": round(SUN_STRENGTH * float(rng.uniform(0.92, 1.08)), 3),
        "zenith": _mix(ZENITH, DUST, dust * 0.35),
        "horizon": _mix((0.62, 0.72, 0.9), DUST, dust),
        "glow": _mix((1.0, 0.85, 0.62), SUN_LOW, warm),
        "glow_amount": round(float(rng.uniform(*GLOW_RANGE)) * (0.6 + 0.8 * dust), 3),
        "fill": round(float(rng.uniform(*FILL_RANGE)), 3),
    }


__all__ = ["draw"]
