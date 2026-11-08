"""Per-scene realism randomisation from named sub-streams (ADR-0003, Q13).

Each component (sun, haze, noise, ...) draws from its own stream, so changing one component's
range never changes what another draws. A fixed value is a range with equal ends. A region is
one scene: it is rendered once and tiled, so its parameters apply to all of its tiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from fraktal.ids import substream
from fraktal.render import AMBIENT, KHARTOUM_TONE, RenderConfig

Range = tuple[float, float]


@dataclass(frozen=True)
class Ranges:
    sun_azimuth_deg: Range = (90.0, 200.0)
    sun_elevation_deg: Range = (35.0, 70.0)
    haze: Range = (0.0, 0.15)
    penumbra_m: Range = (0.0, 0.6)
    psf_sigma_px: Range = (0.5, 1.2)
    noise_sigma: Range = (4.0, 12.0)
    texture: Range = (0.5, 0.9)
    variation: Range = (0.10, 0.25)
    # Extremes (FRK-33). The defaults are fixed at the neutral render values, so the original
    # randomised variant draws exactly what it did before; WIDE opens them up.
    ambient: Range = (AMBIENT, AMBIENT)
    contrast: Range = (1.0, 1.0)
    bright_roofs: Range = (0.0, 0.0)
    blank_prob: Range = (0.0, 0.0)
    cloud_cover: Range = (0.0, 0.0)  # share of the image under cloud (FRK-42)
    tone_jitter: Range = (0.9, 1.1)  # multiplies each channel of the base tone
    base_tone: tuple[float, float, float] = field(default=KHARTOUM_TONE)


WIDE = Ranges(
    ambient=(0.12, 0.30),
    contrast=(1.1, 1.9),
    bright_roofs=(0.05, 0.25),
    blank_prob=(0.0, 0.3),
)

# Humid tropics (FRK-42): greener, hazier, softer light under a higher sun, often cloudy. The tone
# is a neutral guess, not fitted to real tropical imagery (that needs the Q15 test set).
TROPICAL_TONE: tuple[float, float, float] = (0.92, 1.0, 0.86)
TROPICAL = Ranges(
    sun_elevation_deg=(55.0, 85.0),
    haze=(0.12, 0.35),
    ambient=(0.30, 0.45),
    contrast=(0.9, 1.3),
    bright_roofs=(0.05, 0.2),
    blank_prob=(0.0, 0.1),
    cloud_cover=(0.0, 0.7),
    base_tone=TROPICAL_TONE,
)

# Dry Sahelian light (FRK-120): high sun, little haze, no cloud, a near-neutral tone so tan ground
# stays tan. The benchmark chips keep KHARTOUM_TONE through the default ``Ranges``.
ARID = Ranges(
    sun_elevation_deg=(55.0, 85.0),
    haze=(0.05, 0.15),
    ambient=(0.30, 0.45),
    contrast=(0.95, 1.2),
    bright_roofs=(0.05, 0.2),
    base_tone=(1.0, 0.98, 0.94),
)
BIOME_RANGES: dict[str, Ranges] = {"khartoum": ARID}

COMPONENTS = (
    "sun_azimuth_deg",
    "sun_elevation_deg",
    "haze",
    "penumbra_m",
    "psf_sigma_px",
    "noise_sigma",
    "texture",
    "variation",
    "ambient",
    "contrast",
    "bright_roofs",
    "blank_prob",
    "cloud_cover",
)


def _draw(seed: int, scene: int, name: str, bounds: Range, count: int = 1) -> list[float]:
    lo, hi = bounds
    if hi < lo:
        raise ValueError(f"range for {name} is reversed: {bounds}")
    # The unit draw does not depend on the range, so editing one range moves only that value.
    unit = substream(seed, f"realism:{scene}:{name}").uniform(size=count)
    return [float(lo + u * (hi - lo)) for u in unit]


def sample_config(base: RenderConfig, ranges: Ranges, seed: int, scene: int) -> RenderConfig:
    """``base`` with each realism parameter drawn for scene number ``scene``."""
    values: dict[str, Any] = {
        name: _draw(seed, scene, name, getattr(ranges, name))[0] for name in COMPONENTS
    }
    jitter = _draw(seed, scene, "tone", ranges.tone_jitter, count=3)
    tone = (
        ranges.base_tone[0] * jitter[0],
        ranges.base_tone[1] * jitter[1],
        ranges.base_tone[2] * jitter[2],
    )
    scene_seed = int(substream(seed, f"realism:{scene}:seed").integers(0, 2**31 - 1))
    return replace(base, seed=scene_seed, tone=tone, **values)
