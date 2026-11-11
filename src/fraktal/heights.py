"""Storey counts for the 3D path (FRK-62).

The world's heights serve the 2.5D renderer and stay as they are. The 3D path wants a plausible
mix of storeys per biome, so each building's height is remapped by rank: the tallest buildings in
the world stay the tallest, and the set as a whole follows a storey distribution.

These distributions are **assumptions**, written from general knowledge of the building stock
(mostly one to two storeys in a Sahelian low-rise city and in self-built settlements). There is no
real height data in this repository (SpaceNet and Open Cities give footprints only), so nothing
here is fitted. ``dense_urban`` keeps the world's own heights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal.ids import substream

STOREY_M = 3.0
# (storeys, share of buildings); shares sum to one.
PROFILES: dict[str, tuple[tuple[int, float], ...]] = {
    "lowrise": ((1, 0.45), (2, 0.35), (3, 0.12), (4, 0.05), (5, 0.03)),
    "informal": ((1, 0.70), (2, 0.27), (3, 0.03)),
    "city_lowrise": ((1, 0.10), (2, 0.38), (3, 0.30), (4, 0.17), (5, 0.05)),  # Manila-Quezon
    "midrise": ((3, 0.10), (4, 0.15), (6, 0.25), (8, 0.25), (10, 0.15), (12, 0.10)),  # Nairobi
}
BIOME_PROFILE = {
    "khartoum": "lowrise",
    "khartoum_city": "lowrise",
    "manila_quezon": "city_lowrise",
    "nairobi_centre": "midrise",
    "informal_settlement": "informal",
    "oil_palm_plantation": "informal",
    "paddy_terraces": "informal",
    "mangrove_coast": "informal",
}


class HeightError(ValueError):
    pass


def profile_for(biome: str) -> str | None:
    """The profile a biome uses in the 3D path, or None to keep the world's heights."""
    return BIOME_PROFILE.get(biome)


def remap(heights: Any, ids: list[str], profile: str, seed: int) -> Any:
    """New heights in metres: ``storeys * 3 m`` plus a 0 to 0.8 m parapet, chosen by height rank."""
    if profile not in PROFILES:
        raise HeightError(f"unknown height profile {profile!r}; use one of {sorted(PROFILES)}")
    h = np.asarray(heights, dtype="float64")
    n = len(h)
    if n != len(ids):
        raise HeightError("heights and ids must be the same length")
    if n == 0:
        return h
    order = np.lexsort((np.asarray(ids), h))  # ties break on id, so the result is stable
    quantile = np.empty(n)
    quantile[order] = (np.arange(n) + 0.5) / n
    counts = np.array([c for c, _ in PROFILES[profile]])
    edges = np.cumsum([w for _, w in PROFILES[profile]])
    storeys = counts[np.minimum(np.searchsorted(edges, quantile, side="right"), len(counts) - 1)]
    parapet = np.array([substream(seed, f"height:{i}").uniform(0.0, 0.8) for i in ids])
    return storeys * STOREY_M + parapet


__all__ = ["BIOME_PROFILE", "PROFILES", "HeightError", "profile_for", "remap"]
