"""Which kind of tree stands where, for the 3D path (FRK-66).

Three shapes: ``broadleaf`` (a round crown on a trunk), ``palm`` (a bare trunk and a star of
fronds) and ``shrub`` (a low wide mound). A tree under 3.5 m is always a shrub; the others are drawn
from a per-biome mix with a sub-stream of the tree's own position, so a tree keeps its species when
the tile window or the tree count changes. The mixes are assumptions from general knowledge of each
setting, not counts from real imagery.
"""

from __future__ import annotations

from fraktal.ids import substream

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


__all__ = ["DEFAULT_MIX", "MIX", "SPECIES", "SHRUB_BELOW_M", "species_for"]
