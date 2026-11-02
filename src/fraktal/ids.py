"""Deterministic randomness and stable feature IDs (ADR-0003)."""

from __future__ import annotations

import hashlib

import numpy as np


def substream(seed: int, name: str) -> np.random.Generator:
    """A named RNG stream derived from the world seed.

    Streams are independent: adding a new name never changes an existing stream's output.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


def feature_id(world_id: str, kind: str, *parts: int | str) -> str:
    """Stable ID for a feature, independent of generation order."""
    key = "|".join([world_id, kind, *map(str, parts)])
    return hashlib.sha256(key.encode()).hexdigest()[:16]
