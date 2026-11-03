"""A GPU-free realism gauge: how far synthetic chips are from real imagery.

V1's synthetic-only model scored about 0 IoU on real tiles (ADR-0006). These statistics say
where the renderer departs from real imagery, so V3 work can be steered without a training run.
Lower distances are better; 0 means the statistic matches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]
Chip = Array  # (3, H, W) uint8


def _channel_cdfs(chips: Sequence[Chip]) -> Array:
    """Pooled 256-bin CDF per channel, shape (3, 256)."""
    counts = np.zeros((3, 256), dtype="float64")
    for chip in chips:
        for band in range(3):
            counts[band] += np.bincount(chip[band].ravel(), minlength=256)[:256]
    return np.cumsum(counts, axis=1) / counts.sum(axis=1, keepdims=True)


def colour_distance(real: Sequence[Chip], synthetic: Sequence[Chip]) -> float:
    """Mean over channels of the Wasserstein-1 distance between pixel histograms, in 0-1."""
    gap = np.abs(_channel_cdfs(real) - _channel_cdfs(synthetic)).sum(axis=1) / 255.0
    return float(gap.mean())


def _grey(chip: Chip) -> Array:
    return chip.astype("float32").mean(axis=0)


def edge_energy(chip: Chip) -> float:
    """Mean absolute gradient of the grey image: how much local contrast there is."""
    grey = _grey(chip)
    return float((np.abs(np.diff(grey, axis=0)).mean() + np.abs(np.diff(grey, axis=1)).mean()) / 2)


def noise_level(chip: Chip) -> float:
    """Std of the Laplacian: fine texture and sensor noise that smooth renders lack."""
    g = _grey(chip)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.std())


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def _relative_gap(real: float, synthetic: float) -> float:
    return abs(synthetic - real) / max(real, 1e-9)


def compare(real: Sequence[Chip], synthetic: Sequence[Chip]) -> dict[str, Any]:
    """Distances and summary statistics for two chip sets. Both must be non-empty."""
    if not real or not synthetic:
        raise ValueError("need at least one real and one synthetic chip")
    stats: dict[str, Any] = {}
    for name, chips in (("real", real), ("synthetic", synthetic)):
        pixels = np.stack([c.reshape(3, -1) for c in chips], axis=0)
        stats[name] = {
            "chips": len(chips),
            "channel_mean": [round(float(v), 2) for v in pixels.mean(axis=(0, 2))],
            "channel_std": [round(float(v), 2) for v in pixels.std(axis=(0, 2))],
            "edge_energy": round(_mean([edge_energy(c) for c in chips]), 3),
            "noise_level": round(_mean([noise_level(c) for c in chips]), 3),
        }
    r, s = stats["real"], stats["synthetic"]
    return {
        **stats,
        "distances": {
            "colour_w1": round(colour_distance(real, synthetic), 4),
            "edge_energy_gap": round(_relative_gap(r["edge_energy"], s["edge_energy"]), 4),
            "noise_gap": round(_relative_gap(r["noise_level"], s["noise_level"]), 4),
        },
    }


def sample(chips: Sequence[Chip], limit: int) -> list[Chip]:
    """Evenly spaced subset, so the choice is deterministic and spans the set."""
    if len(chips) <= limit:
        return list(chips)
    picks = np.linspace(0, len(chips) - 1, limit).round().astype(int)
    return [chips[i] for i in picks]
