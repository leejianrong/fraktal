"""Overhead cables between the roadside poles of a city street (FRK-147).

The photos of Manila, Nairobi and Khartoum are crossed by wires: bundles of them from pole to pole
and across the road, a large part of why real frames have twice the fine detail of ours. Each wire
is a thin ribbon in segments that follows a shallow parabola between two poles. The ribbons are
packed like the compound walls (one mesh in Blender). A biome takes part by having an entry in
``STYLES``; the others are unchanged.

Pure geometry from the pole positions, with a sub-stream per span for the sag, so the same world
gives the same wires.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraktal.ids import substream

Array = np.ndarray[Any, Any]

MATERIALS: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "cable": ((0.03, 0.03, 0.035), 0.6, 0.0),
}
WIRE_M = 0.06  # ribbon width: about a pixel at 30 m in a 75 degree frame of 1000 px
SEGMENTS = 10  # straight pieces per wire
SAG_RATIO = (0.015, 0.035)  # sag as a share of the span
MIN_SPAN_M = 6.0
NEIGHBOURS = 2  # each pole is wired to the nearest two poles further along the list
# per biome: (pole spacing in metres, pole height, wires per span, widest span as a multiple of
# the spacing)
STYLES: dict[str, tuple[float, float, int, float]] = {
    "manila_quezon": (28.0, 9.0, 5, 2.4),  # dense bundles on concrete poles
    "nairobi_centre": (36.0, 9.0, 3, 2.4),
    "khartoum_city": (45.0, 8.5, 2, 2.4),
}


def pole_plan(biome: str, default_spacing_m: float, default_height_m: float) -> tuple[float, float]:
    """(spacing, height) of the roadside poles in ``biome``."""
    style = STYLES.get(biome)
    return (style[0], style[1]) if style else (default_spacing_m, default_height_m)


def build(poles: list[list[float]], biome: str, seed: int) -> list[tuple[Array, Array]]:
    """Every wire between ``poles`` (x, y, top) as ``(polyline, normal)``.

    The polyline is ``(SEGMENTS + 1, 3)`` points; the normal is the horizontal unit vector across
    the span, along which the ribbon gets its width. Empty for a biome without a style.
    """
    style = STYLES.get(biome)
    if style is None or len(poles) < 2:
        return []
    spacing, _, wires, reach = style
    xy = np.array([[p[0], p[1]] for p in poles], dtype="float64")
    top = np.array([p[2] for p in poles], dtype="float64")
    out: list[tuple[Array, Array]] = []
    for i in range(len(poles)):
        dist = np.hypot(*(xy[i + 1 :] - xy[i]).T) if i + 1 < len(poles) else np.zeros(0)
        near = [j for j in np.argsort(dist) if MIN_SPAN_M < dist[j] <= spacing * reach][:NEIGHBOURS]
        for j in near:
            k = i + 1 + int(j)
            rng = substream(seed, f"cables:{i}:{k}")
            a, b = xy[i], xy[k]
            span = float(np.hypot(*(b - a)))
            sag_ratio = float(rng.uniform(*SAG_RATIO))
            direction = (b - a) / span
            normal = np.array([-direction[1], direction[0], 0.0])
            for w in range(wires):
                drop = 0.3 + 0.32 * w + float(rng.uniform(-0.05, 0.05))
                lateral = float(rng.uniform(-0.25, 0.25))
                t = np.linspace(0.0, 1.0, SEGMENTS + 1)
                px = a[0] + (b[0] - a[0]) * t + normal[0] * lateral * (4 * t * (1 - t))
                py = a[1] + (b[1] - a[1]) * t + normal[1] * lateral * (4 * t * (1 - t))
                pz = (top[i] + (top[k] - top[i]) * t) - drop - span * sag_ratio * 4 * t * (1 - t)
                out.append((np.column_stack([px, py, pz]), normal))
    return out


def pack(wires: list[tuple[Array, Array]]) -> dict[str, Any]:
    """Arrays for Blender, laid out like ``markings.pack``: two triangles per ribbon piece."""
    materials = [[n, list(m[0]), m[1], m[2]] for n, m in MATERIALS.items()]
    if not wires:
        return {
            "vertices": np.zeros((0, 3), "float32"), "faces": np.zeros((0, 3), "int32"),
            "face_material": np.zeros(0, "int16"), "face_owner": np.zeros(0, "int32"),
            "materials": materials,
        }  # fmt: skip
    verts, faces = [], []
    base = 0
    for line, normal in wires:
        half = normal * (WIRE_M / 2.0)
        left, right = line + half, line - half
        verts += [left, right]
        n = len(line)
        for s in range(n - 1):
            a, b, c, d = base + s, base + n + s, base + n + s + 1, base + s + 1
            faces += [[a, b, c], [a, c, d]]
        base += 2 * n
    return {
        "vertices": np.vstack(verts).astype("float32"),
        "faces": np.asarray(faces, dtype="int32"),
        "face_material": np.zeros(len(faces), dtype="int16"),
        "face_owner": np.zeros(len(faces), dtype="int32"),
        "materials": materials,
    }


def span_length(wires: list[tuple[Array, Array]]) -> float:
    """Total wire length in metres, for a quick count in the scene stats."""
    return float(sum(math.fsum(np.hypot(*np.diff(w[:, :2], axis=0).T)) for w, _ in wires))


__all__ = ["MATERIALS", "STYLES", "build", "pack", "pole_plan", "span_length"]
