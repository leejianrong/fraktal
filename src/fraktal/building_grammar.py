"""A building grammar for the 3D path: one typology, the flat-roof block (ADR-0010, FRK-49).

``build`` turns a footprint ring, a height and a roof material into a polygon mesh: walls, a flat
roof with a parapet, window rows per storey, a street door, and roof clutter (a water tank, a
stair box). Every choice is drawn from ``substream(seed, "building:<id>")``, so a building looks
the same wherever and whenever it is built. The mesh is plain arrays (the scene is data: the Blender
side only reads them), and each face carries the name of a material from ``assets/materials``.

One typology only. Other roof shapes and the courtyard house are future work; a footprint that
cannot be built (fewer than three points, no area) raises ``ValueError``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from fraktal.ids import substream

WALLS = ("wall-plaster-pale", "wall-plaster-sand", "wall-plaster-ochre", "wall-concrete")
ROOFS = {"tin": "roof-tin", "tile": "roof-tile"}
FLAT_ROOFS = ("roof-flat-concrete", "roof-flat-pale", "roof-flat-terracotta", "roof-flat-bitumen")
OPENINGS = ("window-glass", "door")
CLUTTER = ("tank",)
MATERIALS = (*WALLS, *FLAT_ROOFS, *ROOFS.values(), *OPENINGS, *CLUTTER)
STOREY_M = 3.0
PARAPET_M = (0.35, 0.7)
PARAPET_THICKNESS_M = 0.18
WINDOW_M = (1.1, 1.3)  # width, height
SILL_M = 0.9
MIN_EDGE_FOR_WINDOWS_M = 3.0
POP_OUT_M = 0.03  # windows and doors sit this far proud of the wall so they are not coplanar


@dataclass
class Mesh:
    """A polygon soup: ``faces`` index ``vertices``; ``materials`` names each face's material."""

    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)

    def add(self, points: Sequence[tuple[float, float, float]], material: str) -> None:
        base = len(self.vertices)
        self.vertices.extend(points)
        self.faces.append(tuple(range(base, base + len(points))))
        self.materials.append(material)


def _inset(ring: np.ndarray, poly: Polygon, distance: float) -> np.ndarray | None:
    """The ring moved inward by ``distance`` along each corner's bisector, or None if it folds.

    The inset has one vertex per footprint vertex, so the parapet can be stitched edge to edge.
    """
    n = len(ring)
    edges = np.roll(ring, -1, axis=0) - ring  # CCW ring: the inward normal is to the left
    length = np.maximum(np.hypot(edges[:, 0], edges[:, 1]), 1e-9)
    inward = np.column_stack([-edges[:, 1], edges[:, 0]]) / length[:, None]
    bisector = inward + np.roll(inward, 1, axis=0)
    norm = np.maximum(np.hypot(bisector[:, 0], bisector[:, 1]), 1e-9)
    cos_half = np.clip(norm / 2.0, 0.3, 1.0)  # limits the miter on sharp corners
    moved = ring + bisector / norm[:, None] * (distance / cos_half)[:, None]
    inner = Polygon(moved)
    if n < 3 or not inner.is_valid or inner.area < 1.0 or not poly.contains(inner):
        return None
    return moved  # type: ignore[no-any-return]


def _quad(a: Any, b: Any, z0: float, z1: float) -> list[tuple[float, float, float]]:
    return [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]


def _prism(cx: float, cy: float, radius: float, z0: float, z1: float, sides: int, mesh: Mesh,
           material: str) -> None:  # fmt: skip
    angles = np.linspace(0.0, 2 * math.pi, sides, endpoint=False)
    ring = [(cx + radius * math.cos(t), cy + radius * math.sin(t)) for t in angles]
    for i in range(sides):
        mesh.add(_quad(ring[i], ring[(i + 1) % sides], z0, z1), material)
    mesh.add([(x, y, z1) for x, y in ring], material)


def _box(cx: float, cy: float, w: float, d: float, z0: float, z1: float, mesh: Mesh,
         material: str) -> None:  # fmt: skip
    c = [(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2), (cx + w / 2, cy + d / 2),
         (cx - w / 2, cy + d / 2)]  # fmt: skip
    for i in range(4):
        mesh.add(_quad(c[i], c[(i + 1) % 4], z0, z1), material)
    mesh.add([(x, y, z1) for x, y in c], material)


def build(
    building_id: str,
    ring: Sequence[Sequence[float]],
    height_m: float,
    roof_material: str | None,
    seed: int,
) -> Mesh:
    """The mesh of one building, in the ring's own (tile) coordinates, ground at z = 0."""
    pts = np.asarray(ring, dtype="float64")
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    poly = Polygon(pts) if len(pts) >= 3 else None
    if poly is None or not poly.is_valid or poly.area < 1.0:
        raise ValueError(f"building {building_id}: footprint has no usable area")
    poly = orient(poly, 1.0)
    pts = np.asarray(poly.exterior.coords[:-1], dtype="float64")
    n = len(pts)
    rng = substream(seed, f"building:{building_id}")
    wall = WALLS[int(rng.integers(0, len(WALLS)))]
    flat = roof_material not in ROOFS
    roof = FLAT_ROOFS[int(rng.integers(0, len(FLAT_ROOFS)))] if flat else ROOFS[str(roof_material)]
    height = max(2.5, float(height_m))
    floors = max(1, round(height / STOREY_M))
    storey = height / floors
    mesh = Mesh()

    for i in range(n):  # walls
        mesh.add(_quad(pts[i], pts[(i + 1) % n], 0.0, height), wall)

    top = height
    inner = None
    if flat:  # parapet: an outer lip, an inner face and a cap, stitched edge to edge
        parapet = float(rng.uniform(*PARAPET_M))
        inner = _inset(pts, poly, PARAPET_THICKNESS_M)
        if inner is not None:
            for i in range(n):
                j = (i + 1) % n
                mesh.add(_quad(pts[i], pts[j], height, height + parapet), wall)
                cap = [(pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]),
                       (inner[j][0], inner[j][1]), (inner[i][0], inner[i][1])]  # fmt: skip
                mesh.add([(x, y, height + parapet) for x, y in cap], wall)
                mesh.add(_quad(inner[j], inner[i], height, height + parapet), wall)
    surface = inner if inner is not None else pts
    mesh.add([(x, y, top) for x, y in surface], roof)

    # Openings: a window grid per edge long enough, a door on the longest edge.
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    spacing = float(rng.uniform(2.6, 3.4))
    fill = float(rng.uniform(0.75, 1.0))
    door_edge = int(np.argmax(lengths))
    for i in range(n):
        length = float(lengths[i])
        if length < MIN_EDGE_FOR_WINDOWS_M:
            continue
        direction = edges[i] / length
        outward = np.array([direction[1], -direction[0]])  # CCW ring: outward is to the right
        cols = max(1, int((length - 1.0) // spacing))
        for c in range(cols):
            along = (c + 0.5) * length / cols
            if i == door_edge and c == cols // 2:  # the door takes this bay's ground floor
                half = 0.5
                a = pts[i] + direction * (along - half) + outward * POP_OUT_M
                b = pts[i] + direction * (along + half) + outward * POP_OUT_M
                mesh.add(_quad(a, b, 0.0, 2.1), "door")
            for f in range(floors):
                if f == 0 and i == door_edge and c == cols // 2:
                    continue
                if rng.uniform() > fill:
                    continue
                z0 = f * storey + SILL_M
                if z0 + WINDOW_M[1] > (f + 1) * storey:
                    continue
                half = WINDOW_M[0] / 2
                a = pts[i] + direction * (along - half) + outward * POP_OUT_M
                b = pts[i] + direction * (along + half) + outward * POP_OUT_M
                mesh.add(_quad(a, b, z0, z0 + WINDOW_M[1]), "window-glass")

    if flat and poly.area >= 60.0:  # roof clutter, kept clear of the parapet
        room = poly.buffer(-1.8)
        minx, miny, maxx, maxy = poly.bounds
        placed = 0
        for _ in range(30):
            if placed >= 2 or room.is_empty:
                break
            x, y = float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy))
            if not room.contains(Point(x, y)):
                continue
            if placed == 0:
                _prism(x, y, float(rng.uniform(0.8, 1.1)), top, top + 1.3, 8, mesh, "tank")
            elif floors >= 2:
                _box(x, y, 2.2, 2.2, top, top + 2.4, mesh, wall)
            placed += 1
    return mesh


def pack(meshes: Sequence[Mesh]) -> dict[str, Any]:
    """Several meshes as flat arrays: ``vertices`` (N, 3), ``face_sizes``, ``face_verts``,
    ``face_material`` (an index into ``materials``), and the ``materials`` names."""
    names = list(MATERIALS)
    vertices: list[Any] = []
    sizes: list[int] = []
    verts: list[int] = []
    mats: list[int] = []
    offset = 0
    for mesh in meshes:
        vertices.extend(mesh.vertices)
        for face, material in zip(mesh.faces, mesh.materials, strict=True):
            sizes.append(len(face))
            verts.extend(offset + i for i in face)
            mats.append(names.index(material))
        offset += len(mesh.vertices)
    return {
        "vertices": np.asarray(vertices, dtype="float32").reshape(-1, 3),
        "face_sizes": np.asarray(sizes, dtype="int32"),
        "face_verts": np.asarray(verts, dtype="int32"),
        "face_material": np.asarray(mats, dtype="int16"),
        "materials": names,
    }


__all__ = ["MATERIALS", "Mesh", "build", "pack"]
