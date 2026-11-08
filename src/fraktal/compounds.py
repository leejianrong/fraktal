"""Walled compounds (FRK-106): a wall round a share of the parcels that hold a building, with a
gap for a gate on the side facing the road.

The walls are derived from the world's parcel, building and road layers when a scene is exported,
so the world files do not change and a world keeps its ids. One compound is one ``wall`` instance
in the label pass (ADR-0012 addendum); the gate is a gap, not an object. Randomness only through
``substream`` (ADR-0003).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.propgen import MATERIALS

Array = np.ndarray[Any, Any]

WALL_SHARE = {  # of the parcels with a building, set by eye from the board; the default is 0.3
    "khartoum": 0.55,
    "dense_urban": 0.4,
    "informal_settlement": 0.25,
    "oil_palm_plantation": 0.05,
    "paddy_terraces": 0.05,
    "mangrove_coast": 0.05,
}
DEFAULT_SHARE = 0.3
WALL_MATERIALS = (
    "wall-cream", "wall-ochre", "wall-grey", "wall-brick",
    "wall-white", "wall-sand", "wall-terracotta",
)  # fmt: skip
SHEET_MATERIALS = ("sheet-zinc", "sheet-rust", "sheet-blue")
# Shares of (block wall, block wall with piers, sheet-metal fence) by biome, set by eye
KIND_MIX = {
    "informal_settlement": (0.35, 0.15, 0.5),
    "khartoum": (0.6, 0.3, 0.1),
    "dense_urban": (0.55, 0.3, 0.15),
}
DEFAULT_KIND_MIX = (0.6, 0.25, 0.15)
KINDS = ("block", "piered", "sheet")
PIER_SPACING_M, PIER_M = 3.5, 0.45
YARD_TREE_SHARE = 0.3
INSET_M = 0.4  # the wall stands inside the parcel edge
THICKNESS_M = 0.22
SHEET_THICKNESS_M = 0.05
GATE_M = 3.0
MIN_AREA_M2 = 120.0


def share_for(biome: str) -> float:
    return WALL_SHARE.get(biome, DEFAULT_SHARE)


def build(
    parcels: Any,
    buildings: Any,
    roads: Any,
    origin: tuple[float, float],
    size_m: float,
    seed: int,
    share: float,
    biome: str = "",
) -> list[dict[str, Any]]:
    """Compounds as dicts: ``id``, ``kind``, ``height_m``, ``material``, ``segments`` (each
    [x0, y0, x1, y1] in tile metres) and ``yard`` (a free point inside, or None). The three layers
    are GeoDataFrames in the tile's CRS."""
    if share <= 0 or len(parcels) == 0:
        return []
    from shapely.geometry import box

    west, south = origin
    tile = box(west, south, west + size_m, south + size_m)
    built = set(map(str, buildings["parcel_id"])) if "parcel_id" in buildings else set()
    road_lines = {r.id: r.geometry for r in roads.itertuples()}
    out: list[dict[str, Any]] = []
    for parcel in parcels.sort_values("id").itertuples():
        rng = substream(seed, f"compound:{parcel.id}")
        roll, height = float(rng.uniform()), float(rng.uniform(1.8, 2.4))
        material = WALL_MATERIALS[int(rng.integers(0, len(WALL_MATERIALS)))]
        kind_roll, sheet_pick = float(rng.uniform()), int(rng.integers(0, len(SHEET_MATERIALS)))
        if roll >= share or str(parcel.id) not in built:
            continue
        inner = parcel.geometry.buffer(-INSET_M, join_style="mitre")
        if inner.is_empty or inner.geom_type != "Polygon" or inner.area < MIN_AREA_M2:
            continue
        if not inner.intersects(tile):
            continue
        pts = np.asarray(inner.exterior.coords)
        road = road_lines.get(getattr(parcel, "road_id", None))
        edges = list(zip(pts[:-1], pts[1:], strict=True))
        gate = 0
        if road is not None:
            gate = int(np.argmin([road.distance(_mid(a, b)) for a, b in edges]))
        segments: list[list[float]] = []
        for i, (a, b) in enumerate(edges):
            length = float(np.hypot(*(b - a)))
            if length < 0.5:
                continue
            if i != gate:
                segments.append(_seg(a, b, origin))
                continue
            gap = min(GATE_M, 0.6 * length)
            u, m = (b - a) / length, (a + b) / 2
            segments.append(_seg(a, m - u * gap / 2, origin))
            segments.append(_seg(m + u * gap / 2, b, origin))
        mix = KIND_MIX.get(biome, DEFAULT_KIND_MIX)
        kind = KINDS[0 if kind_roll < mix[0] else 1 if kind_roll < mix[0] + mix[1] else 2]
        if kind == "sheet":
            material, height = SHEET_MATERIALS[sheet_pick], min(height, 2.1)
        inside = buildings[buildings["parcel_id"].astype(str) == str(parcel.id)] if built else None
        free = inner.buffer(-1.0)
        if inside is not None and len(inside):
            free = free.difference(inside.geometry.union_all().buffer(2.0))
        yard = None
        if not free.is_empty and free.area > 25.0:
            point = free.representative_point()
            yard = [round(float(point.x - west), 2), round(float(point.y - south), 2)]
        out.append(
            {
                "id": str(parcel.id),
                "kind": kind,
                "height_m": round(height, 2),
                "material": material,
                "segments": segments,
                "yard": yard,
            }
        )
    return out


def _mid(a: Array, b: Array) -> Any:
    from shapely.geometry import Point

    return Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _seg(a: Array, b: Array, origin: tuple[float, float]) -> list[float]:
    return [
        round(float(a[0] - origin[0]), 2),
        round(float(a[1] - origin[1]), 2),
        round(float(b[0] - origin[0]), 2),
        round(float(b[1] - origin[1]), 2),
    ]


_BOX = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]]
)  # fmt: skip


def pack(compounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Arrays for Blender, like ``props_scene.pack``: a box per wall segment."""
    names = list(MATERIALS)
    verts: list[Array] = []
    faces: list[Array] = []
    mats: list[Array] = []
    owners: list[Array] = []
    offset = 0
    for index, c in enumerate(compounds):
        material = names.index(c["material"])

        def add(
            quad: list[tuple[float, float]],
            top: float,
            index: int = index,
            material: int = material,
        ) -> None:
            nonlocal offset
            box = [(x, y, 0.0) for x, y in quad] + [(x, y, top) for x, y in quad]
            verts.append(np.asarray(box, dtype="float32"))
            faces.append(_BOX.astype("int32") + offset)
            mats.append(np.full(len(_BOX), material, dtype="int16"))
            owners.append(np.full(len(_BOX), index, dtype="int32"))
            offset += 8

        for x0, y0, x1, y1 in c["segments"]:
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 0.05:
                continue
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            thick = SHEET_THICKNESS_M if c.get("kind") == "sheet" else THICKNESS_M
            nx, ny = -uy * thick / 2, ux * thick / 2
            add(
                [(x0 - nx, y0 - ny), (x1 - nx, y1 - ny), (x1 + nx, y1 + ny), (x0 + nx, y0 + ny)],
                c["height_m"],
            )
            if c.get("kind") == "piered":  # a post at each end and every few metres between
                steps = max(1, round(length / PIER_SPACING_M))
                for k in range(steps + 1):
                    px, py = x0 + ux * length * k / steps, y0 + uy * length * k / steps
                    h = PIER_M / 2
                    add(
                        [(px - h, py - h), (px + h, py - h), (px + h, py + h), (px - h, py + h)],
                        c["height_m"] + 0.25,
                    )
    if not verts:
        return {
            "vertices": np.zeros((0, 3), "float32"), "faces": np.zeros((0, 3), "int32"),
            "face_material": np.zeros(0, "int16"), "face_owner": np.zeros(0, "int32"),
            "materials": [],
        }  # fmt: skip
    return {
        "vertices": np.vstack(verts),
        "faces": np.vstack(faces),
        "face_material": np.concatenate(mats),
        "face_owner": np.concatenate(owners),
        "materials": [[n, list(MATERIALS[n][0]), MATERIALS[n][1], MATERIALS[n][2]] for n in names],
    }
