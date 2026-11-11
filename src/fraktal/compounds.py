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
# wall colour -> the plaster or concrete in assets/materials/ that it is tinted onto
TEXTURE_BASE = {
    "wall-cream": "wall-plaster-pale", "wall-white": "wall-plaster-pale",
    "wall-sand": "wall-plaster-sand", "wall-ochre": "wall-plaster-ochre",
    "wall-grey": "wall-concrete", "wall-brick": "wall-plaster-cracked",
    "wall-terracotta": "wall-paint-peeling",
}  # fmt: skip
# Mean RGB of each base texture file, measured on the JPEGs: tint = colour / mean, so a wall
# averages the colour it was given (a guessed 0.75 left plaster 10% too bright and the generated
# plaster and paint 20 to 25% too dark). ``test_compounds`` re-measures them to catch drift.
TEXTURE_MEAN = {
    "wall-plaster-pale": (0.843, 0.828, 0.815),
    "wall-plaster-sand": (0.843, 0.828, 0.815),
    "wall-plaster-ochre": (0.843, 0.828, 0.815),
    "wall-concrete": (0.723, 0.723, 0.723),
    "wall-plaster-cracked": (0.652, 0.603, 0.547),
    "wall-paint-peeling": (0.559, 0.575, 0.562),
}
TINT_MAX = 1.6
CAP_M, CAP_OVERHANG_M = 0.08, 0.05  # the coping stone on a block wall
INSET_M = 0.4  # the wall stands inside the parcel edge
THICKNESS_M = 0.22
SHEET_THICKNESS_M = 0.05
GATE_M = 3.0
MIN_AREA_M2 = 120.0
IRREGULAR_SHARE = 0.7  # of compounds that get slanted, chamfered, notched or kinked walls
CUT_COUNTS = (0.1, 0.3, 0.35, 0.25)  # shares that take 0, 1, 2 or 3 cuts
ROUND_SHARE = 0.35  # of reshaped compounds that also get rounded corners
ARC_STEPS = 8
MIN_KEPT_SHARE = 0.55  # of the lot the walls must still enclose


def textures() -> dict[str, list[Any]]:
    """For Blender: wall colour name -> [material file, tint]. Sheet metal stays a flat colour."""
    out: dict[str, list[Any]] = {}
    for name, base in TEXTURE_BASE.items():
        pairs = zip(MATERIALS[name][0], TEXTURE_MEAN[base], strict=True)
        tint = [round(min(c / m, TINT_MAX), 3) for c, m in pairs]
        out[name] = [base, tint]
    return out


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
    irregular: float = IRREGULAR_SHARE,
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
        inside = buildings[buildings["parcel_id"].astype(str) == str(parcel.id)] if built else None
        keep = None
        if inside is not None and len(inside):
            keep = inside.geometry.union_all().buffer(1.0)
        shape_rng = substream(seed, f"compound:shape:{parcel.id}")
        if float(shape_rng.uniform()) < irregular:
            inner = reshape(inner, keep, shape_rng)
        pts = np.asarray(inner.exterior.coords)
        road = road_lines.get(getattr(parcel, "road_id", None))
        edges = list(zip(pts[:-1], pts[1:], strict=True))
        gate = 0
        if road is not None:
            lengths = [float(np.hypot(*(b - a))) for a, b in edges]
            far = [d if n >= GATE_M / 0.6 else 1e9 for (a, b), n in zip(edges, lengths, strict=True)
                   for d in [road.distance(_mid(a, b))]]  # fmt: skip
            gate = int(np.argmin(far))
        segments: list[list[float]] = []
        for i, (a, b) in enumerate(edges):
            length = float(np.hypot(*(b - a)))
            if length < 1e-6:
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


def reshape(poly: Any, keep: Any, rng: Any) -> Any:
    """``poly`` with up to three cuts taken off (straight, or a curved bite) and maybe rounded
    corners, so its walls meet at odd angles and bow.

    Cuts only remove ground, so the result stays inside the lot. A cut that would touch ``keep``
    (the house, with a margin) or leave under ``MIN_KEPT_SHARE`` of the lot is skipped.
    """
    from shapely.geometry import Polygon

    count = int(rng.choice(len(CUT_COUNTS), p=CUT_COUNTS))
    shape, start, done = poly, poly.area, 0
    for _ in range(count * 4):  # a few tries per cut: many fall on the house or a short side
        if done == count:
            break
        pts = np.asarray(shape.exterior.coords)[:-1]
        n, i = len(pts), int(rng.integers(0, len(pts)))
        v, nxt, prev = pts[i], pts[(i + 1) % n], pts[i - 1]
        to_next, to_prev = nxt - v, prev - v
        ln, lp = float(np.hypot(*to_next)), float(np.hypot(*to_prev))
        if min(ln, lp) < 6.0:
            continue
        un, up = to_next / ln, to_prev / lp
        kind = int(rng.integers(0, 5))
        d1 = float(rng.uniform(2.0, min(12.0, 0.45 * lp)))
        d2 = float(rng.uniform(2.0, min(12.0, 0.45 * ln)))
        if kind == 0:  # a corner cut off at a slant
            cut = [v, v + up * d1, v + un * d2]
        elif kind == 1:  # a corner notched out
            cut = [v, v + up * d1, v + up * d1 + un * d2, v + un * d2]
        elif kind == 2:  # a whole side leaned over: a trapezoid
            cut = [v, nxt, v + up * min(d1 + 3.0, 0.45 * lp)]
        elif kind == 3 or kind == 4:  # a wedge, or a curved bite, pushed into the middle of a side
            t, w = float(rng.uniform(0.3, 0.7)), float(rng.uniform(3.0, min(9.0, 0.5 * ln)))
            mid = v + un * ln * t
            normal = np.array([-un[1], un[0]])
            if shape.contains(_pt(mid - normal * 0.3)):
                normal = -normal
            if not shape.contains(_pt(mid + normal * 0.3)):
                continue
            depth = float(rng.uniform(1.5, 4.0))
            if kind == 3:
                cut = [mid - un * w / 2, mid + normal * depth, mid + un * w / 2]
            else:  # a parabolic arc: the wall bows inwards
                cut = [
                    mid
                    + un * w * (k / ARC_STEPS - 0.5)
                    + normal * depth * (1 - (2 * k / ARC_STEPS - 1) ** 2)
                    for k in range(ARC_STEPS + 1)
                ]
        trial = shape.difference(Polygon(cut))
        if trial.geom_type == "MultiPolygon":
            trial = max(trial.geoms, key=lambda g: g.area)
        if trial.geom_type != "Polygon" or trial.area < MIN_KEPT_SHARE * start:
            continue
        if keep is not None and not trial.contains(keep):
            continue
        shape, done = trial.simplify(0.05), done + 1
    if done and float(rng.uniform()) < ROUND_SHARE:  # round the convex corners
        r = float(rng.uniform(2.0, 5.0))
        rounded = shape.buffer(-r).buffer(r, quad_segs=4).simplify(0.03)
        fits = keep is None or rounded.contains(keep)
        if rounded.geom_type == "Polygon" and rounded.area >= MIN_KEPT_SHARE * start and fits:
            shape = rounded
    return shape


def _pt(p: Array) -> Any:
    from shapely.geometry import Point

    return Point(float(p[0]), float(p[1]))


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
            base: float = 0.0,
        ) -> None:
            nonlocal offset
            box = [(x, y, base) for x, y in quad] + [(x, y, top) for x, y in quad]
            verts.append(np.asarray(box, dtype="float32"))
            faces.append(_BOX.astype("int32") + offset)
            mats.append(np.full(len(_BOX), material, dtype="int16"))
            owners.append(np.full(len(_BOX), index, dtype="int32"))
            offset += 8

        for x0, y0, x1, y1 in c["segments"]:
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 1e-3:
                continue
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            thick = SHEET_THICKNESS_M if c.get("kind") == "sheet" else THICKNESS_M
            nx, ny = -uy * thick / 2, ux * thick / 2
            add(
                [(x0 - nx, y0 - ny), (x1 - nx, y1 - ny), (x1 + nx, y1 + ny), (x0 + nx, y0 + ny)],
                c["height_m"],
            )
            if c.get("kind") in ("block", "piered"):  # a coping stone along the top
                ct = thick + 2 * CAP_OVERHANG_M
                cx, cy = -uy * ct / 2, ux * ct / 2
                add(
                    [
                        (x0 - cx, y0 - cy),
                        (x1 - cx, y1 - cy),
                        (x1 + cx, y1 + cy),
                        (x0 + cx, y0 + cy),
                    ],  # fmt: skip
                    c["height_m"] + CAP_M,
                    base=c["height_m"],
                )
            if c.get("kind") == "piered":  # a post at each end and every few metres between
                if length < 2.0:  # a short run of a curve: no post on every facet
                    continue
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
