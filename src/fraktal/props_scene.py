"""Props placed in a 3D scene and packed for Blender (FRK-124).

``sample`` picks where props go (open ground, clear of buildings and roads; tuk-tuks on streets) and
``pack`` turns them into flat arrays, the way buildings are, so Blender only reads data. A prop's
class comes from the catalogue below, never from the model that made the shape. Randomness only
through ``substream`` (ADR-0003).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraktal.ids import substream
from fraktal.labels3d import CLASSES
from fraktal.propgen import GENERATORS, MATERIALS
from fraktal.render import ROAD_ARTERIAL, ROAD_LOCAL

Array = np.ndarray[Any, Any]

PROP_CLASS = {
    "container": "yard_object",
    "water_tank": "yard_object",
    "market_stall": "yard_object",
    "signboard": "yard_object",
    "tuk_tuk": "vehicle",
}
MIX = {
    "container": 0.12,
    "water_tank": 0.22,
    "market_stall": 0.28,
    "signboard": 0.18,
    "tuk_tuk": 0.20,
}
HALF_EXTENT = {  # half length, half width in metres, for the clearance check
    "container": (3.1, 1.3), "water_tank": (0.8, 0.8), "market_stall": (1.5, 1.1),
    "signboard": (1.3, 0.4), "tuk_tuk": (1.5, 0.8),
}  # fmt: skip
VARIANTS = 8  # distinct shapes per kind, so a tile reuses geometry the way instancing would
OPEN_GROUND = (20, 30, 40, 60)  # shrub, grass, crop, bare
STREET = (ROAD_LOCAL, ROAD_ARTERIAL)  # the road raster's local and arterial codes
SCALE_SIGMA, SCALE_LIMITS = 0.04, (0.92, 1.08)  # a log-normal size jitter, clamped
TINT = (0.85, 0.95, 1.05, 1.15)  # brightness of every part colour, one step per prop
WEAR = ((1.0, 0.0), (0.93, 0.12), (0.84, 0.25))  # (colour multiplier, roughness added)
WEAR_P = (0.5, 0.3, 0.2)
MIN_SEPARATION_M = 4.0
CANDIDATES_PER_PROP = 12


def _lookup(raster: Array, gsd_m: float, side: int, x: Array, y: Array) -> Array:
    row = np.clip(side - 1 - (y / gsd_m).astype(int), 0, side - 1)
    col = np.clip((x / gsd_m).astype(int), 0, side - 1)
    return np.asarray(raster[row, col])


def sample(
    surface: Array,
    mask: Array,
    road: Array | None,
    gsd_m: float,
    side: int,
    density_per_km2: float,
    seed: int,
    street_codes: tuple[int, ...] = STREET,
) -> list[dict[str, Any]]:
    """Props as dicts (kind, variant, x, y, yaw, class). ``mask`` > 0 is a building footprint."""
    size_m = side * gsd_m
    n = int(round(density_per_km2 * size_m * size_m / 1e6))
    if n <= 0:
        return []
    rng = substream(seed, "props:sample")
    count = n * CANDIDATES_PER_PROP
    kinds = list(MIX)
    kind_idx = rng.choice(
        len(kinds), size=count, p=np.asarray(list(MIX.values())) / sum(MIX.values())
    )
    xs, ys = rng.uniform(0, size_m, count), rng.uniform(0, size_m, count)
    yaw = rng.uniform(0, 2 * math.pi, count)
    variant = rng.integers(0, VARIANTS, count)
    ok = np.ones(count, dtype=bool)
    for k, kind in enumerate(kinds):
        sel = kind_idx == k
        hl, hw = HALF_EXTENT[kind]
        clear = np.ones(count, dtype=bool)
        for da, db in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            px = xs + da * hl * np.cos(yaw) - db * hw * np.sin(yaw)
            py = ys + da * hl * np.sin(yaw) + db * hw * np.cos(yaw)
            inside = (px >= 0) & (px < size_m) & (py >= 0) & (py < size_m)
            free = _lookup(mask, gsd_m, side, px, py) == 0
            if kind == "tuk_tuk":
                where = (
                    np.ones(count, dtype=bool)
                    if road is None
                    else np.isin(_lookup(road, gsd_m, side, px, py), street_codes)
                )
            else:
                where = np.isin(_lookup(surface, gsd_m, side, px, py), OPEN_GROUND)
                if road is not None:
                    where &= _lookup(road, gsd_m, side, px, py) == 0
            clear &= inside & free & where
        ok &= ~sel | clear
        if kind == "tuk_tuk" and road is None:
            ok &= ~sel
    vary = substream(seed, "props:vary")  # its own stream, so placement draws are unchanged
    scales = np.clip(np.exp(vary.normal(0.0, SCALE_SIGMA, count)), *SCALE_LIMITS)
    tints = vary.integers(0, len(TINT), count)
    wears = vary.choice(len(WEAR), size=count, p=WEAR_P)
    chosen: list[dict[str, Any]] = []
    placed = np.empty((0, 2))
    for i in np.flatnonzero(ok):
        if len(chosen) >= n:
            break
        p = np.array([xs[i], ys[i]])
        if len(placed) and np.min(np.hypot(*(placed - p).T)) < MIN_SEPARATION_M:
            continue
        kind = kinds[int(kind_idx[i])]
        placed = np.vstack([placed, p])
        chosen.append(
            {
                "kind": kind,
                "variant": int(variant[i]),
                "x": round(float(xs[i]), 2),
                "y": round(float(ys[i]), 2),
                "yaw": round(float(yaw[i]), 4),
                "scale": round(float(scales[i]), 3),
                "tint": int(tints[i]),
                "wear": int(wears[i]),
                "class": CLASSES[PROP_CLASS[kind]],
            }
        )
    return chosen


def repeat_stats(props: list[dict[str, Any]]) -> dict[str, Any]:
    """How much a tile repeats itself. A rate is the share of props whose shape (kind and variant)
    or look (shape, tint and wear) was already used by an earlier prop in the tile."""
    looks = {(p["kind"], p["variant"], p.get("tint", 0), p.get("wear", 0)) for p in props}
    shapes = {(p["kind"], p["variant"]) for p in props}
    n = len(props)
    return {
        "count": n,
        "distinct_shapes": len(shapes),
        "distinct_looks": len(looks),
        "shape_repeat_rate": round(1.0 - len(shapes) / n, 3) if n else 0.0,
        "look_repeat_rate": round(1.0 - len(looks) / n, 3) if n else 0.0,
    }


def _look(name: str, tint: int, wear: int) -> tuple[str, list[float], float, float]:
    rgb, rough, metallic = MATERIALS[name]
    mult, extra = TINT[tint] * WEAR[wear][0], WEAR[wear][1]
    colour = [round(min(1.0, c * mult), 4) for c in rgb]
    return f"{name}~t{tint}w{wear}", colour, round(min(1.0, rough + extra), 3), metallic


def pack(props: list[dict[str, Any]]) -> dict[str, Any]:
    """Flat arrays for Blender: ``vertices`` (N, 3), triangle ``faces`` (M, 3), ``face_material``
    (an index into ``materials``), ``face_owner`` (the prop each face belongs to) and ``materials``
    (``[name, rgb, roughness, metallic]``)."""
    cache: dict[tuple[str, int], Any] = {}
    slots: dict[str, int] = {}  # looks in order of first use
    looks: list[list[Any]] = []
    verts_out: list[Array] = []
    faces_out: list[Array] = []
    mat_out: list[Array] = []
    owner_out: list[Array] = []
    offset = 0
    for index, prop in enumerate(props):
        key = (prop["kind"], prop["variant"])
        if key not in cache:
            cache[key] = GENERATORS[prop["kind"]](prop["variant"])
        c, s = math.cos(prop["yaw"]), math.sin(prop["yaw"])
        k = prop.get("scale", 1.0)
        for part in cache[key].parts:
            v = part.verts * k
            moved = np.column_stack(
                [
                    prop["x"] + c * v[:, 0] - s * v[:, 1],
                    prop["y"] + s * v[:, 0] + c * v[:, 1],
                    v[:, 2],
                ]
            )
            look = _look(part.material, prop.get("tint", 2), prop.get("wear", 0))
            if look[0] not in slots:
                slots[look[0]] = len(looks)
                looks.append(list(look))
            verts_out.append(moved.astype("float32"))
            faces_out.append(part.faces.astype("int32") + offset)
            mat_out.append(np.full(len(part.faces), slots[look[0]], dtype="int16"))
            owner_out.append(np.full(len(part.faces), index, dtype="int32"))
            offset += len(v)
    if not verts_out:
        empty = np.zeros((0, 3))
        return {
            "vertices": empty.astype("float32"), "faces": empty.astype("int32"),
            "face_material": np.zeros(0, "int16"), "face_owner": np.zeros(0, "int32"),
            "materials": [],
        }  # fmt: skip
    return {
        "vertices": np.vstack(verts_out),
        "faces": np.vstack(faces_out),
        "face_material": np.concatenate(mat_out),
        "face_owner": np.concatenate(owner_out),
        "materials": looks,
    }
