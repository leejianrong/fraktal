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
from shapely.geometry import Point

from fraktal.ids import substream
from fraktal.labels3d import CLASSES
from fraktal.propgen import GENERATORS, MATERIALS
from fraktal.render import ROAD_ARTERIAL, ROAD_LOCAL
from fraktal.trees3d import wall_distance

Array = np.ndarray[Any, Any]

PROP_CLASS = {
    "container": "yard_object",
    "water_tank": "yard_object",
    "market_stall": "yard_object",
    "signboard": "yard_object",
    "tuk_tuk": "vehicle",
    "silo": "yard_object",
    "motorbike": "vehicle",
    "minibus": "vehicle",
    "dumpster": "yard_object",
    "barrel_stack": "yard_object",
    "rubbish_pile": "yard_object",
    "pedestrian": "person",
}
MIX = {
    "container": 0.12,
    "water_tank": 0.22,
    "market_stall": 0.28,
    "signboard": 0.18,
    "tuk_tuk": 0.20,
}
STREET_MIX = {  # the street pass (FRK-112): vehicles and people on the road, bins and litter by it
    "motorbike": 0.30,
    "minibus": 0.08,
    "pedestrian": 0.22,
    "dumpster": 0.08,
    "barrel_stack": 0.12,
    "rubbish_pile": 0.20,
}
STREET_DENSITY_RATIO = 1.5  # street props per km2, as a multiple of the scene prop density
ON_STREET = ("tuk_tuk", "motorbike", "minibus", "pedestrian")  # these stand on the road raster
HALF_EXTENT = {  # half length, half width in metres, for the clearance check
    "container": (3.1, 1.3), "water_tank": (0.8, 0.8), "market_stall": (1.5, 1.1),
    "signboard": (1.3, 0.4), "tuk_tuk": (1.5, 0.8), "silo": (2.6, 2.6),
    "motorbike": (1.05, 0.45), "minibus": (2.75, 1.0), "dumpster": (1.25, 0.8),
    "barrel_stack": (0.95, 0.65), "rubbish_pile": (1.7, 1.7), "pedestrian": (0.3, 0.35),
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
    walls: list[list[float]] | None = None,
    mix: dict[str, float] | None = None,
    stream: str = "props:sample",
    avoid: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Props as dicts (kind, variant, x, y, yaw, class). ``mask`` > 0 is a building footprint;
    ``walls`` (segments x0 y0 x1 y1 in tile metres) keep props from standing in a compound wall.
    ``mix`` and ``stream`` pick another kind of prop with its own random draws (the street pass);
    ``avoid`` are props already placed, which new ones keep their distance from."""
    mix = MIX if mix is None else mix
    size_m = side * gsd_m
    n = int(round(density_per_km2 * size_m * size_m / 1e6))
    if n <= 0:
        return []
    rng = substream(seed, stream)
    count = n * CANDIDATES_PER_PROP
    kinds = list(mix)
    kind_idx = rng.choice(
        len(kinds), size=count, p=np.asarray(list(mix.values())) / sum(mix.values())
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
            if kind in ON_STREET:
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
        if kind in ON_STREET and road is None:
            ok &= ~sel
    if walls:
        reach = np.array([max(HALF_EXTENT[k]) for k in kinds])[kind_idx] + 0.3
        ok &= wall_distance(xs, ys, np.asarray(walls, dtype="float64").reshape(-1, 4)) > reach
    vary = substream(seed, "props:vary" if stream == "props:sample" else f"{stream}:vary")
    scales = np.clip(np.exp(vary.normal(0.0, SCALE_SIGMA, count)), *SCALE_LIMITS)
    tints = vary.integers(0, len(TINT), count)
    wears = vary.choice(len(WEAR), size=count, p=WEAR_P)
    chosen: list[dict[str, Any]] = []
    placed = np.array([[p["x"], p["y"]] for p in avoid or []]).reshape(-1, 2)
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


def yard_props(sites: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Containers, tanks and silos in the yards of industrial sites.

    Each site is ``{"id", "yard": Polygon, "yaw"}``: the free apron in tile metres, clear of the
    halls, and the angle of the road it fronts. Containers stand along that angle. Props come
    from a stream per site, so one site never moves another's.
    """
    out: list[dict[str, Any]] = []
    vary = substream(seed, "props:yard:vary")
    for site in sorted(sites, key=lambda s: str(s["id"])):
        rng = substream(seed, f"props:yard:{site['id']}")
        yard = site["yard"]
        if yard.is_empty:
            continue
        x0, y0, x1, y1 = yard.bounds
        want = int(yard.area / 55.0)  # one object per ~55 m2 of apron
        placed: list[tuple[float, float, float]] = []
        for _ in range(want * 10):
            if len(placed) >= want:
                break
            kind = ("container", "container", "container", "water_tank", "silo")[
                int(rng.integers(0, 5))
            ]
            r = max(HALF_EXTENT[kind])
            x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
            if not yard.buffer(-r).contains(Point(x, y)):
                continue
            if any(math.hypot(x - px, y - py) < r + pr + 1.2 for px, py, pr in placed):
                continue
            placed.append((x, y, r))
            turn = math.pi / 2 if rng.uniform() < 0.3 else 0.0
            spin = float(rng.uniform(0, 6.28))
            yaw = float(site.get("yaw", 0.0)) + turn if kind == "container" else spin
            scale = float(np.clip(np.exp(vary.normal(0, SCALE_SIGMA)), *SCALE_LIMITS))
            out.append(
                {
                    "kind": kind,
                    "variant": int(rng.integers(0, VARIANTS)),
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "yaw": round(yaw, 4),
                    "scale": round(scale, 3),
                    "tint": int(vary.integers(0, len(TINT))),
                    "wear": int(vary.choice(len(WEAR), p=WEAR_P)),
                    "class": CLASSES[PROP_CLASS[kind]],
                }
            )
    return out


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
