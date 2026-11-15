"""Road surface detail seen from the street: lane lines, crossings, kerbs, pavements (FRK-153).

The painted road is one flat colour. From eye level that is half of every frame, and the evaluation
segmenter reads it as pavement (Manila-Quezon 0.20 of the frame against 0.02 in the photos). This
module makes thin boxes along the road centre lines, in tile metres: white lane lines and zebra
stripes a centimetre or two above the road, kerbs and pavements 12 cm up. They are cut wherever a
junction, a building or a wall is in the way, and packed like the compound walls
(``compounds.pack``) for one mesh in Blender. A biome takes part by having an entry in ``STYLES``;
the others are unchanged.

Pure geometry from the road lines: no randomness, so the same world gives the same marks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from fraktal.ids import substream

Array = np.ndarray[Any, Any]

MATERIALS: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "line-white": ((0.80, 0.80, 0.77), 0.7, 0.0),
    "line-yellow": ((0.78, 0.62, 0.10), 0.7, 0.0),
    "kerb": ((0.58, 0.56, 0.53), 0.9, 0.0),
    "pavement": ((0.50, 0.46, 0.42), 0.95, 0.0),
    # Road wear (FRK-167), set against the asphalt grey of ground_soil.SEALED_LOCAL_RGB (0.33).
    "road-patch": ((0.21, 0.21, 0.22), 0.9, 0.0),  # a fresh dark patch over a trench
    "road-patch-pale": ((0.41, 0.40, 0.38), 0.95, 0.0),  # an old sun-bleached one
    "road-stain": ((0.24, 0.24, 0.24), 0.8, 0.0),  # oil, and the grime along a gutter
    "road-polish": ((0.35, 0.35, 0.35), 0.8, 0.0),  # tyre tracks worn smooth
    "road-crack": ((0.11, 0.11, 0.11), 0.95, 0.0),
    "manhole": ((0.19, 0.19, 0.19), 0.5, 0.0),
}
LINE_M = 0.15  # width of a painted line
PAINT_M = 0.015  # paint stands this high above the road
KERB_M, KERB_HEIGHT_M = 0.20, 0.12
DASH_M, PERIOD_M = 3.0, 8.0
ZEBRA_STRIPE_M, ZEBRA_PITCH_M, ZEBRA_DEPTH_M = 0.5, 1.0, 3.0
STEP_M = 1.5  # length of one kerb or pavement piece
MIN_PAVED_M = 6.0  # narrower roads, footpaths and tracks get no kerb or marks
LANE_LINES_FROM_M = 18.0  # an arterial this wide gets lane lines either side of its centre


@dataclass(frozen=True)
class Style:
    centre: str  # "dashed" or "none"
    zebra: bool
    kerb: bool
    pavement_m: float  # width of a raised pavement beyond the kerb, 0 for none


STYLES: dict[str, Style] = {
    "manila_quezon": Style("dashed", True, True, 0.0),  # photos: marked asphalt, little pavement
    "nairobi_centre": Style("dashed", True, True, 2.5),  # paver pavements, bollards, markings
    "khartoum_city": Style("none", False, True, 0.0),  # faint markings, bare edges
}

Part = tuple[list[tuple[float, float]], float, float, str]  # quad, base z, top z, material

WEAR_TOP_M = 0.004  # wear lies under the paint (PAINT_M) and is staggered so overlaps do not fight
TRACK_OFFSET_M, TRACK_WIDTH_M, TRACK_STEP_M = 0.85, 0.65, 3.0
PATCH_EVERY_M, MANHOLE_EVERY_M, CRACK_EVERY_M = 25.0, 70.0, 30.0
GUTTER_M = 0.4  # grime on the road beside a kerb


@dataclass(frozen=True)
class RoadLine:
    xy: Sequence[tuple[float, float]]  # tile metres
    width_m: float
    road_class: str


def _paved(r: RoadLine) -> bool:
    return r.road_class in ("arterial", "local") and r.width_m >= MIN_PAVED_M


def _frame(line: LineString, s: float) -> tuple[float, float, float, float]:
    """Point and unit tangent of ``line`` at arc length ``s``."""
    a, b = line.interpolate(max(s - 0.5, 0.0)), line.interpolate(min(s + 0.5, line.length))
    dx, dy = b.x - a.x, b.y - a.y
    n = math.hypot(dx, dy) or 1.0
    p = line.interpolate(s)
    return p.x, p.y, dx / n, dy / n


def _quad(
    line: LineString, s0: float, s1: float, off0: float, off1: float
) -> list[tuple[float, float]]:
    """The strip of ``line`` between arc lengths ``s0`` and ``s1`` and left offsets off0 to off1."""
    out = []
    for s, off in ((s0, off0), (s1, off0), (s1, off1), (s0, off1)):
        x, y, tx, ty = _frame(line, s)
        out.append((x - ty * off, y + tx * off))  # the left of the direction of travel
    return out


def _cut(quad: list[tuple[float, float]], zone: Any) -> bool:
    cx = sum(p[0] for p in quad) / 4.0
    cy = sum(p[1] for p in quad) / 4.0
    return bool(zone is not None and not zone.is_empty and zone.contains(Point(cx, cy)))


def build(
    roads: Sequence[RoadLine], hazards: Any, biome: str, seed: int | None = None
) -> list[Part]:
    """Marks and raised edges for ``roads`` in a biome with a ``STYLES`` entry, else none.

    ``hazards`` is a shapely geometry (building footprints, walls) that raised edges keep out of.
    With a ``seed`` the road also gets wear (patches, cracks, stains, manholes, tyre tracks),
    drawn from the world seed so the same world wears the same way.
    """
    style = STYLES.get(biome)
    if style is None:
        return []
    paved = [r for r in roads if _paved(r)]
    lines = [LineString(r.xy) for r in paved]
    carriage = [
        ln.buffer(r.width_m / 2.0, cap_style="flat") for ln, r in zip(lines, paved, strict=True)
    ]
    junctions = []  # where two carriageways overlap, and a little beyond
    for i in range(len(paved)):
        for j in range(i + 1, len(paved)):
            overlap = carriage[i].intersection(carriage[j])
            if not overlap.is_empty:
                junctions.append(overlap.buffer(2.0))
    junction_zone = unary_union(junctions) if junctions else None
    if junction_zone is not None:
        shapely.prepare(junction_zone)
    parts: list[Part] = []
    for i, (ln, r) in enumerate(zip(lines, paved, strict=True)):
        others = (
            unary_union([c for k, c in enumerate(carriage) if k != i])
            if len(carriage) > 1
            else None
        )
        keep_out = unary_union([g for g in (others, hazards) if g is not None and not g.is_empty])
        shapely.prepare(keep_out)
        half = r.width_m / 2.0
        if style.centre == "dashed":
            offsets, material = [0.0], "line-white"
            if r.road_class == "arterial" and r.width_m >= LANE_LINES_FROM_M:
                offsets = [-r.width_m / 4.0, r.width_m / 4.0]
            for off in offsets:
                s = 0.0
                while s + DASH_M <= ln.length:
                    q = _quad(ln, s, s + DASH_M, off - LINE_M / 2, off + LINE_M / 2)
                    if not _cut(q, junction_zone):
                        parts.append((q, 0.0, PAINT_M, material))
                    s += PERIOD_M
            if r.road_class == "arterial" and r.width_m >= LANE_LINES_FROM_M:  # a solid centre
                s = 0.0
                while s + STEP_M <= ln.length:
                    q = _quad(ln, s, s + STEP_M, -LINE_M, LINE_M)
                    if not _cut(q, junction_zone):
                        parts.append((q, 0.0, PAINT_M, "line-yellow"))
                    s += STEP_M
        for side in (-1.0, 1.0):
            reach = [(half, half + KERB_M, "kerb")] if style.kerb else []
            if style.pavement_m > 0:
                reach.append((half + KERB_M, half + KERB_M + style.pavement_m, "pavement"))
            for a, b, material in reach:
                s = 0.0
                while s + STEP_M <= ln.length:
                    q = _quad(ln, s, s + STEP_M, side * a, side * b)
                    if not _cut(q, keep_out):
                        parts.append((q, 0.0, KERB_HEIGHT_M, material))
                    s += STEP_M
        if seed is not None:
            parts += _wear(ln, r, style, keep_out, junction_zone, seed, i)
    if style.zebra:
        parts += _zebras(paved, lines, carriage)
    return parts


def _rect(
    x: float, y: float, half_along: float, half_across: float, angle: float
) -> list[tuple[float, float]]:
    """A rectangle centred on (x, y), its long side along ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [
        (x + c * a - s * b, y + s * a + c * b)
        for a, b in (
            (-half_along, -half_across),
            (half_along, -half_across),
            (half_along, half_across),
            (-half_along, half_across),
        )
    ]


def _wear(
    ln: LineString,
    road: RoadLine,
    style: Style,
    keep_out: Any,
    junction_zone: Any,
    seed: int,
    index: int,
) -> list[Part]:
    """Wear on one carriageway: tyre tracks and oil, patches, manholes, cracks and gutter grime."""
    rng = substream(seed, f"road-wear:{index}")
    half, length = road.width_m / 2.0, ln.length
    parts: list[Part] = []
    count = 0

    def add(quad: list[tuple[float, float]], material: str, zone: Any = keep_out) -> None:
        nonlocal count
        if _cut(quad, zone) or _cut(quad, junction_zone):
            return
        parts.append((quad, 0.0, WEAR_TOP_M + 0.0005 * (count % 6), material))
        count += 1

    for lane in (-half / 2.0, half / 2.0):  # two lanes, tracks under the wheels, oil between
        s = 0.0
        while s + TRACK_STEP_M <= length:
            for side in (-1.0, 1.0):
                if rng.uniform() < 0.9:
                    c = lane + side * TRACK_OFFSET_M
                    track = _quad(
                        ln, s, s + TRACK_STEP_M, c - TRACK_WIDTH_M / 2, c + TRACK_WIDTH_M / 2
                    )
                    add(track, "road-polish")
            if rng.uniform() < 0.4:
                add(_quad(ln, s, s + TRACK_STEP_M, lane - 0.15, lane + 0.15), "road-stain")
            s += TRACK_STEP_M
    if style.kerb:
        for side in (-1.0, 1.0):
            s = 0.0
            while s + STEP_M <= length:
                if rng.uniform() < 0.8:
                    add(
                        _quad(ln, s, s + STEP_M, side * (half - GUTTER_M), side * half),
                        "road-stain",
                    )
                s += STEP_M
    for _ in range(int(rng.poisson(length / PATCH_EVERY_M))):
        x, y, tx, ty = _frame(ln, float(rng.uniform(0, length)))
        off = float(rng.uniform(-half + 1.4, half - 1.4))  # the whole patch stays on the road
        cx, cy = x - ty * off, y + tx * off
        pale = rng.uniform() < 0.4
        add(_rect(cx, cy, float(rng.uniform(0.8, 2.5)), float(rng.uniform(0.4, 1.3)),
                  math.atan2(ty, tx)), "road-patch-pale" if pale else "road-patch")  # fmt: skip
    for _ in range(int(rng.poisson(length / MANHOLE_EVERY_M))):
        x, y, tx, ty = _frame(ln, float(rng.uniform(0, length)))
        off = float(rng.choice([-half / 2.0, half / 2.0]) + rng.uniform(-0.5, 0.5))
        cx, cy = x - ty * off, y + tx * off
        for turn in (0.0, math.pi / 4):  # two squares make an octagon
            add(_rect(cx, cy, 0.35, 0.35, math.atan2(ty, tx) + turn), "manhole")
    for _ in range(int(rng.poisson(length / CRACK_EVERY_M))):
        x, y, tx, ty = _frame(ln, float(rng.uniform(0, length)))
        off = float(rng.uniform(-half + 0.5, half - 0.5))
        px, py = x - ty * off, y + tx * off
        heading = float(rng.uniform(0, 2 * math.pi))
        for _step in range(int(rng.integers(6, 13))):
            heading += float(rng.normal(0.0, 0.5))
            seg = float(rng.uniform(0.4, 0.9))
            nx, ny = px + seg * math.cos(heading), py + seg * math.sin(heading)
            if Point(nx, ny).distance(ln) > half - 0.2:  # a crack ends at the road edge
                break
            mx, my = (px + nx) / 2.0, (py + ny) / 2.0
            add(_rect(mx, my, seg / 2.0 + 0.02, 0.02, heading), "road-crack")
            px, py = nx, ny
    return parts


def _zebras(
    paved: Sequence[RoadLine], lines: Sequence[LineString], carriage: Sequence[Any]
) -> list[Part]:
    """A crossing on every approach to a junction that has an arterial in it."""
    parts: list[Part] = []
    for i in range(len(paved)):
        for j in range(i + 1, len(paved)):
            if "arterial" not in (paved[i].road_class, paved[j].road_class):
                continue
            meet = lines[i].intersection(lines[j])
            if meet.is_empty or meet.geom_type != "Point":
                continue
            for a, b in ((i, j), (j, i)):  # a crossing on road a, beside the junction with b
                s_mid = lines[a].project(meet)
                back = paved[b].width_m / 2.0 + 2.0 + ZEBRA_DEPTH_M / 2.0
                for s in (s_mid - back, s_mid + back):
                    if s - ZEBRA_DEPTH_M / 2 < 0 or s + ZEBRA_DEPTH_M / 2 > lines[a].length:
                        continue
                    span = paved[a].width_m / 2.0 - 0.8
                    off = -span
                    while off + ZEBRA_STRIPE_M <= span:
                        q = _quad(
                            lines[a],
                            s - ZEBRA_DEPTH_M / 2,
                            s + ZEBRA_DEPTH_M / 2,
                            off,
                            off + ZEBRA_STRIPE_M,
                        )
                        parts.append((q, 0.0, PAINT_M, "line-white"))
                        off += ZEBRA_PITCH_M
    return parts


_BOX = np.array(
    [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
     [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]]
)  # fmt: skip


def pack(parts: Sequence[Part]) -> dict[str, Any]:
    """Arrays for Blender, laid out like ``compounds.pack``: a box and a material per part."""
    names = list(MATERIALS)
    materials = [[n, list(MATERIALS[n][0]), MATERIALS[n][1], MATERIALS[n][2]] for n in names]
    if not parts:
        return {
            "vertices": np.zeros((0, 3), "float32"), "faces": np.zeros((0, 3), "int32"),
            "face_material": np.zeros(0, "int16"), "face_owner": np.zeros(0, "int32"),
            "materials": materials,
        }  # fmt: skip
    verts, faces, mats = [], [], []
    for k, (quad, base, top, material) in enumerate(parts):
        box = [(x, y, base) for x, y in quad] + [(x, y, top) for x, y in quad]
        verts.append(np.asarray(box, dtype="float32"))
        faces.append(_BOX.astype("int32") + 8 * k)
        mats.append(np.full(len(_BOX), names.index(material), dtype="int16"))
    return {
        "vertices": np.vstack(verts), "faces": np.vstack(faces),
        "face_material": np.concatenate(mats),
        "face_owner": np.zeros(len(_BOX) * len(parts), dtype="int32"),
        "materials": materials,
    }  # fmt: skip


__all__ = ["MATERIALS", "STYLES", "RoadLine", "Style", "build", "pack"]
