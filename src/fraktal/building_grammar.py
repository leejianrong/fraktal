"""A building grammar for the 3D path: one typology, the flat-roof block (ADR-0010, FRK-49).

``build`` turns a footprint ring, a height and a roof material into a polygon mesh: walls, a flat
roof with a parapet, window rows per storey, a street door, and roof clutter (a water tank, a
stair box). Every choice is drawn from ``substream(seed, "building:<id>")``, so a building looks
the same wherever and whenever it is built. The mesh is plain arrays (the scene is data: the Blender
side only reads them), and each face carries the name of a material from ``assets/materials``.

Three typologies (FRK-63): the flat-roof ``block``, the ``shed`` (pitched tin or tile roof, for a
footprint with a recorded roof material) and the ``courtyard`` house (a ring of rooms round a yard,
for some low rectangles). A footprint that cannot be built (fewer than three points, no area)
raises ``ValueError``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import split

from fraktal.ids import substream

WALLS = ("wall-plaster-pale", "wall-plaster-sand", "wall-plaster-ochre", "wall-concrete")
GENERATED_WALLS = ("wall-paint-peeling", "wall-plaster-cracked", "wall-concrete-cracked")
GENERATED_WALL_SHARE = 0.25  # of buildings; SDXL textures (FRK-76), an assumed share
ROOFS = {"tin": "roof-tin", "tile": "roof-tile"}
CLAY_TILE = "roof-tile-clay"
CLAY_TILE_SHARE = 0.5  # of tile roofs, the rest use the CC0 tile map
TIN_ROOFS = (
    "roof-tin",
    "roof-tin-red",
    "roof-tin-rust",
    "roof-tin-dark",
    "roof-tin-blue",
    "roof-tin-green",
    "roof-tin-white",
    "roof-tin-pink",
)  # silver, red, rusty, dark, then the painted colours seen on the reference board (FRK-110)
TIN_WEIGHTS = (
    0.15,
    0.35,
    0.30,
    0.20,
    0.0,
    0.0,
    0.0,
    0.0,
)  # the original mix, for a caller with no biome
# Per-biome mixes in the order of TIN_ROOFS, set by eye from the board: a wide spread of rust, red,
# blue, green, white and a little pink. Khartoum is drier and plainer.
TIN_MIXES: dict[str, tuple[float, ...]] = {
    "khartoum": (0.30, 0.08, 0.38, 0.10, 0.05, 0.0, 0.09, 0.0),
}
SOLAR_SHARE, BRICK_SHARE, EXTRA_TANK_SHARE = 0.20, 0.15, 0.20  # of flat roofs, assumed shares
TIN_MIX_TROPICAL = (0.12, 0.20, 0.22, 0.08, 0.16, 0.08, 0.10, 0.04)
ROOF_CLUTTER = ("solar-panel", "brick-stack", "tank-black", "tank-blue")
ROOF_GEAR = ("metal-grey",)  # air-conditioner boxes, dishes and vent stacks (code-written shapes)
AC_SHARE, DISH_SHARE, VENT_SHARE, STAIR_HUT_SHARE = 0.35, 0.2, 0.25, 0.5
FLAT_ROOFS = ("roof-flat-concrete", "roof-flat-pale", "roof-flat-terracotta", "roof-flat-bitumen")
OPENINGS = ("window-glass", "door")
CLUTTER = ("tank",)
FASCIA = ("fascia-red", "fascia-blue", "fascia-green", "fascia-yellow")  # shop signs and awnings
SHUTTERS = ("shutter-green", "shutter-brown")  # window shutters, painted wood
DETAIL = ("wall-plinth", "window-frame", "balcony", "roller-door", *FASCIA, *SHUTTERS)
MATERIALS = (
    *WALLS,
    *GENERATED_WALLS,
    CLAY_TILE,
    *FLAT_ROOFS,
    *ROOFS.values(),
    *OPENINGS,
    *CLUTTER,
    *DETAIL,
    *TIN_ROOFS[1:],
    *ROOF_CLUTTER,
    *ROOF_GEAR,
)
STOREY_M = 3.0
COURTYARD_MIN_AREA_M2 = 140.0
COURTYARD_MAX_HEIGHT_M = 7.5
COURTYARD_SHARE = 0.35
HALL_MIN_AREA_M2 = 400.0  # a long, large flat-footed building is an industrial hall
HALL_MIN_ASPECT = 1.8
PARAPET_M = (0.35, 0.7)
PARAPET_THICKNESS_M = 0.18
WINDOW_M = (1.1, 1.3)  # width, height
SILL_M = 0.9
MIN_EDGE_FOR_WINDOWS_M = 3.0
PLINTH_M = 0.9  # a darker band along the foot of a wall
FRAME_M = 0.08  # the window frame shows this much around the glass
SHOP_SHARE = 0.2  # of flat-roofed blocks with a long enough street wall
SHOP_MIN_EDGE_M, SHOP_MIN_STOREY_M = 7.0, 3.3
SHOP_WIDTH_M = (4.5, 7.5)
SHOP_SHUTTER_M, SHOP_DOOR_M = (2.4, 2.45), 1.0  # shutter width, height; the side door's width
FASCIA_Z_M = (2.7, 3.25)  # the sign band, above the shutter
AWNING_SHARE, AWNING_OUT_M, AWNING_Z_M = 0.5, 1.1, (2.6, 2.25)  # at the wall, at the outer edge
SHUTTER_SHARE = 0.22  # of blocks whose windows have painted shutters either side
SHUTTER_W_M = 0.32
ROLLER_DOOR_M = (4.0, 4.4)  # width, height of a hall's roller door
ROLLER_SPACING_M = 11.0  # one door per this much of a long wall
BALCONY_SHARE = 0.35  # of buildings with three storeys or more
BALCONY_BAY_SHARE = 0.3  # of the upper-floor bays of such a building
POP_OUT_M = 0.03  # windows and doors sit this far proud of the wall so they are not coplanar


@dataclass
class Mesh:
    """A polygon soup: ``faces`` index ``vertices``; ``materials`` names each face's material."""

    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, ...]] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    typology: str = "block"

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


def _balcony(mesh: Mesh, centre: Any, direction: Any, outward: Any, z: float) -> None:
    """A slab 1.8 m wide and 0.9 m deep at floor level, with a solid rail round three sides."""
    half, depth = 0.9, 0.9
    p0, p1 = centre - direction * half, centre + direction * half
    q0, q1 = p0 + outward * depth, p1 + outward * depth
    top, bottom, rail = z + 0.1, z - 0.1, z + 1.0
    mesh.add([(q0[0], q0[1], top), (q1[0], q1[1], top), (p1[0], p1[1], top), (p0[0], p0[1], top)],
             "balcony")  # fmt: skip
    mesh.add([(p0[0], p0[1], bottom), (p1[0], p1[1], bottom), (q1[0], q1[1], bottom),
              (q0[0], q0[1], bottom)], "balcony")  # fmt: skip
    mesh.add(_quad(q0, q1, bottom, top), "balcony")
    for a, b in ((p0, q0), (q0, q1), (q1, p1)):  # the rail: front and both sides
        mesh.add(_quad(a, b, top, rail), "balcony")


def _roller_doors(mesh: Mesh, pts: Any, height: float) -> dict[int, list[tuple[float, float]]]:
    """Roller doors along each long wall of a hall; returns the stretch each one covers per edge."""
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    taken: dict[int, list[tuple[float, float]]] = {}
    width, tall = ROLLER_DOOR_M[0], min(ROLLER_DOOR_M[1], height - 1.0)
    for i, length in enumerate(lengths):
        if length < 0.6 * float(lengths.max()) or tall < 2.5:
            continue
        count = max(1, int(length // ROLLER_SPACING_M))
        direction = edges[i] / length
        outward = np.array([direction[1], -direction[0]])
        for c in range(count):
            along = (c + 0.5) * length / count
            a = pts[i] + direction * (along - width / 2) + outward * POP_OUT_M
            b = pts[i] + direction * (along + width / 2) + outward * POP_OUT_M
            mesh.add(_quad(a, b, 0.0, tall), "roller-door")
            taken.setdefault(i, []).append((along - width / 2 - 0.5, along + width / 2 + 0.5))
    return taken


def plan_shop(
    pts: Any, seed: int, building_id: str, storey: float
) -> tuple[int, float, float, str, bool] | None:
    """A shopfront on the longest wall: ``(edge, from, to, fascia material, awning)``, or None.

    Its own random stream (``shop:<id>``), so the draws behind every other part of a building
    stay where they were. The span is centred on the wall and kept off its corners.
    """
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    edge = int(np.argmax(lengths))
    length = float(lengths[edge])
    if length < SHOP_MIN_EDGE_M or storey < SHOP_MIN_STOREY_M:
        return None
    rng = substream(seed, f"shop:{building_id}")
    if float(rng.uniform()) >= SHOP_SHARE:
        return None
    width = min(float(rng.uniform(*SHOP_WIDTH_M)), length - 2.0)
    lo = (length - width) / 2.0
    fascia = FASCIA[int(rng.integers(0, len(FASCIA)))]
    return edge, lo, lo + width, fascia, float(rng.uniform()) < AWNING_SHARE


def _shopfront(mesh: Mesh, pts: Any, plan: tuple[int, float, float, str, bool]) -> None:
    """A roller shutter, a side door, a fascia sign and (half the time) an awning on one wall."""
    edge, lo, hi, fascia, awning = plan
    a = pts[edge]
    direction = (pts[(edge + 1) % len(pts)] - a) / max(
        float(np.hypot(*(pts[(edge + 1) % len(pts)] - a))), 1e-9
    )
    outward = np.array([direction[1], -direction[0]])

    def at(s: float, out: float = POP_OUT_M) -> Any:
        return a + direction * s + outward * out

    shutter_w = min(SHOP_SHUTTER_M[0], hi - lo - SHOP_DOOR_M - 0.6)
    mesh.add(_quad(at(lo + 0.3), at(lo + 0.3 + shutter_w), 0.0, SHOP_SHUTTER_M[1]), "roller-door")
    mesh.add(_quad(at(hi - 0.3 - SHOP_DOOR_M), at(hi - 0.3), 0.0, 2.1), "door")
    z0, z1 = FASCIA_Z_M
    mesh.add(_quad(at(lo, 0.12), at(hi, 0.12), z0, z1), fascia)
    if awning:
        near, far = at(lo, 0.1), at(lo, 0.1 + AWNING_OUT_M)
        near2, far2 = at(hi, 0.1), at(hi, 0.1 + AWNING_OUT_M)
        za, zb = AWNING_Z_M
        mesh.add([(*near, za), (*near2, za), (*far2, zb), (*far, zb)], fascia)
        mesh.add([(*far, zb), (*far2, zb), (*near2, za), (*near, za)], fascia)  # seen from below


def _shutter_pair(mesh: Mesh, a: Any, b: Any, direction: Any, outward: Any, z0: float,
                  material: str) -> None:  # fmt: skip
    """Two painted shutter leaves beside a window whose glass runs from ``a`` to ``b``."""
    top = z0 + WINDOW_M[1] + 0.06
    gap = direction * 0.04
    leaf = direction * SHUTTER_W_M
    for p, q in ((a - gap - leaf, a - gap), (b + gap, b + gap + leaf)):
        mesh.add(_quad(p + outward * 0.02, q + outward * 0.02, z0 - 0.03, top), material)


def _openings(
    mesh: Mesh,
    pts: Any,
    rng: Any,
    floors: int,
    storey: float,
    door: bool,
    balconies: Any = None,
    blocked: dict[int, list[tuple[float, float]]] | None = None,
    ground_only: bool = False,
    shutter: str | None = None,
) -> None:
    """A window grid on every edge long enough and, if ``door``, a door on the longest edge.

    ``blocked`` spans along an edge are taken by something else (a hall's roller door, or with
    ``ground_only`` a shopfront, which leaves the upper floors' windows); ``shutter`` names the
    material of painted shutters drawn either side of every window.

    ``pts`` is a ring whose right-hand side is outside: a counter-clockwise outline, or a
    clockwise courtyard so the windows face the open space.
    """
    n = len(pts)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    spacing = float(rng.uniform(2.6, 3.4))
    fill = float(rng.uniform(0.75, 1.0))
    door_edge = int(np.argmax(lengths)) if door else -1
    for i in range(n):
        length = float(lengths[i])
        if length < MIN_EDGE_FOR_WINDOWS_M:
            continue
        direction = edges[i] / length
        outward = np.array([direction[1], -direction[0]])  # CCW ring: outward is to the right
        cols = max(1, int((length - 1.0) // spacing))
        for c in range(cols):
            along = (c + 0.5) * length / cols
            spans = blocked.get(i, ()) if blocked is not None else ()
            covered = any(lo <= along <= hi for lo, hi in spans)
            if covered and not ground_only:
                continue  # a roller door takes this bay
            if i == door_edge and c == cols // 2 and not covered:  # the door takes this bay's floor
                half = 0.5
                a = pts[i] + direction * (along - half) + outward * POP_OUT_M
                b = pts[i] + direction * (along + half) + outward * POP_OUT_M
                mesh.add(_quad(a, b, 0.0, 2.1), "door")
            for f in range(floors):
                if f == 0 and (covered or (i == door_edge and c == cols // 2)):
                    continue
                if rng.uniform() > fill:
                    continue
                z0 = f * storey + SILL_M
                if z0 + WINDOW_M[1] > (f + 1) * storey:
                    continue
                half = WINDOW_M[0] / 2
                a = pts[i] + direction * (along - half) + outward * POP_OUT_M
                b = pts[i] + direction * (along + half) + outward * POP_OUT_M
                rim = direction * FRAME_M
                mesh.add(  # the frame, a little larger than the glass and just behind it
                    _quad(a - rim - outward * 0.01, b + rim - outward * 0.01,
                          z0 - FRAME_M, z0 + WINDOW_M[1] + FRAME_M), "window-frame")  # fmt: skip
                a2, b2 = a + outward * 0.04, b + outward * 0.04
                mesh.add(_quad(a2, b2, z0, z0 + WINDOW_M[1]), "window-glass")
                sill_a, sill_b = (
                    a - direction * 0.1 + outward * 0.1,
                    b + direction * 0.1 + outward * 0.1,
                )
                mesh.add(_quad(sill_a, sill_b, z0 - 0.1, z0 - 0.02), "window-frame")
                if shutter is not None:
                    _shutter_pair(mesh, a, b, direction, outward, z0, shutter)
                if balconies is not None and f >= 1 and balconies.uniform() < BALCONY_BAY_SHARE:
                    _balcony(mesh, pts[i] + direction * along, direction, outward, f * storey)


def _gable(
    poly: Polygon, pts: Any, height: float, wall: str, roof: str, rng: Any, floors: int,
    storey: float, rollers: bool = False,
) -> Mesh | None:  # fmt: skip
    """A pitched roof along the footprint's long axis, walls rising to the ridge at the gable ends.

    Returns None when the ridge does not cut the footprint in two, so the caller keeps the block.
    """
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords[:4], dtype="float64")
    sides = np.roll(rect, -1, axis=0) - rect
    long_side = sides[int(np.argmax(np.hypot(sides[:, 0], sides[:, 1])))]
    u = long_side / max(float(np.hypot(*long_side)), 1e-9)
    perp = np.array([-u[1], u[0]])
    centre = np.array([poly.centroid.x, poly.centroid.y])
    dist = (pts - centre) @ perp  # signed distance from the ridge line
    half_width = float(np.abs(dist).max())
    if half_width < 1.0:
        return None
    rise = float(np.clip(half_width * 0.22, 0.5, 1.5))  # a shallow pitch, about 12 degrees

    def lift(d: float) -> float:
        return height + rise * (1.0 - min(abs(d), half_width) / half_width)

    reach = float(np.hypot(*np.ptp(pts, axis=0))) + 1.0
    ridge = LineString([tuple(centre - u * reach), tuple(centre + u * reach)])
    pieces = [g for g in split(poly, ridge).geoms if isinstance(g, Polygon) and g.area > 1e-6]
    if len(pieces) != 2:
        return None
    mesh = Mesh(typology="shed")
    n = len(pts)
    for i in range(n):  # a wall follows the roof line, with a point where the ridge crosses it
        j = (i + 1) % n
        top = [(pts[j][0], pts[j][1], lift(dist[j]))]
        if dist[i] * dist[j] < 0:
            t = dist[i] / (dist[i] - dist[j])
            x, y = pts[i] + (pts[j] - pts[i]) * t
            top.append((float(x), float(y), height + rise))
        top.append((pts[i][0], pts[i][1], lift(dist[i])))
        mesh.add([(pts[i][0], pts[i][1], 0.0), (pts[j][0], pts[j][1], 0.0), *top], wall)
    for piece in pieces:
        face = np.asarray(orient(piece, 1.0).exterior.coords[:-1], dtype="float64")
        d = (face - centre) @ perp
        mesh.add([(x, y, lift(float(dd))) for (x, y), dd in zip(face, d, strict=True)], roof)
    blocked = _roller_doors(mesh, pts, height) if rollers else None
    _openings(mesh, pts, rng, floors, storey, door=not rollers, blocked=blocked)
    return mesh


def _courtyard(
    poly: Polygon, pts: Any, height: float, wall: str, roof: str, rng: Any, floors: int,
    storey: float,
) -> Mesh | None:  # fmt: skip
    """A ring of rooms round an open yard: outer and inner walls, a flat roof ring, openings."""
    side = float(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T).min())
    yard = _inset(pts, poly, float(np.clip(0.25 * side, 3.0, 5.5)))
    if yard is None:
        return None
    mesh = Mesh(typology="courtyard")
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        mesh.add(_quad(pts[i], pts[j], 0.0, height), wall)
        mesh.add(_quad(yard[j], yard[i], 0.0, height), wall)  # faces the yard
        ring = [pts[i], pts[j], yard[j], yard[i]]
        mesh.add([(float(x), float(y), height) for x, y in ring], roof)
    _openings(mesh, pts, rng, floors, storey, door=True)
    _openings(mesh, yard[::-1], rng, floors, storey, door=False)
    return mesh


def tin_mix(biome: str) -> tuple[float, ...]:
    """The tin roof colour weights, in the order of ``TIN_ROOFS``, for a biome."""
    return TIN_MIXES.get(biome, TIN_MIX_TROPICAL)


def typology_for(
    building_id: str, poly: Polygon, vertices: int, height: float, roof_material: str | None,
    seed: int,
) -> str:  # fmt: skip
    """``shed`` for a recorded tin or tile roof, a ``courtyard`` for some low roomy rectangles."""
    if roof_material in ROOFS:
        return "shed"
    if vertices == 4 and poly.area >= HALL_MIN_AREA_M2:
        rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords[:4], dtype="float64")
        a, b = np.hypot(*(rect[1] - rect[0])), np.hypot(*(rect[2] - rect[1]))
        if max(a, b) / max(min(a, b), 1e-9) >= HALL_MIN_ASPECT:
            return "industrial_hall"
    roll = float(substream(seed, f"typology:{building_id}").uniform())
    sides = np.hypot(*(np.diff(np.asarray(poly.exterior.coords), axis=0)).T)
    roomy = vertices == 4 and poly.area >= COURTYARD_MIN_AREA_M2 and sides.min() >= 11.0
    if roomy and height <= COURTYARD_MAX_HEIGHT_M and roll < COURTYARD_SHARE:
        return "courtyard"
    return "block"


def build(
    building_id: str,
    ring: Sequence[Sequence[float]],
    height_m: float,
    roof_material: str | None,
    seed: int,
    roof_mix: Sequence[float] | None = None,
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
    if (
        roof_material == "tin"
    ):  # real tin roofs are silver, painted, rusty or dark, not all one grey
        pick = float(substream(seed, f"roofcolour:{building_id}").uniform())
        weights = np.asarray(roof_mix if roof_mix is not None else TIN_WEIGHTS, dtype="float64")
        roof = TIN_ROOFS[
            int(np.searchsorted(np.cumsum(weights / weights.sum()), pick, side="right"))
        ]
    look = substream(seed, f"surface:{building_id}")  # a separate draw: the ones above keep theirs
    if float(look.uniform()) < GENERATED_WALL_SHARE:
        wall = GENERATED_WALLS[int(look.integers(0, len(GENERATED_WALLS)))]
    if roof_material == "tile" and float(look.uniform()) < CLAY_TILE_SHARE:
        roof = CLAY_TILE
    height = max(2.5, float(height_m))
    floors = max(1, round(height / STOREY_M))
    storey = height / floors
    typology = typology_for(building_id, poly, n, height, roof_material, seed)
    if typology == "shed":
        shed = _gable(poly, pts, height, wall, roof, rng, floors, storey)
        if shed is not None:
            return shed
    elif typology == "industrial_hall":
        tin = TIN_ROOFS[int(look.integers(0, 3))]
        hall = _gable(poly, pts, height, "wall-concrete", tin, rng, 1, height, rollers=True)
        if hall is not None:
            hall.typology = "industrial_hall"
            return hall
    elif typology == "courtyard":
        yard = _courtyard(poly, pts, height, wall, roof, rng, floors, storey)
        if yard is not None:
            return yard
    mesh = Mesh()

    for i in range(n):  # walls, with a darker plinth along the foot
        if height > PLINTH_M + 2.0:
            mesh.add(_quad(pts[i], pts[(i + 1) % n], 0.0, PLINTH_M), "wall-plinth")
            mesh.add(_quad(pts[i], pts[(i + 1) % n], PLINTH_M, height), wall)
        else:
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

    facade = substream(seed, f"facade:{building_id}")  # its own stream: the openings stay put
    balconies = facade if floors >= 3 and facade.uniform() < BALCONY_SHARE else None
    shop = plan_shop(pts, seed, building_id, storey) if flat else None
    painted = None
    if flat and float(substream(seed, f"shutters:{building_id}").uniform()) < SHUTTER_SHARE:
        painted = SHUTTERS[int(substream(seed, f"shutters:{building_id}:colour").integers(0, 2))]
    blocked = {shop[0]: [(shop[1] - 0.4, shop[2] + 0.4)]} if shop else None
    _openings(mesh, pts, rng, floors, storey, door=True, balconies=balconies, blocked=blocked,
              ground_only=True, shutter=painted)  # fmt: skip
    if shop:
        _shopfront(mesh, pts, shop)

    if flat and poly.area >= 60.0:  # roof clutter, kept clear of the parapet
        room = poly.buffer(-1.8)
        minx, miny, maxx, maxy = poly.bounds
        placed = 0
        spots: list[tuple[float, float]] = []
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
            spots.append((x, y))
        _more_clutter(poly, room, spots, top, mesh, substream(seed, f"clutter:{building_id}"))
        _rooftop_gear(poly, room, spots, top, mesh, substream(seed, f"gear:{building_id}"),
                      wall, floors)  # fmt: skip
    return mesh


def _more_clutter(
    poly: Polygon, room: Any, spots: list[tuple[float, float]], top: float, mesh: Mesh, rng: Any
) -> None:
    """Solar panels, stacked bricks and coloured tanks on a share of flat roofs (FRK-110).

    Its own random stream, so the buildings drawn before this keep their looks. Items keep 3 m
    from the first clutter and from each other.
    """
    if room.is_empty:
        return
    minx, miny, maxx, maxy = poly.bounds
    wanted: list[str] = []
    if float(rng.uniform()) < SOLAR_SHARE:
        wanted.append("solar-panel")
    if float(rng.uniform()) < BRICK_SHARE:
        wanted.append("brick-stack")
    if float(rng.uniform()) < EXTRA_TANK_SHARE:
        wanted.append(("tank-black", "tank-blue")[int(rng.integers(0, 2))])
    for kind in wanted:
        for _ in range(20):
            x, y = float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy))
            if not room.contains(Point(x, y)) or any(
                math.hypot(x - a, y - b) < 3.0 for a, b in spots
            ):
                continue
            if kind == "solar-panel":  # a panel on a low stand: a thin slab above a short post
                _box(x, y, 0.15, 0.15, top, top + 0.35, mesh, "wall-concrete")
                _box(x, y, 2.0, 1.3, top + 0.35, top + 0.45, mesh, kind)
            elif kind == "brick-stack":
                _box(x, y, 1.2, 0.7, top, top + 0.55, mesh, kind)
            else:
                _prism(x, y, float(rng.uniform(0.45, 0.65)), top, top + 1.1, 10, mesh, kind)
            spots.append((x, y))
            break


def _rooftop_gear(
    poly: Polygon,
    room: Any,
    spots: list[tuple[float, float]],
    top: float,
    mesh: Mesh,
    rng: Any,
    wall: str,
    floors: int,
) -> None:
    """Air-conditioner boxes, a dish on a post, a vent stack and a stair hut on flat roofs.

    A stream of its own, so every building drawn before this keeps its look.
    """
    if room.is_empty:
        return
    minx, miny, maxx, maxy = poly.bounds
    wanted: list[str] = []
    if float(rng.uniform()) < AC_SHARE:
        wanted += ["ac"] * int(rng.integers(1, 4))
    if float(rng.uniform()) < DISH_SHARE:
        wanted.append("dish")
    if float(rng.uniform()) < VENT_SHARE:
        wanted.append("vent")
    if floors >= 3 and float(rng.uniform()) < STAIR_HUT_SHARE:
        wanted.append("hut")
    for kind in wanted:
        for _ in range(20):
            x, y = float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy))
            gap = 2.0 if kind == "ac" else 3.0
            near = any(math.hypot(x - a, y - b) < gap for a, b in spots)
            if not room.contains(Point(x, y)) or near:
                continue
            if kind == "ac":
                _box(x, y, 0.9, 0.55, top, top + 0.7, mesh, "metal-grey")
            elif kind == "dish":
                _box(x, y, 0.1, 0.1, top, top + 1.5, mesh, "metal-grey")
                r = float(rng.uniform(0.4, 0.6))
                _prism(x, y, r, top + 1.5, top + 1.56, 12, mesh, "metal-grey")
            elif kind == "vent":
                height = float(rng.uniform(1.2, 2.0))
                _prism(x, y, 0.15, top, top + height, 8, mesh, "metal-grey")
            else:
                _box(x, y, 2.4, 2.2, top, top + 2.5, mesh, wall)
            spots.append((x, y))
            break


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
