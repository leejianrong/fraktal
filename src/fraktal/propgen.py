"""Parametric props written as code (FRK-130): shipping container, water tank, market stall,
signboard and tuk-tuk. Each generator takes a seed and returns coloured parts, in metres, x along
the length, y across, z up, origin at the ground below the centre.

Pure numpy, no Blender: the shapes are unit-testable and deterministic (ADR-0003, randomness only
through ``substream``). ``scripts/props/bpy_build.py`` turns the parts into Blender objects, renders
them and exports GLB. Every material is a flat colour with a roughness, so no third-party asset or
model is involved.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fraktal.ids import substream

# material key -> (linear-ish sRGB colour, roughness, metallic)
MATERIALS: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "steel-dark": ((0.12, 0.12, 0.13), 0.55, 0.8),
    "rubber": ((0.04, 0.04, 0.04), 0.9, 0.0),
    "wood": ((0.45, 0.30, 0.16), 0.8, 0.0),
    "wood-dark": ((0.28, 0.18, 0.10), 0.85, 0.0),
    "glass": ((0.55, 0.65, 0.7), 0.1, 0.0),
    "plastic-black": ((0.05, 0.05, 0.055), 0.5, 0.0),
    "plastic-blue": ((0.08, 0.2, 0.55), 0.45, 0.0),
    "plastic-white": ((0.6, 0.6, 0.58), 0.45, 0.0),
    "paint-red": ((0.55, 0.1, 0.08), 0.6, 0.3),
    "paint-rust": ((0.42, 0.2, 0.1), 0.85, 0.4),
    "paint-blue": ((0.1, 0.22, 0.45), 0.6, 0.3),
    "paint-green": ((0.12, 0.35, 0.18), 0.6, 0.3),
    "paint-yellow": ((0.85, 0.65, 0.08), 0.55, 0.2),
    "paint-grey": ((0.45, 0.46, 0.46), 0.6, 0.3),
    "paint-orange": ((0.75, 0.32, 0.08), 0.6, 0.3),
    "tarp-blue": ((0.1, 0.3, 0.7), 0.7, 0.0),
    "tarp-green": ((0.15, 0.4, 0.2), 0.7, 0.0),
    "tarp-orange": ((0.8, 0.4, 0.1), 0.7, 0.0),
    "tarp-white": ((0.6, 0.6, 0.56), 0.7, 0.0),
    "wall-cream": ((0.80, 0.74, 0.62), 0.9, 0.0),
    "wall-ochre": ((0.72, 0.55, 0.36), 0.9, 0.0),
    "wall-grey": ((0.55, 0.55, 0.52), 0.9, 0.0),
    "wall-brick": ((0.55, 0.30, 0.22), 0.9, 0.0),
    "wall-white": ((0.84, 0.82, 0.76), 0.9, 0.0),
    "wall-sand": ((0.74, 0.64, 0.48), 0.9, 0.0),
    "wall-terracotta": ((0.62, 0.36, 0.26), 0.9, 0.0),
    "sheet-zinc": ((0.52, 0.54, 0.54), 0.45, 0.7),
    "sheet-rust": ((0.45, 0.25, 0.15), 0.7, 0.4),
    "sheet-blue": ((0.15, 0.25, 0.45), 0.55, 0.5),
    "fruit-red": ((0.7, 0.1, 0.08), 0.6, 0.0),
    "fruit-yellow": ((0.8, 0.65, 0.1), 0.6, 0.0),
    "fruit-green": ((0.25, 0.5, 0.12), 0.6, 0.0),
}


@dataclass(frozen=True)
class Part:
    material: str
    verts: np.ndarray  # (n, 3) float64
    faces: np.ndarray  # (m, 3) int, outward counter-clockwise


@dataclass(frozen=True)
class Prop:
    kind: str
    seed: int
    parts: tuple[Part, ...]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        allv = np.vstack([p.verts for p in self.parts])
        return allv.min(axis=0), allv.max(axis=0)

    def size(self) -> tuple[float, float, float]:
        lo, hi = self.bounds()
        s = hi - lo
        return float(s[0]), float(s[1]), float(s[2])


_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2],  # bottom (-z)
        [4, 5, 6], [4, 6, 7],  # top (+z)
        [0, 1, 5], [0, 5, 4],  # -y
        [2, 3, 7], [2, 7, 6],  # +y
        [1, 2, 6], [1, 6, 5],  # +x
        [3, 0, 4], [3, 4, 7],  # -x
    ]
)  # fmt: skip


def _yaw(verts: np.ndarray, yaw: float, about: tuple[float, float]) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    out = verts.copy()
    dx, dy = verts[:, 0] - about[0], verts[:, 1] - about[1]
    out[:, 0] = about[0] + c * dx - s * dy
    out[:, 1] = about[1] + s * dx + c * dy
    return out


def box(
    material: str,
    centre: tuple[float, float, float],
    size: tuple[float, float, float],
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> Part:
    """An axis-aligned box, optionally pitched about y then yawed about z around its centre."""
    sx, sy, sz = size
    corners = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
        dtype=float,
    ) * np.array([sx, sy, sz]) / 2  # fmt: skip
    if pitch:
        c, s = math.cos(pitch), math.sin(pitch)
        x, z = corners[:, 0].copy(), corners[:, 2].copy()
        corners[:, 0], corners[:, 2] = c * x + s * z, -s * x + c * z
    verts = corners + np.array(centre)
    if yaw:
        verts = _yaw(verts, yaw, (centre[0], centre[1]))
    return Part(material, verts, _BOX_FACES.copy())


def cylinder(
    material: str,
    centre: tuple[float, float, float],
    radius: float,
    length: float,
    axis: str = "z",
    sides: int = 20,
) -> Part:
    """A closed cylinder along z (upright), or along y (wheels, axle-aligned)."""
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    bottom = np.column_stack([ring, np.full(sides, -length / 2)])
    top = np.column_stack([ring, np.full(sides, length / 2)])
    verts = np.vstack([bottom, top, [[0, 0, -length / 2]], [[0, 0, length / 2]]])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces += [[i, j, sides + j], [i, sides + j, sides + i]]
        faces.append([2 * sides, j, i])
        faces.append([2 * sides + 1, sides + i, sides + j])
    if axis == "y":  # a rotation about x (z -> y), so the winding is unchanged
        verts = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])
    return Part(material, verts + np.array(centre), np.array(faces))


def _wheel(material: str, x: float, y: float, radius: float, width: float) -> Part:
    return cylinder(material, (x, y, radius), radius, width, axis="y", sides=18)


def _pick(rng: np.random.Generator, options: list[str]) -> str:
    return options[int(rng.integers(len(options)))]


def container(seed: int) -> Prop:
    """An ISO shipping container: corrugated side walls, frame, door end with locking bars."""
    rng = substream(seed, "propgen.container")
    length = 6.06 if rng.random() < 0.7 else 12.19
    width, height = 2.44, 2.59
    paint = _pick(
        rng, ["paint-rust", "paint-blue", "paint-green", "paint-red", "paint-grey", "paint-orange"]
    )
    parts: list[Part] = []
    pitch = 0.15
    n = int(length / pitch)
    z0, zh = 0.15, height - 0.3
    for side in (-1, 1):  # corrugation: alternating tilted ribs along each side wall
        for i in range(n):
            x = -length / 2 + (i + 0.5) * pitch
            tilt = 0.5 if i % 2 else -0.5
            parts.append(
                box(
                    paint,
                    (x, side * (width / 2 - 0.03), z0 + zh / 2),
                    (pitch * 0.6, 0.04, zh),
                    yaw=tilt,
                )
            )
    parts.append(box(paint, (0, 0, height - 0.04), (length, width, 0.08)))  # roof
    parts.append(box("steel-dark", (0, 0, 0.1), (length, width, 0.2)))  # base frame
    parts.append(
        box(paint, (-length / 2 + 0.02, 0, height / 2), (0.04, width, height))
    )  # blank end
    parts.append(box(paint, (length / 2 - 0.02, 0, height / 2), (0.04, width, height)))  # door end
    for sx in (-1, 1):
        for sy in (-1, 1):  # corner posts
            parts.append(
                box(
                    "steel-dark",
                    (sx * (length / 2 - 0.08), sy * (width / 2 - 0.08), height / 2),
                    (0.16, 0.16, height),
                )
            )
    for k in range(4):  # locking bars on the doors
        y = -0.75 + k * 0.5
        parts.append(
            cylinder("steel-dark", (length / 2 + 0.03, y, height / 2), 0.025, height - 0.3, sides=8)
        )
    return Prop("container", seed, tuple(parts))


def water_tank(seed: int) -> Prop:
    """A cylindrical roto-moulded tank on a steel stand with a lid."""
    rng = substream(seed, "propgen.water_tank")
    radius = float(rng.uniform(0.42, 0.62))
    body_h = float(rng.uniform(1.0, 1.5))
    leg_h = float(rng.uniform(0.4, 0.9))
    colour = _pick(rng, ["plastic-black", "plastic-black", "plastic-blue", "plastic-white"])
    parts = [cylinder(colour, (0, 0, leg_h + body_h / 2), radius, body_h, sides=28)]
    parts.append(
        cylinder(colour, (0, 0, leg_h + body_h + 0.04), radius * 0.4, 0.08, sides=16)
    )  # lid
    for rim in (0.25, 0.75):  # moulded ribs
        parts.append(cylinder(colour, (0, 0, leg_h + body_h * rim), radius * 1.03, 0.05, sides=28))
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(
                box(
                    "steel-dark",
                    (sx * radius * 0.75, sy * radius * 0.75, leg_h / 2),
                    (0.06, 0.06, leg_h),
                )
            )
    parts.append(
        box("steel-dark", (0, 0, leg_h - 0.03), (radius * 2.2, radius * 2.2, 0.06))
    )  # platform
    return Prop("water_tank", seed, tuple(parts))


def silo(seed: int) -> Prop:
    """A grain or cement silo: a tall steel cylinder with a conical cap, a ladder and a collar."""
    rng = substream(seed, "propgen.silo")
    radius = float(rng.uniform(1.4, 2.4))
    height = float(rng.uniform(6.0, 10.0))
    colour = _pick(rng, ["sheet-zinc", "sheet-zinc", "paint-grey", "plastic-white"])
    parts = [cylinder(colour, (0, 0, height / 2), radius, height, sides=32)]
    cap_h = radius * 0.5
    ang = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    ring = np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.full(32, height)])
    apex = np.array([[0.0, 0.0, height + cap_h]])
    faces = [[i, (i + 1) % 32, 32] for i in range(32)]
    parts.append(Part(colour, np.vstack([ring, apex]), np.array(faces)))
    collar = cylinder("steel-dark", (0, 0, height * 0.05), radius * 1.04, height * 0.1, sides=32)
    parts.append(collar)
    parts.append(box("steel-dark", (radius + 0.05, 0, height / 2), (0.08, 0.5, height)))  # ladder
    return Prop("silo", seed, tuple(parts))


def market_stall(seed: int) -> Prop:
    """A wooden market stall with a tarpaulin roof and goods on the counter."""
    rng = substream(seed, "propgen.market_stall")
    length = float(rng.uniform(1.8, 2.8))
    depth = float(rng.uniform(1.2, 1.8))
    height = float(rng.uniform(2.1, 2.5))
    tarp = _pick(rng, ["tarp-blue", "tarp-green", "tarp-orange", "tarp-white"])
    parts: list[Part] = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            h = height if sy < 0 else height - 0.25  # the roof slopes down at the back
            parts.append(
                box(
                    "wood",
                    (sx * (length / 2 - 0.04), sy * (depth / 2 - 0.04), h / 2),
                    (0.08, 0.08, h),
                )
            )
    slope = math.atan2(0.25, depth)
    parts.append(
        box(tarp, (0, 0, height - 0.125), (length + 0.3, math.hypot(depth + 0.3, 0.25), 0.04))
    )
    parts[-1] = _tilt_roof(parts[-1], -slope)
    parts.append(box("wood-dark", (0, -depth / 2 + 0.3, 0.9), (length, 0.6, 0.06)))  # counter top
    parts.append(box("wood", (0, -depth / 2 + 0.3, 0.45), (length - 0.1, 0.5, 0.8)))  # counter body
    if rng.random() < 0.6:
        parts.append(
            box("wood", (0, depth / 2 - 0.05, height / 2 - 0.1), (length, 0.05, height - 0.3))
        )  # back wall
    n_goods = int(rng.integers(4, 9))
    for i in range(n_goods):
        x = -length / 2 + 0.3 + (length - 0.6) * (i + rng.random() * 0.5) / n_goods
        parts.append(
            box(
                _pick(rng, ["fruit-red", "fruit-yellow", "fruit-green", "wood-dark"]),
                (x, -depth / 2 + 0.3, 1.0 + 0.1),
                (0.25, 0.3, 0.2),
            )
        )
    return Prop("market_stall", seed, tuple(parts))


def _tilt_roof(part: Part, slope: float) -> Part:
    """Tilt a flat roof slab about the x axis so the front is higher than the back."""
    v = part.verts.copy()
    cy, cz = v[:, 1].mean(), v[:, 2].mean()
    dy, dz = v[:, 1] - cy, v[:, 2] - cz
    c, s = math.cos(slope), math.sin(slope)
    v[:, 1] = cy + c * dy - s * dz
    v[:, 2] = cz + s * dy + c * dz
    return Part(part.material, v, part.faces)


def signboard(seed: int) -> Prop:
    """A flat board on two wooden posts, with a frame around the face."""
    rng = substream(seed, "propgen.signboard")
    width = float(rng.uniform(1.2, 2.4))
    board_h = float(rng.uniform(0.5, 0.9))
    post_h = float(rng.uniform(1.8, 2.6))
    face = _pick(rng, ["paint-yellow", "paint-blue", "paint-red", "plastic-white", "paint-green"])
    parts = [box(face, (0, 0, post_h - board_h / 2 - 0.1), (width, 0.05, board_h))]
    parts.append(box("wood-dark", (0, 0, post_h - 0.08), (width + 0.1, 0.07, 0.06)))  # frame top
    parts.append(
        box("wood-dark", (0, 0, post_h - board_h - 0.12), (width + 0.1, 0.07, 0.06))
    )  # frame bottom
    for sx in (-1, 1):
        parts.append(box("wood", (sx * (width / 2 - 0.05), -0.06, post_h / 2), (0.1, 0.1, post_h)))
    return Prop("signboard", seed, tuple(parts))


def tuk_tuk(seed: int) -> Prop:
    """A three-wheeler: one front wheel, a rear body with seat and a canopy on pillars."""
    rng = substream(seed, "propgen.tuk_tuk")
    body = _pick(rng, ["paint-yellow", "paint-green", "paint-blue", "paint-red"])
    wheel_r = float(rng.uniform(0.2, 0.26))
    length = float(rng.uniform(2.6, 2.9))
    width = float(rng.uniform(1.25, 1.45))
    height = float(rng.uniform(1.65, 1.8))
    parts: list[Part] = [
        _wheel("rubber", length / 2 - 0.35, 0.0, wheel_r, 0.12),  # front
        _wheel("rubber", -length / 2 + 0.45, -width / 2 + 0.08, wheel_r, 0.14),
        _wheel("rubber", -length / 2 + 0.45, width / 2 - 0.08, wheel_r, 0.14),
        box(body, (0.0, 0, 0.45 + wheel_r / 2), (length * 0.85, width * 0.9, 0.35)),  # chassis tub
        box(body, (length / 2 - 0.55, 0, 0.8), (0.7, 0.55, 0.55)),  # front cowl
        box(
            "glass", (length / 2 - 0.85, 0, 1.15), (0.04, width * 0.85, 0.5), pitch=-0.35
        ),  # windscreen
        box("steel-dark", (length / 2 - 0.45, 0, 1.0), (0.05, 0.6, 0.05)),  # handlebar
        box("wood-dark", (-0.35, 0, 0.82), (0.9, width * 0.8, 0.12)),  # passenger bench
        box("wood-dark", (-0.8, 0, 1.1), (0.08, width * 0.8, 0.5)),  # bench back
    ]
    roof_x = -0.2
    parts.append(box(body, (roof_x, 0, height - 0.04), (length * 0.7, width, 0.07)))  # canopy
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(
                box(
                    "steel-dark",
                    (roof_x + sx * length * 0.32, sy * (width / 2 - 0.04), (height + 0.5) / 2),
                    (0.05, 0.05, height - 0.5),
                )
            )
    return Prop("tuk_tuk", seed, tuple(parts))


GENERATORS: dict[str, Callable[[int], Prop]] = {
    "container": container,
    "water_tank": water_tank,
    "market_stall": market_stall,
    "signboard": signboard,
    "tuk_tuk": tuk_tuk,
    "silo": silo,
}
