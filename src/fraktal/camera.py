"""Camera poses and seeded trajectories for the 3D path (ADR-0010, FRK-51).

Everything is in the exported scene's tile coordinates: metres east (x) and north (y) from the
tile's south-west corner, z up from the ground plane. A pose is where the camera is, which way it
points (``heading_deg`` clockwise from north, ``off_nadir_deg`` from straight down, 0 being a
nadir view) and its field of view across the image. The pose of every frame is written to
``render.json`` so a sequence can be repeated or used as ground truth.

``trajectory`` is deterministic: the same arguments and seed give the same poses, and the seed is
spent through ``fraktal.ids.substream`` only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from fraktal.ids import substream
from fraktal.kernels import flight

KINDS = ("still", "orbit", "flight", "path")
STREET = "street"  # eye-level walk along a route; needs the scene, so it is not in ``trajectory``
EYE_M = 1.6
CLEARANCE_RADIUS_M = 1.5  # the ground is the highest point this close to the camera
WALL_THICKNESS_M = 0.3
LATERAL_CHOICES = (0.0, 2.0, -2.0, 3.5, -3.5, 5.0, -5.0)  # metres from the road centre line
EDGE_MARGIN = 0.1  # the view centre stays this share of the tile side away from the tile edge


@dataclass(frozen=True)
class Pose:
    index: int
    t_s: float
    x: float
    y: float
    z: float
    heading_deg: float
    off_nadir_deg: float
    fov_deg: float
    target_x: float  # where the centre of the image meets the ground
    target_y: float

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


def _pose(
    index: int,
    t_s: float,
    target: tuple[float, float],
    heading_deg: float,
    off_nadir_deg: float,
    altitude_m: float,
    fov_deg: float,
    z_m: float | None = None,
) -> Pose:
    """The pose that looks at ``target`` from ``altitude_m``, behind it along ``heading_deg``.

    ``z_m`` is the camera's height above the tile's datum when the altitude is above ground."""
    back = altitude_m * math.tan(math.radians(off_nadir_deg))
    h = math.radians(heading_deg)
    x = target[0] - back * math.sin(h)
    y = target[1] - back * math.cos(h)
    return Pose(index, t_s, x, y, altitude_m if z_m is None else z_m, heading_deg % 360.0,
                off_nadir_deg, fov_deg, target[0], target[1])  # fmt: skip


def trajectory(
    kind: str,
    size_m: float,
    seed: int,
    *,
    altitude_m: float = 300.0,
    off_nadir_deg: float = 35.0,
    fov_deg: float = 40.0,
    duration_s: float = 2.0,
    fps: float = 12.0,
    speed_mps: float = 40.0,
    sweep_deg: float = 90.0,
    waypoints: list[tuple[float, float]] | None = None,
    max_accel: float = 3.0,
    max_yaw_rate_dps: float = 60.0,
    ground: Any = None,
) -> list[Pose]:
    """Poses for ``still`` (one frame), ``orbit`` (circle the tile centre), ``flight`` (a line) or
    ``path`` (fly through ``waypoints``, in tile metres, with bounded acceleration and yaw rate; the
    duration follows). With ``ground`` (``ground(x, y)`` in metres), a path's ``altitude_m`` is
    above the highest ground within ``CLEARANCE_RADIUS_M`` of the camera, so it cannot go under
    the terrain; without it the altitude is absolute, as before.

    The start heading is drawn from the seed. A flight is a straight pass over the tile centre at
    ``speed_mps`` with a small slow wobble in heading and height. Raises ``ValueError`` when the
    view centre would leave the tile (shorten the flight or the duration).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown trajectory {kind!r}; choose from {', '.join(KINDS)}")
    if not 0.0 <= off_nadir_deg < 80.0:
        raise ValueError("off-nadir angle must be in [0, 80) degrees")
    rng = substream(seed, f"camera:{kind}")
    heading0 = float(rng.uniform(0.0, 360.0))
    phase = float(rng.uniform(0.0, 2 * math.pi))
    if kind == "path":
        return _path(
            waypoints, size_m, altitude_m, off_nadir_deg, fov_deg, fps, speed_mps, max_accel,
            max_yaw_rate_dps, ground,
        )  # fmt: skip
    n = 1 if kind == "still" else max(2, round(duration_s * fps))
    centre = (size_m / 2, size_m / 2)
    poses: list[Pose] = []
    for i in range(n):
        t = i / fps
        u = i / (n - 1) if n > 1 else 0.0
        if kind == "still":
            pose = _pose(i, t, centre, heading0, off_nadir_deg, altitude_m, fov_deg)
        elif kind == "orbit":
            heading = heading0 + sweep_deg * u
            pose = _pose(i, t, centre, heading, off_nadir_deg, altitude_m, fov_deg)
        else:
            run = speed_mps * duration_s
            h = math.radians(heading0)
            along = -run / 2 + run * u
            target = (centre[0] + along * math.sin(h), centre[1] + along * math.cos(h))
            wobble = math.sin(2 * math.pi * 0.5 * u + phase)
            pose = _pose(
                i, t, target, heading0 + 1.0 * wobble, off_nadir_deg, altitude_m + 2.0 * wobble,
                fov_deg,
            )  # fmt: skip
        poses.append(pose)
    _check_inside(poses, size_m)
    return poses


def _check_inside(poses: list[Pose], size_m: float) -> None:
    lo, hi = EDGE_MARGIN * size_m, (1 - EDGE_MARGIN) * size_m
    for p in poses:
        if not (lo <= p.target_x <= hi and lo <= p.target_y <= hi):
            raise ValueError(
                f"frame {p.index} looks at ({p.target_x:.0f}, {p.target_y:.0f}), outside the tile "
                f"margin [{lo:.0f}, {hi:.0f}]; use a larger tile or a shorter flight"
            )


def _path(
    waypoints: list[tuple[float, float]] | None,
    size_m: float,
    altitude_m: float,
    off_nadir_deg: float,
    fov_deg: float,
    fps: float,
    speed_mps: float,
    max_accel: float,
    max_yaw_rate_dps: float,
    ground: Any = None,
) -> list[Pose]:
    """Fly a polyline with a point-mass drone (``fraktal.kernels.flight``): it starts and ends at
    rest, accelerates within ``max_accel``, rounds corners and turns at most ``max_yaw_rate_dps``.
    The camera looks ahead along the heading."""
    if waypoints is None or len(waypoints) < 2:
        raise ValueError("a path needs at least two waypoints")
    run = flight.fly(
        np.asarray(waypoints, dtype=float), cruise_mps=speed_mps, max_accel=max_accel,
        max_yaw_rate_dps=max_yaw_rate_dps, fps=fps,
    )  # fmt: skip
    back = altitude_m * math.tan(math.radians(off_nadir_deg))
    heading = np.degrees(np.unwrap(np.radians(run["yaw_deg"])))
    poses = []
    for i, (x, y) in enumerate(run["xy"]):
        h = math.radians(float(heading[i]))
        target = (float(x) + back * math.sin(h), float(y) + back * math.cos(h))
        z = None if ground is None else float(ground_under(ground, x, y)) + altitude_m
        poses.append(
            _pose(
                i,
                float(run["t"][i]),
                target,
                float(heading[i]),
                off_nadir_deg,
                altitude_m,
                fov_deg,
                z,
            )  # fmt: skip
        )
    _check_inside(poses, size_m)
    return poses


def follow_poses(
    x: Any,
    y: Any,
    heading: Any,
    t: Any,
    *,
    mount_height_m: float,
    forward_m: float,
    pitch_deg: float,
    fov_deg: float,
    ground: Any = None,
) -> list[Pose]:
    """A camera riding on an agent: ``heading`` is radians anticlockwise from east, as in tracks.

    The camera sits ``forward_m`` ahead of the agent's centre and ``mount_height_m`` above the
    ground (``ground(x, y)`` gives the ground height, zero when omitted) and looks along the
    heading, ``pitch_deg`` below the horizon. Its target is a point 40 m ahead on the ground.
    """
    x, y, heading, t = (np.asarray(a, dtype="float64") for a in (x, y, heading, t))
    if not 0.0 <= pitch_deg < 60.0:
        raise ValueError("pitch must be in [0, 60) degrees below the horizon")
    compass = (90.0 - np.degrees(heading)) % 360.0  # clockwise from north, like the other poses
    sin, cos = np.sin(np.radians(compass)), np.cos(np.radians(compass))
    cx, cy = x + forward_m * sin, y + forward_m * cos
    z = mount_height_m + (
        np.asarray(ground(cx, cy), dtype="float64") if ground else np.zeros_like(cx)
    )
    return [
        Pose(i, float(t[i]), float(cx[i]), float(cy[i]), float(z[i]), float(compass[i]),
             90.0 - pitch_deg, fov_deg, float(cx[i] + 40.0 * sin[i]), float(cy[i] + 40.0 * cos[i]))
        for i in range(len(t))
    ]  # fmt: skip


def ground_under(ground: Any, x: Any, y: Any, radius_m: float = CLEARANCE_RADIUS_M) -> Any:
    """The highest ground at ``(x, y)`` and eight points ``radius_m`` around it."""
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    heights = [np.asarray(ground(x, y), dtype="float64")]
    for a in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
        heights.append(np.asarray(ground(x + radius_m * np.cos(a), y + radius_m * np.sin(a))))
    return np.max(heights, axis=0)


def obstacles(buildings: list[dict[str, Any]], walls: list[list[float]]) -> Any:
    """Everything an eye-level camera must stay out of, as one shapely geometry in tile metres:
    building footprints (``scene["buildings"]``) and wall segments (``[x0, y0, x1, y1]``, from
    each of ``scene["walls"]``'s ``segments``), walls given a thickness."""
    parts: list[Any] = [Polygon(b["ring"]) for b in buildings if len(b["ring"]) >= 4]
    parts += [
        LineString([(x0, y0), (x1, y1)]).buffer(WALL_THICKNESS_M / 2)
        for x0, y0, x1, y1 in walls
        if (x0, y0) != (x1, y1)
    ]
    return unary_union(parts)


def clear_xy(x: Any, y: Any, hazards: Any, margin_m: float) -> tuple[Any, Any]:
    """Move each point out of ``hazards`` (and ``margin_m`` clear of it), away from the nearest
    edge. A point already further than ``margin_m`` stays where it is."""
    out_x, out_y = np.array(x, dtype="float64"), np.array(y, dtype="float64")
    if hazards.is_empty:
        return out_x, out_y
    for i in range(len(out_x)):
        for _ in range(3):  # a push can land near a second wall
            pt = Point(out_x[i], out_y[i])
            if hazards.contains(pt):
                q = nearest_points(hazards.boundary, pt)[0]
                sign = -1.0
            elif hazards.distance(pt) < margin_m:
                q = nearest_points(hazards, pt)[0]
                sign = 1.0
            else:
                break
            d = np.array([pt.x - q.x, pt.y - q.y]) * sign
            norm = float(np.hypot(*d))
            if norm < 1e-9:
                break
            out_x[i] = q.x + d[0] / norm * margin_m
            out_y[i] = q.y + d[1] / norm * margin_m
    return out_x, out_y


def blocked_share(x: Any, y: Any, heading_deg: Any, hazards: Any, view_m: float = 2.5) -> float:
    """The share of frames whose line of sight, ``view_m`` ahead along the compass heading,
    meets a hazard. Those frames are mostly a wall filling the picture."""
    if hazards.is_empty or len(x) == 0:
        return 0.0
    h = np.radians(np.asarray(heading_deg, dtype="float64"))
    hits = 0
    for xi, yi, hi in zip(np.asarray(x), np.asarray(y), h, strict=True):
        ray = LineString([(xi, yi), (xi + view_m * math.sin(hi), yi + view_m * math.cos(hi))])
        hits += bool(hazards.intersects(ray))
    return hits / len(x)


def street_poses(
    waypoints: list[tuple[float, float]],
    *,
    speed_mps: float,
    fps: float,
    ground: Any,
    hazards: Any,
    eye_m: float = EYE_M,
    lateral_m: float | None = 0.0,
    margin_m: float = 1.0,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    smooth_m: float = 6.0,
) -> tuple[list[Pose], dict[str, float]]:
    """Walk ``waypoints`` (tile metres) at ``speed_mps`` with the camera ``eye_m`` above the
    ground, looking along the route.

    ``lateral_m`` shifts the route to the left (positive) or right of the line, for a footpath
    beside a road centre line; ``None`` tries ``LATERAL_CHOICES`` (lane, kerb, footpath on both
    sides) and keeps the one with the clearest view, the nearest to the line on a tie.

    The camera stays ``margin_m`` clear of ``hazards`` (see ``obstacles``), sits above the
    highest ground within ``CLEARANCE_RADIUS_M`` and has its route smoothed over ``smooth_m``.
    Returns the poses and ``{"blocked_share", "pushed_m"}`` (plus ``lateral_m`` when chosen), the
    share being of frames with a hazard within 2.5 m ahead.
    """
    if lateral_m is None:
        runs = [
            street_poses(
                waypoints,
                speed_mps=speed_mps,
                fps=fps,
                ground=ground,
                hazards=hazards,
                eye_m=eye_m,
                lateral_m=d,
                margin_m=margin_m,
                pitch_deg=pitch_deg,
                fov_deg=fov_deg,
                smooth_m=smooth_m,
            )  # fmt: skip
            for d in LATERAL_CHOICES
        ]
        poses, stats = min(runs, key=lambda r: r[1]["blocked_share"])
        return poses, {**stats, "lateral_m": LATERAL_CHOICES[runs.index((poses, stats))]}
    if len(waypoints) < 2:
        raise ValueError("a street walk needs at least two waypoints")
    if speed_mps <= 0 or fps <= 0:
        raise ValueError("speed and fps must be positive")
    line = LineString(waypoints)
    if lateral_m:
        line = line.offset_curve(lateral_m)
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda g: g.length)
    step = speed_mps / fps
    s = np.arange(0.0, line.length, step)
    if len(s) < 2:
        raise ValueError("the route is too short for even two frames")
    pts = np.array([line.interpolate(float(d)).coords[0] for d in s])
    k = max(1, round(smooth_m / step / 2))  # moving average, edges padded
    padded = np.pad(pts, ((k, k), (0, 0)), mode="edge")
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(2)], axis=1)
    heading = np.arctan2(np.gradient(sm[:, 1]), np.gradient(sm[:, 0]))  # anticlockwise from east
    px, py = clear_xy(sm[:, 0], sm[:, 1], hazards, margin_m)
    pushed = float(np.hypot(px - sm[:, 0], py - sm[:, 1]).max())
    poses = follow_poses(
        px, py, heading, np.arange(len(s)) / fps, mount_height_m=eye_m, forward_m=0.0,
        pitch_deg=pitch_deg, fov_deg=fov_deg, ground=lambda gx, gy: ground_under(ground, gx, gy),
    )  # fmt: skip
    share = blocked_share(px, py, [p.heading_deg for p in poses], hazards)
    return poses, {"blocked_share": round(share, 4), "pushed_m": round(pushed, 3)}


__all__ = [
    "KINDS", "Pose", "STREET", "blocked_share", "clear_xy", "follow_poses", "ground_under",
    "obstacles", "street_poses", "trajectory",
]  # fmt: skip
