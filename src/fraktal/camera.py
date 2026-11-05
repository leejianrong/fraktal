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

from fraktal.ids import substream
from fraktal.kernels import flight

KINDS = ("still", "orbit", "flight", "path")
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
) -> Pose:
    """The pose that looks at ``target`` from ``altitude_m``, behind it along ``heading_deg``."""
    back = altitude_m * math.tan(math.radians(off_nadir_deg))
    h = math.radians(heading_deg)
    x = target[0] - back * math.sin(h)
    y = target[1] - back * math.cos(h)
    return Pose(index, t_s, x, y, altitude_m, heading_deg % 360.0, off_nadir_deg, fov_deg,
                target[0], target[1])  # fmt: skip


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
) -> list[Pose]:
    """Poses for ``still`` (one frame), ``orbit`` (circle the tile centre), ``flight`` (a line) or
    ``path`` (fly through ``waypoints``, in tile metres, with bounded acceleration and yaw rate; the
    duration follows).

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
            max_yaw_rate_dps,
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
        poses.append(
            _pose(
                i, float(run["t"][i]), target, float(heading[i]), off_nadir_deg, altitude_m, fov_deg
            )  # fmt: skip
        )
    _check_inside(poses, size_m)
    return poses


__all__ = ["KINDS", "Pose", "trajectory"]
