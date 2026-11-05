"""A point-mass drone that follows a polyline with bounded acceleration and yaw rate (ADR-0005).

Arrays in, arrays out. The drone starts at rest on the first waypoint, chases a point a short way
ahead on the path, speeds up and slows down within ``max_accel`` (also the braking limit, so it
stops on the last waypoint), and turns its heading toward its velocity no faster than
``max_yaw_rate``. The integration is sequential by nature (a few thousand small steps), so it is a
plain loop rather than a vectorised one. Deterministic: no randomness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

DT = 0.02  # integration step, seconds
LOOKAHEAD_S = 1.0  # how far ahead on the path the drone aims, in seconds of cruise
YAW_TAU = 0.4  # seconds; how quickly the heading chases the velocity direction
VEL_TAU = 0.6  # seconds; how quickly the velocity chases the wanted velocity
CORNER_RADIUS_M = 15.0  # a corner is flown at the speed where this radius needs ``max_accel``
CORNER_MIN_TURN_DEG = 10.0  # gentler bends than this are flown at full speed
PROFILE_MARGIN = 0.8  # the speed profile uses this share of ``max_accel``, so tracking has room
DONE_DISTANCE_M = 1.5  # a little over the one-metre path grid
DONE_SPEED_MPS = 0.2


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _speed_profile(
    pts: np.ndarray[Any, Any],
    along: np.ndarray[Any, Any],
    s_grid: np.ndarray[Any, Any],
    cruise: float,
    accel: float,
) -> np.ndarray[Any, Any]:
    """The fastest speed at each metre of the path: cruise, slower at corners, zero at the end.

    A cap at each corner (from its turn angle), then a backward pass so the drone can brake in time
    and a forward pass so it can accelerate out, both at ``accel``.
    """
    cap = np.full(len(s_grid), float(cruise))
    corner = min(cruise, math.sqrt(accel * CORNER_RADIUS_M))
    for k in range(1, len(pts) - 1):
        a, b = pts[k] - pts[k - 1], pts[k + 1] - pts[k]
        turn = abs(_wrap(math.atan2(b[0], b[1]) - math.atan2(a[0], a[1])))
        if math.degrees(turn) > CORNER_MIN_TURN_DEG:
            cap[int(np.argmin(np.abs(s_grid - along[k])))] = corner
    cap[0], cap[-1] = 0.0, 0.0  # starts and ends at rest
    ds = float(s_grid[1] - s_grid[0])
    for i in range(len(cap) - 2, -1, -1):  # braking
        cap[i] = min(cap[i], math.sqrt(cap[i + 1] ** 2 + 2 * accel * ds))
    for i in range(1, len(cap)):  # accelerating
        cap[i] = min(cap[i], math.sqrt(cap[i - 1] ** 2 + 2 * accel * ds))
    return cap


def fly(
    waypoints: np.ndarray[Any, Any],
    *,
    cruise_mps: float,
    max_accel: float,
    max_yaw_rate_dps: float,
    fps: float,
    max_seconds: float | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    """Fly through ``waypoints`` (N, 2) in metres; return samples at ``fps`` as arrays.

    Keys: ``t`` (s), ``xy`` (M, 2), ``yaw_deg`` (heading clockwise from north), ``speed``.
    """
    pts = np.asarray(waypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("waypoints must be an (N, 2) array with N >= 2")
    if min(cruise_mps, max_accel, max_yaw_rate_dps, fps) <= 0:
        raise ValueError("speed, acceleration, yaw rate and fps must be positive")
    seg = np.hypot(*np.diff(pts, axis=0).T)
    if seg.sum() < 1e-6:
        raise ValueError("the waypoints are all the same point")
    along = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(along[-1])
    step = 1.0  # a dense copy of the path, one point a metre, to find the nearest part quickly
    s_grid = np.arange(0.0, total + step, step)
    dense = np.stack([np.interp(s_grid, along, pts[:, k]) for k in range(2)], axis=1)
    limit = max_seconds if max_seconds is not None else 4.0 * total / cruise_mps + 20.0
    max_yaw = math.radians(max_yaw_rate_dps)
    look = max(5.0, cruise_mps * LOOKAHEAD_S)
    profile = _speed_profile(pts, along, s_grid, cruise_mps, max_accel * PROFILE_MARGIN)

    p = pts[0].copy()
    v = np.zeros(2)
    first = dense[min(5, len(dense) - 1)] - dense[0]
    yaw = math.atan2(first[0], first[1])  # facing along the first leg
    idx = 0  # progress along the dense path never goes backwards
    every = max(1, round(1.0 / (fps * DT)))
    steps = int(limit / DT)
    t_out: list[float] = []
    xy_out: list[tuple[float, float]] = []
    yaw_out: list[float] = []
    speed_out: list[float] = []
    for i in range(steps + 1):
        if i % every == 0:
            t_out.append(i * DT)
            xy_out.append((float(p[0]), float(p[1])))
            yaw_out.append(math.degrees(yaw) % 360.0)
            speed_out.append(float(np.hypot(*v)))
        window = dense[idx : idx + 60]
        idx += int(np.argmin(np.hypot(*(window - p).T)))
        remaining = total - s_grid[idx]
        if remaining < DONE_DISTANCE_M and np.hypot(*v) < DONE_SPEED_MPS:
            break
        aim = dense[min(idx + int(look / step), len(dense) - 1)]
        to_aim = aim - p
        dist = float(np.hypot(*to_aim))
        ahead = (
            idx + 1 + int(float(np.hypot(*v)) * VEL_TAU / step)
        )  # the speed lags, so look further on
        want_speed = float(profile[min(ahead, len(profile) - 1)])
        want = to_aim / dist * want_speed if dist > 1e-9 else np.zeros(2)
        accel = (want - v) / VEL_TAU
        norm = float(np.hypot(*accel))
        if norm > max_accel:
            accel *= max_accel / norm
        v = v + accel * DT
        p = p + v * DT
        if np.hypot(*v) > 0.5:
            rate = _wrap(math.atan2(v[0], v[1]) - yaw) / YAW_TAU
            yaw += max(-max_yaw, min(max_yaw, rate)) * DT
    else:
        if remaining > 5.0:
            raise ValueError(f"the drone did not reach the end in {limit:.0f} s (stuck?)")
    return {
        "t": np.asarray(t_out),
        "xy": np.asarray(xy_out),
        "yaw_deg": np.asarray(yaw_out),
        "speed": np.asarray(speed_out),
    }


__all__ = ["fly"]
