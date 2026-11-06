"""Cars driving along fixed routes with speed limits (ADR-0005, ADR-0011 phase 2).

Arrays in, arrays out. Each car follows its own route, described by a speed cap at every metre
(``cap``, from the road class and the bends). ``braking_profile`` lowers the caps ahead of slow
sections so a car can brake in time; ``simulate`` then moves every car along its route, speeding up
and braking within the limits. Cars do not yet react to each other (car following is the next
slice). The cars are stepped together as arrays, and only the time loop is sequential.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def braking_profile(cap: np.ndarray[Any, Any], ds: float, brake: float) -> np.ndarray[Any, Any]:
    """The fastest speed at each grid point that still lets the car brake (at ``brake``) in time."""
    if cap.ndim != 2:
        raise ValueError("cap must be a 2D array, one row per car")
    if ds <= 0 or brake <= 0:
        raise ValueError("ds and brake must be positive")
    out = np.array(cap, dtype="float64")
    for i in range(out.shape[1] - 2, -1, -1):
        out[:, i] = np.minimum(out[:, i], np.sqrt(out[:, i + 1] ** 2 + 2.0 * brake * ds))
    return out


def simulate(
    profile: np.ndarray[Any, Any],
    lengths: np.ndarray[Any, Any],
    ds: float,
    s0: np.ndarray[Any, Any],
    v0: np.ndarray[Any, Any],
    *,
    accel: float,
    brake: float,
    dt: float,
    steps: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Distance along the route and speed of every car at ``steps + 1`` times, ``dt`` apart.

    ``lengths`` is each route's length in metres; a car that reaches its end stops there.
    """
    n = profile.shape[0]
    if profile.ndim != 2 or len(s0) != n or len(v0) != n or len(lengths) != n:
        raise ValueError("profile, lengths, s0 and v0 must have one row or entry per car")
    if min(accel, brake, dt) <= 0 or steps < 0:
        raise ValueError("accel, brake and dt must be positive and steps not negative")
    rows = np.arange(n)
    last = np.maximum((np.asarray(lengths) / ds).astype(int) - 1, 0)
    s = np.asarray(s0, dtype="float64").copy()
    v = np.asarray(v0, dtype="float64").copy()
    out_s = np.empty((n, steps + 1))
    out_v = np.empty((n, steps + 1))
    for k in range(steps + 1):
        out_s[:, k], out_v[:, k] = s, v
        index = np.minimum((s / ds).astype(int) + 1, last)
        want = profile[rows, index]
        v = np.maximum(v + np.clip(want - v, -brake * dt, accel * dt), 0.0)
        s = np.minimum(s + v * dt, np.asarray(lengths) - 1e-6)
    return out_s, out_v


def simulate_interacting(
    profile: np.ndarray[Any, Any],
    xy: np.ndarray[Any, Any],
    heading: np.ndarray[Any, Any],
    rank: np.ndarray[Any, Any],
    lengths: np.ndarray[Any, Any],
    ds: float,
    s0: np.ndarray[Any, Any],
    v0: np.ndarray[Any, Any],
    *,
    accel: float,
    comfort_brake: float,
    max_brake: float,
    dt: float,
    steps: int,
    time_gap: float = 1.2,
    min_gap: float = 2.0,
    car_length: float = 4.5,
    horizon: float = 80.0,
    conflict_radius: float = 4.0,
    conflict_time: float = 5.0,
    stop_margin: float = 2.0,
    patience: float = 5.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Like ``simulate``, but cars follow the car ahead and give way where paths cross.

    ``xy`` (N, L, 2) and ``heading`` (N, L) give each route's position and direction at every grid
    point; ``rank`` (N, L) says how important the road there is (higher goes first, ties go to the
    lower car index). Each step, every car looks for (1) a car ahead of it in its own lane, which it
    follows with the Intelligent Driver Model (a standard rule: brake harder the smaller the gap and
    the faster it closes), and (2) a car of higher priority whose path crosses or merges with its
    own within ``conflict_time`` seconds, which it treats as an obstacle until it has passed (a car
    that has stood still for ``patience`` seconds stops giving way, so a junction never locks up).
    Oncoming cars are ignored. The free-road speed is still ``profile``, so limits and bends apply
    as before.
    """
    n = profile.shape[0]
    if (
        profile.ndim != 2
        or min(len(s0), len(v0), len(lengths)) != n
        or xy.shape[:2] != profile.shape
    ):
        raise ValueError("profile, xy, lengths, s0 and v0 must have one row or entry per car")
    if min(accel, comfort_brake, max_brake, dt, time_gap, car_length) <= 0 or steps < 0:
        raise ValueError("accel, brakes, dt, time_gap and car_length must be positive")
    rows = np.arange(n)
    last = np.maximum((np.asarray(lengths) / ds).astype(int) - 1, 0)
    s = np.asarray(s0, dtype="float64").copy()
    v = np.asarray(v0, dtype="float64").copy()
    out_s = np.empty((n, steps + 1))
    out_v = np.empty((n, steps + 1))
    index = np.arange(n)
    higher = index[None, :] < index[:, None]  # [i, j]: j has the lower index, so wins a tie
    waited = np.zeros(n)  # seconds each car has been standing still
    for k in range(steps + 1):
        out_s[:, k], out_v[:, k] = s, v
        cell = np.minimum((s / ds).astype(int), np.maximum(last - 1, 0))
        w = np.clip(s / ds - cell, 0.0, 1.0)[:, None]
        nxt = np.minimum(cell + 1, last)
        pos = xy[rows, cell] * (1 - w) + xy[rows, nxt] * w
        head = heading[rows, cell] * (1 - w[:, 0]) + heading[rows, nxt] * w[:, 0]
        fwd = np.stack([np.cos(head), np.sin(head)], axis=1)
        vel = fwd * v[:, None]
        d = pos[None, :, :] - pos[:, None, :]  # [i, j] = from car i to car j
        along = np.einsum("ijk,ik->ij", d, fwd)
        across = np.abs(d[:, :, 0] * fwd[:, None, 1] - d[:, :, 1] * fwd[:, None, 0])
        cosine = np.cos(head[None, :] - head[:, None])
        in_lane = (cosine > 0.5) & (along > 0.0) & (along < horizon) & (across < 2.0 + 0.08 * along)
        in_lane &= ~((along > 0.0) & (along.T > 0.0)) | higher  # two cars cannot lead each other
        np.fill_diagonal(in_lane, False)
        gap = np.where(in_lane, along - car_length, np.inf)
        lead_v = np.broadcast_to(v[None, :], gap.shape)
        rel = vel[None, :, :] - vel[:, None, :]
        speed2 = np.einsum("ijk,ijk->ij", rel, rel) + 1e-9
        tca = np.clip(-np.einsum("ijk,ijk->ij", d, rel) / speed2, 0.0, conflict_time)
        closest = np.hypot(*(d + rel * tca[:, :, None]).transpose(2, 0, 1))
        crossing = (cosine > -0.87) & ~in_lane & (closest < conflict_radius) & (along > -1.0)
        mine, theirs = rank[rows, cell], rank[rows, cell][None, :]
        wins = (theirs > mine[:, None]) | ((theirs == mine[:, None]) & higher)
        yield_to = crossing & wins & (waited < patience)[:, None]  # no one waits for ever
        np.fill_diagonal(yield_to, False)
        gap_cross = np.where(
            yield_to, np.hypot(d[:, :, 0], d[:, :, 1]) - conflict_radius - stop_margin, np.inf
        )
        lead_v = np.where(yield_to & (gap_cross < gap), 0.0, lead_v)
        gap = np.minimum(gap, gap_cross)
        choose = np.argmin(gap, axis=1)
        g = np.maximum(gap[rows, choose], 0.05)
        dv = v - lead_v[rows, choose]
        wanted = np.maximum(profile[rows, np.minimum(cell + 1, last)], 0.1)
        free = 1.0 - (v / wanted) ** 4
        star = min_gap + v * time_gap + v * dv / (2.0 * np.sqrt(accel * comfort_brake))
        acc = accel * (
            free - np.where(np.isfinite(gap[rows, choose]), (np.maximum(star, 0.0) / g) ** 2, 0.0)
        )
        v = np.maximum(v + np.clip(acc, -max_brake, accel) * dt, 0.0)
        waited = np.where(v < 1.0, waited + dt, np.maximum(waited - dt, 0.0))
        s = np.minimum(s + v * dt, np.asarray(lengths) - 1e-6)
    return out_s, out_v


__all__ = ["braking_profile", "simulate", "simulate_interacting"]
