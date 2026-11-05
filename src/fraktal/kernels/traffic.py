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


__all__ = ["braking_profile", "simulate"]
