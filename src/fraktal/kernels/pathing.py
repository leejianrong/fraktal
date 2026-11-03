"""Least-cost paths over a cost grid (ADR-0005): arrays in, arrays out.

Dijkstra over eight neighbours. A cell costs ``cost[r, c]`` to enter (multiplied by sqrt(2) for a
diagonal step); non-finite or non-positive costs are impassable. Ties break by cell index, so the
same inputs always give the same path.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True)
def _dijkstra(cost: np.ndarray[Any, Any], start: int, goal: int) -> np.ndarray[Any, Any]:
    rows, cols = cost.shape
    n = rows * cols
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=np.int64)
    dist[start] = 0.0
    heap = [(0.0, start)]
    dr = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
    dc = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    while len(heap) > 0:
        d, cell = heapq.heappop(heap)
        if cell == goal:
            break
        if d > dist[cell]:
            continue
        r, c = cell // cols, cell % cols
        for k in range(8):
            rr, cc = r + dr[k], c + dc[k]
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            step = cost[rr, cc]
            if not (step > 0.0 and step < 1e300):
                continue
            if dr[k] != 0 and dc[k] != 0:
                step *= math.sqrt(2.0)
            nxt = rr * cols + cc
            nd = d + step
            if nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = cell
                heapq.heappush(heap, (nd, nxt))
    return prev


def least_cost_path(
    cost: np.ndarray[Any, Any], start: tuple[int, int], goal: tuple[int, int]
) -> np.ndarray[Any, Any] | None:
    """Cells from ``start`` to ``goal`` inclusive as an (N, 2) array of (row, column), or None
    when the goal cannot be reached."""
    if cost.ndim != 2:
        raise ValueError("cost must be 2D")
    rows, cols = cost.shape
    for name, (r, c) in (("start", start), ("goal", goal)):
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"{name} lies outside the grid")
    s, g = start[0] * cols + start[1], goal[0] * cols + goal[1]
    prev = _dijkstra(np.ascontiguousarray(cost, dtype=np.float64), s, g)
    if s != g and prev[g] < 0:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(int(prev[cells[-1]]))
    flat = np.array(cells[::-1], dtype=np.int64)
    out: np.ndarray[Any, Any] = np.column_stack([flat // cols, flat % cols])
    return out
