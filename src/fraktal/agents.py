"""A road graph and moving cars on it (ADR-0011 phase 2, FRK-81).

``build_graph`` nodes the world's road lines where they meet and gives every road two directed
lanes, offset to the side cars drive on (``drive_side``, a biome setting). ``simulate_cars`` picks
seeded routes through the graph, flies a car along each at the class speed limit (slower in bends),
and returns ``Tracks``: position, heading and speed of every car at every frame time. Everything is
in tile coordinates (metres east and north of the tile's south-west corner), the same frame as the
scene export. Randomness only through ``fraktal.ids.substream``, so the same inputs give the same
tracks.

Cars follow the car ahead in their lane and give way where paths cross (``TrafficConfig.interact``;
the kernel's docstring says how). Bridges are left out of the graph for now: the ground mesh follows
the terrain, so a car on a bridge would sit at water level.
"""

from __future__ import annotations

import colorsys
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point, box
from shapely.ops import substring

from fraktal.ids import substream
from fraktal.kernels import flight, traffic

DRIVE_SIDES = ("left", "right")
GRID_M = 1.0  # routes are sampled every metre
SIM_DT = 0.1  # seconds
MAX_ROUTE_EDGES = 400


class AgentError(ValueError):
    pass


@dataclass(frozen=True)
class TrafficConfig:
    cars_per_km: float = 6.0  # per kilometre of road, both directions together
    drive_side: str = "right"
    arterial_speed_mps: float = 11.0
    local_speed_mps: float = 6.0
    speed_spread: float = 0.15  # each car's own speed is the limit times 1 +- this
    max_accel: float = 2.5
    max_brake: float = 3.5  # hardest braking, and the limit of the speed plan
    comfort_brake: float = 2.5  # what car following aims for
    interact: bool = True  # cars follow the car ahead and give way at crossings
    max_lateral_accel: float = 2.0  # sets the speed through a bend
    min_spawn_gap_m: float = 15.0
    snap_m: float = 1.0  # line ends and crossings closer than this are one junction
    min_edge_m: float = 3.0  # shorter pieces of road are dropped


@dataclass
class Edge:
    u: int
    v: int
    coords: np.ndarray[Any, Any]  # (K, 2), from u to v
    road_class: str
    width_m: float

    @property
    def length(self) -> float:
        return float(np.hypot(*np.diff(self.coords, axis=0).T).sum())


@dataclass
class Graph:
    nodes: np.ndarray[Any, Any]  # (M, 2)
    edges: list[Edge]
    out: list[list[int]]  # per node, the half-edges leaving it; half-edge 2e runs u to v, 2e+1 back
    bounds: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )  # the tile the roads were cut to

    def on_boundary(self, node: int, margin: float = 3.0) -> bool:
        """Whether a junction sits on the tile edge, where the road really goes on."""
        x, y = self.nodes[node]
        x0, y0, x1, y1 = self.bounds
        return bool(min(x - x0, x1 - x, y - y0, y1 - y) <= margin)

    def tail(self, half: int) -> int:
        e = self.edges[half >> 1]
        return e.u if half % 2 == 0 else e.v

    def head(self, half: int) -> int:
        e = self.edges[half >> 1]
        return e.v if half % 2 == 0 else e.u

    def total_length_m(self) -> float:
        return sum(e.length for e in self.edges)


@dataclass(frozen=True)
class Tracks:
    """Where every car is at every frame time; heading is radians anticlockwise from east."""

    ids: list[str]
    t: np.ndarray[Any, Any]  # (T,)
    x: np.ndarray[Any, Any]  # (N, T)
    y: np.ndarray[Any, Any]
    heading: np.ndarray[Any, Any]
    speed: np.ndarray[Any, Any]
    kind: str = "car"
    z: np.ndarray[Any, Any] | None = None  # (N, T) height above the ground, for flying agents

    def as_dict(self) -> dict[str, Any]:
        extra = {"z": np.round(self.z, 3).tolist()} if self.z is not None else {}
        return {
            **extra,
            "kind": self.kind,
            "ids": self.ids,
            "t": np.round(self.t, 4).tolist(),
            "x": np.round(self.x, 3).tolist(),
            "y": np.round(self.y, 3).tolist(),
            "heading": np.round(self.heading, 4).tolist(),
            "speed": np.round(self.speed, 3).tolist(),
        }


def tile_roads(
    world_dir: Any, center_lonlat: tuple[float, float], size_m: float, gsd_m: float = 0.5
) -> tuple[Any, tuple[float, float, float, float], str]:
    """The world's roads in the export's tile frame, the tile bounds and the world's drive side."""
    from shapely.affinity import translate

    from fraktal import procedural
    from fraktal.render import RenderConfig, region_grid

    px = max(1, round(size_m / gsd_m))
    epsg, transform, (_, side) = region_grid(
        center_lonlat, RenderConfig(gsd_m=gsd_m, tile_px=px, size_m=size_m)
    )
    west, north = transform.c, transform.f
    south = north - side * gsd_m
    roads = procedural.read_roads(world_dir).to_crs(epsg)
    roads["geometry"] = [translate(g, -west, -south) for g in roads.geometry]
    manifest = json.loads((Path(world_dir) / procedural.MANIFEST).read_text())
    extent = side * gsd_m
    return roads, (0.0, 0.0, extent, extent), str(manifest.get("drive_side", "right"))


def draw_tracks(graph: Graph, tracks: Tracks, size_m: float, path: Path, px: int = 1000) -> None:
    """A map of the roads and every car's path (a dot where it starts), for looking at."""
    from PIL import Image, ImageDraw

    scale = px / size_m

    def pt(x: float, y: float) -> tuple[float, float]:
        return float(x * scale), float(px - y * scale)

    image = Image.new("RGB", (px, px), (30, 32, 36))
    draw = ImageDraw.Draw(image)
    for edge in graph.edges:
        w = max(1, round(edge.width_m * scale))
        draw.line([pt(x, y) for x, y in edge.coords], fill=(78, 80, 86), width=w)
    for k in range(len(tracks.ids)):
        hue = (k * 47) % 360
        colour = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 0.7, 1.0))
        draw.line([pt(x, y) for x, y in zip(tracks.x[k], tracks.y[k], strict=True)],
                  fill=colour, width=2)  # fmt: skip
        cx, cy = pt(tracks.x[k][0], tracks.y[k][0])
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=colour)
    image.save(path)


def _check_side(side: str) -> None:
    if side not in DRIVE_SIDES:
        raise AgentError(f"drive_side must be one of {DRIVE_SIDES}, got {side!r}")


def _line_parts(geom: Any) -> list[LineString]:
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    return [p for g in getattr(geom, "geoms", []) for p in _line_parts(g)]


def _cut_points(line: LineString, others: list[LineString], snap: float) -> list[float]:
    """Distances along ``line`` where it crosses or is met by another line."""
    cuts = [0.0, line.length]
    for other in others:
        inter = line.intersection(other)
        points: list[Any] = []
        for g in getattr(inter, "geoms", [inter]):
            if g.is_empty:
                continue
            points += [Point(c) for c in g.coords] if g.geom_type != "Point" else [g]
        for end in (Point(other.coords[0]), Point(other.coords[-1])):
            if line.distance(end) <= snap:
                points.append(end)
        cuts += [line.project(p) for p in points]
    return sorted(cuts)


def build_graph(
    roads: Any, bounds: tuple[float, float, float, float], config: TrafficConfig | None = None
) -> Graph:
    """The junction graph of ``roads`` (a layer in tile coordinates) clipped to ``bounds``."""
    config = config or TrafficConfig()
    clip = box(*bounds)
    lines: list[LineString] = []
    meta: list[tuple[str, float]] = []
    for _, row in roads.iterrows():
        if float(row.get("bridge", 0.0) or 0.0) > 0 or row["road_class"] == "footpath":
            continue  # bridges are left out for now, and footpaths are too narrow for a car
        for part in _line_parts(row.geometry.intersection(clip)):
            if part.length >= config.min_edge_m:
                lines.append(part)
                meta.append((str(row["road_class"]), float(row["width_m"])))
    tree = STRtree(lines)
    nodes: list[tuple[float, float]] = []
    cells: dict[tuple[int, int], list[int]] = {}

    def node_for(x: float, y: float) -> int:
        cx, cy = int(x // config.snap_m), int(y // config.snap_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in cells.get((cx + dx, cy + dy), []):
                    if math.hypot(nodes[k][0] - x, nodes[k][1] - y) <= config.snap_m:
                        return k
        nodes.append((x, y))
        cells.setdefault((cx, cy), []).append(len(nodes) - 1)
        return len(nodes) - 1

    edges: list[Edge] = []
    for i, line in enumerate(lines):
        near = [lines[j] for j in tree.query(line.buffer(config.snap_m)) if j != i]
        cuts = _cut_points(line, near, config.snap_m)
        kept = [cuts[0]]
        for c in cuts[1:]:
            if c - kept[-1] >= 0.5:
                kept.append(c)
        if kept[-1] != line.length:
            kept[-1] = line.length
        for a, b in zip(kept, kept[1:], strict=False):
            piece = substring(line, a, b)
            xy = np.asarray(piece.coords, dtype="float64")
            if len(xy) < 2:
                continue
            u, v = node_for(*xy[0]), node_for(*xy[-1])
            if u == v and piece.length < 20.0:
                continue
            xy[0], xy[-1] = nodes[u], nodes[v]
            edges.append(Edge(u, v, xy, *meta[i]))
    out: list[list[int]] = [[] for _ in nodes]
    for e, edge in enumerate(edges):
        out[edge.u].append(2 * e)
        out[edge.v].append(2 * e + 1)
    return Graph(np.asarray(nodes, dtype="float64").reshape(-1, 2), edges, out, bounds)


def _offset(xy: np.ndarray[Any, Any], distance: float) -> np.ndarray[Any, Any]:
    """``xy`` shifted sideways by ``distance`` metres (positive to the left of travel)."""
    try:
        parts = _line_parts(LineString(xy).offset_curve(distance))
        if not parts:
            return np.asarray(xy)
        out = np.asarray(max(parts, key=lambda p: p.length).coords, dtype="float64")
    except Exception:  # noqa: BLE001 - a degenerate piece keeps its centre line
        return np.asarray(xy)
    if np.hypot(*(out[0] - xy[0])) > np.hypot(*(out[-1] - xy[0])):
        out = out[::-1]  # the offset came out reversed
    return out


def lane(graph: Graph, half: int, side: str) -> np.ndarray[Any, Any]:
    """The centre line of half-edge ``half``'s lane: the road shifted a quarter of its width."""
    edge = graph.edges[half >> 1]
    xy = edge.coords if half % 2 == 0 else edge.coords[::-1]
    return _offset(xy, (-1.0 if side == "right" else 1.0) * edge.width_m / 4.0)


def verge(graph: Graph, half: int, left: bool, margin_m: float) -> np.ndarray[Any, Any]:
    """A footpath beside half-edge ``half``, ``margin_m`` outside the road edge (left or right)."""
    edge = graph.edges[half >> 1]
    xy = edge.coords if half % 2 == 0 else edge.coords[::-1]
    return _offset(xy, (1.0 if left else -1.0) * (edge.width_m / 2.0 + margin_m))


def _chaikin(points: np.ndarray[Any, Any], rounds: int) -> np.ndarray[Any, Any]:
    for _ in range(rounds):
        a, b = points[:-1], points[1:]
        mid = np.empty((2 * len(a), 2))
        mid[0::2], mid[1::2] = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
        points = np.vstack([points[:1], mid, points[-1:]])
    return points


def _route(
    graph: Graph, start: int, need_m: float, rng: Any, config: TrafficConfig
) -> tuple[list[int], bool]:
    """Half-edges to follow from ``start`` until the route is ``need_m`` long.

    At a junction the next road is picked by seed, never straight back, with arterials favoured. At
    a dead end inside the tile the car turns round; at a dead end on the tile edge the road goes
    on, so the route ends there (the second value is true) and the car drives off the tile.
    """
    route, total = [start], graph.edges[start >> 1].length
    while total < need_m and len(route) < MAX_ROUTE_EDGES:
        options = [h for h in graph.out[graph.head(route[-1])] if h != route[-1] ^ 1]
        if not options:
            if graph.on_boundary(graph.head(route[-1])):
                return route, True
            options = [route[-1] ^ 1]
        weights = np.array(
            [2.0 if graph.edges[h >> 1].road_class == "arterial" else 1.0 for h in options]
        )
        pick = options[int(rng.choice(len(options), p=weights / weights.sum()))]
        route.append(pick)
        total += graph.edges[pick >> 1].length
    return route, False


@dataclass
class _Path:
    s: np.ndarray[Any, Any]  # arc length of each sample, 1 m apart
    xy: np.ndarray[Any, Any]
    heading: np.ndarray[Any, Any]  # unwrapped, radians anticlockwise from east
    cap: np.ndarray[Any, Any]  # speed cap at each sample
    rank: np.ndarray[Any, Any]  # 1 on an arterial, 0 on a local road: who goes first at a crossing


def _path(
    graph: Graph, route: list[int], config: TrafficConfig, factor: float, leave_m: float = 0.0
) -> _Path:
    pieces = [lane(graph, h, config.drive_side) for h in route]
    limits = [
        config.arterial_speed_mps if graph.edges[h >> 1].road_class == "arterial"
        else config.local_speed_mps
        for h in route
    ]  # fmt: skip
    raw = np.vstack(pieces)
    limit_at_vertex = np.repeat(limits, [len(p) for p in pieces])
    rank_at_vertex = np.repeat(
        [1.0 if graph.edges[h >> 1].road_class == "arterial" else 0.0 for h in route],
        [len(p) for p in pieces],
    )
    if leave_m > 0:  # the road goes on past the tile edge: carry straight on
        away = raw[-1] - raw[-2]
        raw = np.vstack([raw, raw[-1] + away / np.hypot(*away) * leave_m])
        limit_at_vertex = np.append(limit_at_vertex, limit_at_vertex[-1])
        rank_at_vertex = np.append(rank_at_vertex, rank_at_vertex[-1])
    seg = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(raw, axis=0).T))])
    smooth = _chaikin(raw, 3)
    seg_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(smooth, axis=0).T))])
    s = np.arange(0.0, seg_s[-1], GRID_M)
    xy = np.stack([np.interp(s, seg_s, smooth[:, k]) for k in range(2)], axis=1)
    d = np.gradient(xy, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    window = 3
    turn = np.abs(heading[window:] - heading[:-window]) / (window * GRID_M)
    kappa = np.concatenate([turn, np.full(window, turn[-1] if len(turn) else 0.0)])
    padded = np.pad(kappa, 2, mode="edge")
    kappa = np.max([padded[k : k + len(kappa)] for k in range(5)], axis=0)  # a bend's neighbours
    bend = np.sqrt(config.max_lateral_accel / np.maximum(kappa, 1e-4))
    vertex = np.searchsorted(seg, s * seg[-1] / seg_s[-1])  # the smoothed path is a little shorter
    base = limit_at_vertex[np.minimum(vertex, len(limit_at_vertex) - 1)] * factor
    rank = rank_at_vertex[np.minimum(vertex, len(rank_at_vertex) - 1)]
    return _Path(s, xy, heading, np.minimum(base, bend), rank)


@dataclass(frozen=True)
class WalkerConfig:
    people_per_km: float = 10.0  # per kilometre of road, both sides together
    walk_speed_mps: float = 1.4
    speed_spread: float = 0.2  # each person's pace is the walking speed times 1 +- this
    verge_m: float = 1.2  # how far outside the road edge people walk
    stops_per_min: float = 2.0  # how often someone stops (to chat, look, wait)
    stop_s: tuple[float, float] = (2.0, 8.0)  # how long a stop lasts


def simulate_people(
    graph: Graph, times: Any, seed: int, config: WalkerConfig | None = None
) -> Tracks:
    """People walking on the verges of ``graph``'s roads at each of ``times``.

    Each person picks a verge side and a seeded route through the junctions (the straight link
    across a junction is walked as it comes), at a pace of their own with random stops. People do
    not react to cars and cars do not react to people (yet).
    """
    config = config or WalkerConfig()
    t = np.asarray(times, dtype="float64")
    if t.ndim != 1 or len(t) < 1 or np.any(np.diff(t) < 0):
        raise AgentError("times must be a non-empty ascending 1D array")
    if config.people_per_km < 0 or config.walk_speed_mps <= 0:
        raise AgentError("people_per_km must not be negative and walk_speed_mps must be positive")
    n = int(round(config.people_per_km * graph.total_length_m() / 1000.0))
    empty = np.zeros((0, len(t)))
    if n == 0 or not graph.edges:
        return Tracks([], t, empty, empty, empty, empty, kind="person")
    rng = substream(seed, "pedestrians")
    duration = float(t[-1] - t[0])
    steps = int(math.ceil(duration / SIM_DT))
    lengths = np.array([e.length for e in graph.edges])
    reach = config.walk_speed_mps * (1.0 + config.speed_spread) * duration + 30.0
    walkers: list[tuple[_Path, float, float]] = []
    walking = np.ones((n, steps), dtype=bool)
    for k in range(n):
        e = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
        half = 2 * e + int(rng.integers(0, 2))
        along = float(rng.uniform(0.0, lengths[e]))
        along = along if half % 2 == 0 else lengths[e] - along
        left = bool(rng.integers(0, 2))
        route, leaves = _route(graph, half, along + reach, rng, TrafficConfig())
        raw = np.vstack([verge(graph, h, left, config.verge_m) for h in route])
        if leaves:
            away = raw[-1] - raw[-2]
            raw = np.vstack([raw, raw[-1] + away / np.hypot(*away) * reach])
        smooth = _chaikin(raw, 3)
        seg_s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(smooth, axis=0).T))])
        s = np.arange(0.0, seg_s[-1], GRID_M)
        xy = np.stack([np.interp(s, seg_s, smooth[:, c]) for c in range(2)], axis=1)
        heading = np.unwrap(np.arctan2(*np.gradient(xy, axis=0).T[::-1]))
        pace = config.walk_speed_mps * (1.0 + config.speed_spread * float(rng.uniform(-1.0, 1.0)))
        walkers.append(
            (_Path(s, xy, heading, np.zeros(0), np.zeros(0)), pace, min(along, s[-1] - 1.0))
        )
        for _ in range(int(rng.poisson(config.stops_per_min * duration / 60.0))):
            start = int(rng.uniform(0.0, duration) / SIM_DT)
            length = int(rng.uniform(*config.stop_s) / SIM_DT)
            walking[k, start : start + length] = False
    clock = (t - t[0]) / SIM_DT
    lo = np.minimum(np.floor(clock).astype(int), steps - 1) if steps else np.zeros(len(t), int)
    x, y, heading_out, speed = (np.empty((n, len(t))) for _ in range(4))
    done = np.concatenate([np.zeros((n, 1)), np.cumsum(walking, axis=1)], axis=1) * SIM_DT
    for k, (path, pace, s0) in enumerate(walkers):
        travelled = s0 + pace * done[k, np.minimum(np.round(clock).astype(int), steps)]
        x[k] = np.interp(travelled, path.s, path.xy[:, 0])
        y[k] = np.interp(travelled, path.s, path.xy[:, 1])
        heading_out[k] = np.interp(travelled, path.s, path.heading)
        speed[k] = np.where(walking[k, lo] if steps else True, pace, 0.0)
    ids = [f"person-{k:04d}" for k in range(n)]
    wrapped = (heading_out + math.pi) % (2 * math.pi) - math.pi
    return Tracks(ids, t, x, y, wrapped, speed, kind="person")


@dataclass(frozen=True)
class DroneConfig:
    count: int = 3
    altitude_m: tuple[float, float] = (40.0, 120.0)  # above the ground, one level per drone
    speed_mps: float = 12.0
    max_accel: float = 3.0
    max_yaw_rate_dps: float = 90.0
    margin: float = 0.1  # waypoints stay this share of the tile side inside the tile


def simulate_drones(
    size_m: float, times: Any, seed: int, config: DroneConfig | None = None
) -> Tracks:
    """Drones flying seeded routes over the tile, at ``times``.

    Each drone starts at a random point and flies through random waypoints with the same
    point-mass model as the camera (``fraktal.kernels.flight``), at its own height. They do not
    avoid each other or the camera (they fly at different heights, so a crossing is a near miss).
    """
    config = config or DroneConfig()
    t = np.asarray(times, dtype="float64")
    if t.ndim != 1 or len(t) < 1 or np.any(np.diff(t) < 0):
        raise AgentError("times must be a non-empty ascending 1D array")
    if config.count < 0 or config.speed_mps <= 0:
        raise AgentError("drone count must not be negative and speed must be positive")
    n = config.count
    empty = np.zeros((0, len(t)))
    if n == 0:
        return Tracks([], t, empty, empty, empty, empty, kind="drone", z=empty)
    rng = substream(seed, "drones")
    lo, hi = config.margin * size_m, (1.0 - config.margin) * size_m
    reach = config.speed_mps * float(t[-1] - t[0]) + 20.0
    x, y, heading, speed, z = (np.empty((n, len(t))) for _ in range(5))
    for k in range(n):
        pts = [rng.uniform(lo, hi, 2)]
        length = 0.0
        while length < reach and len(pts) < 12:
            nxt = rng.uniform(lo, hi, 2)
            if float(np.hypot(*(nxt - pts[-1]))) < 30.0:
                continue
            length += float(np.hypot(*(nxt - pts[-1])))
            pts.append(nxt)
        if len(pts) < 2:  # the tile is too small for a leg of 30 m
            raise AgentError("the tile is too small for drone routes")
        run = flight.fly(
            np.array(pts), cruise_mps=config.speed_mps * (0.8 + 0.4 * float(rng.uniform())),
            max_accel=config.max_accel, max_yaw_rate_dps=config.max_yaw_rate_dps, fps=10.0,
        )  # fmt: skip
        clock = np.clip(t - t[0], 0.0, run["t"][-1])  # hovers at the end if the route is short
        for out, col in ((x, run["xy"][:, 0]), (y, run["xy"][:, 1]), (speed, run["speed"])):
            out[k] = np.interp(clock, run["t"], col)
        yaw = np.unwrap(np.radians(run["yaw_deg"]))
        heading[k] = np.pi / 2 - np.interp(clock, run["t"], yaw)
        z[k] = float(rng.uniform(*config.altitude_m))
    wrapped = (heading + math.pi) % (2 * math.pi) - math.pi
    ids = [f"drone-{k:04d}" for k in range(n)]
    return Tracks(ids, t, x, y, wrapped, speed, kind="drone", z=z)


def simulate_cars(
    graph: Graph, times: Any, seed: int, config: TrafficConfig | None = None
) -> Tracks:
    """Cars on ``graph`` at each of ``times`` (seconds, ascending, the first is the start)."""
    config = config or TrafficConfig()
    _check_side(config.drive_side)
    t = np.asarray(times, dtype="float64")
    if t.ndim != 1 or len(t) < 1 or np.any(np.diff(t) < 0):
        raise AgentError("times must be a non-empty ascending 1D array")
    if config.cars_per_km < 0:
        raise AgentError("cars_per_km must not be negative")
    n = int(round(config.cars_per_km * graph.total_length_m() / 1000.0))
    empty = np.zeros((0, len(t)))
    if n == 0 or not graph.edges:
        return Tracks([], t, empty, empty, empty, empty)
    rng = substream(seed, "traffic")
    duration = float(t[-1] - t[0])
    lengths = np.array([e.length for e in graph.edges])
    starts: list[tuple[int, float]] = []
    for _ in range(n):
        for _try in range(30):
            e = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
            half = 2 * e + int(rng.integers(0, 2))
            offset = float(rng.uniform(0.0, lengths[e]))
            along = (
                offset if half % 2 == 0 else lengths[e] - offset
            )  # distance from the half-edge start
            near = [a for h, a in starts if h == half and abs(a - along) < config.min_spawn_gap_m]
            if not near:
                break
        starts.append((half, along))
    paths: list[_Path] = []
    s0, v0 = [], []
    top = max(config.arterial_speed_mps, config.local_speed_mps) * (1.0 + config.speed_spread)
    for half, along in starts:
        factor = 1.0 + config.speed_spread * float(rng.uniform(-1.0, 1.0))
        need = along + top * duration + 50.0
        route, leaves = _route(graph, half, need, rng, config)
        path = _path(graph, route, config, factor, need if leaves else 0.0)
        paths.append(path)
        s0.append(min(along, path.s[-1] - 1.0))
        v0.append(
            float(path.cap[min(int(along), len(path.cap) - 1)]) * float(rng.uniform(0.7, 1.0))
        )
    width = max(len(p.s) for p in paths)
    cap = np.zeros((n, width))
    for k, p in enumerate(paths):
        cap[k, : len(p.cap)] = p.cap
    profile = traffic.braking_profile(cap, GRID_M, config.max_brake * 0.8)
    lens = np.array([p.s[-1] + GRID_M for p in paths])
    steps = int(math.ceil(duration / SIM_DT))
    if config.interact:
        xy_all = np.zeros((n, width, 2))
        head_all = np.zeros((n, width))
        rank_all = np.zeros((n, width))
        for k, p in enumerate(paths):  # past a route's end, hold its last point
            m = len(p.s)
            xy_all[k, :m], xy_all[k, m:] = p.xy, p.xy[-1]
            head_all[k, :m], head_all[k, m:] = p.heading, p.heading[-1]
            rank_all[k, :m], rank_all[k, m:] = p.rank, p.rank[-1]
        s_all, v_all = traffic.simulate_interacting(
            profile, xy_all, head_all, rank_all, lens, GRID_M, np.array(s0), np.array(v0),
            accel=config.max_accel, comfort_brake=config.comfort_brake,
            max_brake=config.max_brake * 2.0, dt=SIM_DT, steps=steps,
        )  # fmt: skip
    else:
        s_all, v_all = traffic.simulate(
            profile, lens, GRID_M, np.array(s0), np.array(v0),
            accel=config.max_accel, brake=config.max_brake, dt=SIM_DT, steps=steps,
        )  # fmt: skip
    clock = (t - t[0]) / SIM_DT
    lo = np.minimum(np.floor(clock).astype(int), steps)
    hi = np.minimum(lo + 1, steps)
    frac = clock - lo
    s_t = s_all[:, lo] * (1 - frac) + s_all[:, hi] * frac
    v_t = v_all[:, lo] * (1 - frac) + v_all[:, hi] * frac
    x, y, heading = (np.empty((n, len(t))) for _ in range(3))
    for k, p in enumerate(paths):
        x[k] = np.interp(s_t[k], p.s, p.xy[:, 0])
        y[k] = np.interp(s_t[k], p.s, p.xy[:, 1])
        heading[k] = np.interp(s_t[k], p.s, p.heading)
    ids = [f"car-{k:04d}" for k in range(n)]
    return Tracks(ids, t, x, y, (heading + math.pi) % (2 * math.pi) - math.pi, v_t)


__all__ = [
    "AgentError", "Graph", "Tracks", "TrafficConfig", "WalkerConfig", "build_graph", "lane",
    "simulate_cars", "simulate_people", "verge", "draw_tracks", "tile_roads",
]  # fmt: skip
