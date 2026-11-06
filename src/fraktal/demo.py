"""A one-file browser demo of a world tile (ADR-0011 phase 4/5, FRK-90).

``build`` writes ``index.html`` with everything embedded: the tile's roads, building footprints
and trees, a looping simulation of cars, people and drones, and a player the viewer steers with the
keyboard as a car, a person or a drone. It is a top-down 2D view of the same world and the same
agent simulations the renderer uses; the player's physics are the demo's own and nothing reacts to
the player. Opens offline from the file system, no server needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fraktal import agents
from fraktal.blender_export import export_scene

TEMPLATE = Path(__file__).with_name("demo_template.html")
COLOURS = ["#c9ced6", "#2f5fc4", "#b5402f", "#e2e0d8", "#2a2d33", "#3c8c5a", "#d9a227"]


def _tracks(t: agents.Tracks) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ids": t.ids,
        "x": np.round(t.x, 1).tolist(),
        "y": np.round(t.y, 1).tolist(),
        "h": np.round(t.heading, 2).tolist(),
    }
    if t.z is not None:
        out["z"] = np.round(t.z, 1).tolist()
    return out


def _start(graph: agents.Graph) -> list[float]:
    """The middle of the longest road, facing along it: a place the player can drive from."""
    edge = max(graph.edges, key=lambda e: e.length)
    k = len(edge.coords) // 2
    a, b = edge.coords[max(k - 1, 0)], edge.coords[min(k + 1, len(edge.coords) - 1)]
    p = edge.coords[k]
    return [
        round(float(p[0]), 1),
        round(float(p[1]), 1),
        round(float(np.arctan2(b[1] - a[1], b[0] - a[0])), 3),
    ]


def build(
    world_dir: Path,
    out_dir: Path,
    center_lonlat: tuple[float, float] | None = None,
    *,
    size_m: float = 500.0,
    seed: int = 0,
    duration_s: float = 60.0,
    fps: float = 10.0,
    cars_per_km: float = 12.0,
    people_per_km: float = 10.0,
    drones: int = 3,
) -> dict[str, Any]:
    """Write ``out_dir/index.html`` and return a summary."""
    if center_lonlat is None:
        center_lonlat = tuple(json.loads((world_dir / "world.json").read_text())["center_lonlat"])
    roads, bounds, side = agents.tile_roads(world_dir, center_lonlat, size_m)
    config = agents.TrafficConfig(cars_per_km=cars_per_km, drive_side=side)
    graph = agents.build_graph(roads, bounds, config)
    if not graph.edges:
        raise ValueError("no roads in this tile; pick another centre")
    times = np.arange(0.0, duration_s, 1.0 / fps)
    cars = agents.simulate_cars(graph, times, seed, config)
    people = agents.simulate_people(graph, times, seed, agents.WalkerConfig(people_per_km))
    flyers = agents.simulate_drones(size_m, times, seed, agents.DroneConfig(count=drones))
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / ".scene"
    scene = export_scene(
        world_dir, scratch, center_lonlat, size_m, 0.5, seed=seed, vehicle_density=0.0
    )
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()
    data = {
        "size": scene["size_m"],
        "drive_side": side,
        "fps": fps,
        "frames": len(times),
        "colours": COLOURS,
        "start": _start(graph),
        "roads": [
            {"pts": np.round(e.coords, 1).tolist(), "w": round(e.width_m, 1)} for e in graph.edges
        ],
        "buildings": [
            {"ring": [[round(x, 1), round(y, 1)] for x, y in b["ring"]], "h": b["height_m"]}
            for b in scene["buildings"]
        ],
        "trees": [[t[0], t[1], t[2]] for t in scene["trees"]],
        "cars": _tracks(cars),
        "people": _tracks(people),
        "drones": _tracks(flyers),
    }
    text = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (out_dir / "index.html").write_text(TEMPLATE.read_text().replace("__DATA__", text))
    return {
        "out": str(out_dir / "index.html"),
        "cars": len(cars.ids),
        "people": len(people.ids),
        "drones": len(flyers.ids),
        "buildings": len(data["buildings"]),
        "trees": len(data["trees"]),
        "roads": len(data["roads"]),
        "bytes": (out_dir / "index.html").stat().st_size,
    }


__all__ = ["build"]
