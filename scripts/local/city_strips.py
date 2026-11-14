#!/usr/bin/env python
"""Render eye-level street strips for city biomes on this machine, the local twin of
scripts/runpod/city_strips.sh. Same world, route and render settings, so strips are comparable.

Usage: scripts/local/city_strips.py <tag> <biome>... [--samples 32] [--device gpu|cpu]
  e.g. scripts/local/city_strips.py 20 manila_quezon nairobi_centre khartoum_city
Output: data/review/street<tag>_<biome>/out/review/fly/frames/*.png (the layout the pod script
unpacks to) and commit.txt with the HEAD and whether the tree was dirty. Compare with:
  fraktal streets <frames dir> --city '<glob>' --data-dir data --limit 30 --segment --sheet <png>
On WSL, FRAKTAL_BPY_PYTHON=scripts/wsl/winblender.sh runs the Windows Blender (docs/HANDOVER.md).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from shapely.geometry import box
from shapely.ops import substring

from fraktal import agents, biomes, procedural
from fraktal.procedural import generate_world

ROOT = Path(__file__).resolve().parents[2]
CENTER0 = (32.55, 15.64)  # the pods place every biome's world here


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def make_world_and_route(
    work: Path, biome: str, length_m: float, overrides: dict[str, object]
) -> tuple[tuple[float, float], str]:
    config = biomes.resolve(biome, {"size_m": 3000.0, **overrides})
    generate_world(work / "world", CENTER0, 3, config)
    roads = procedural.read_roads(work / "world").to_crs(4326)
    weight = roads.length
    centroids = roads.geometry.centroid
    center = (
        float((centroids.x * weight).sum() / weight.sum()),
        float((centroids.y * weight).sum() / weight.sum()),
    )
    local, _bounds, side = agents.tile_roads(work / "world", center, 800.0)
    inner = box(160, 160, 640, 640)
    pieces = [
        p
        for g in local.geometry
        for p in getattr(g.intersection(inner), "geoms", [g.intersection(inner)])
        if p.geom_type == "LineString"
    ]
    best = substring(max(pieces, key=lambda p: p.length), 0, length_m).simplify(10.0)
    route = [[x, y] for x, y in best.coords]
    (work / "path.json").write_text(json.dumps({"waypoints": route}))
    (work / "center.txt").write_text(f"{center[0]:f} {center[1]:f}")
    return center, side


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("biomes", nargs="+")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--speed", type=float, default=8.0, help="metres between frames at 1 fps")
    ap.add_argument("--length-m", type=float, default=420.0)
    ap.add_argument(
        "--override",
        default="{}",
        help='JSON of biome settings for every biome, e.g. {"buildings": {"front_setback_m": 6}}',
    )
    ap.add_argument("--people-per-km", type=float, default=30.0)
    ap.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("FRAKTAL_DATA_DIR", ROOT / "data"))
    )
    args = ap.parse_args()

    os.environ.setdefault("FRAKTAL_ASSETS", str(ROOT / "assets"))
    commit = git("rev-parse", "HEAD") + (" dirty" if git("status", "--porcelain") else "")
    if args.override != "{}":
        commit += f" override={args.override}"
    progress = args.data_dir / "review" / f"cities{args.tag}_progress.txt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    for biome in args.biomes:
        out = args.data_dir / "review" / f"street{args.tag}_{biome}"
        work = out / "work"
        fly = out / "out" / "review" / "fly"
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        center, side = make_world_and_route(work, biome, args.length_m, json.loads(args.override))
        cmd = [
            "fraktal", "render3d", str(work / "world"), "--out", str(fly),
            "--center", f"{center[0]} {center[1]}", "--size-m", "800", "--gsd-m", "0.5",
            "--path", str(work / "path.json"), "--fov", "75", "--frame-px", "512",
            "--trajectory", "street", "--eye-m", "1.6", "--speed", str(args.speed), "--fps", "1",
            "--traffic", "--cars-per-km", "25", "--people-per-km", str(args.people_per_km),
            "--samples", str(args.samples), "--seed", "7", "--device", args.device,
        ]  # fmt: skip
        # --center takes two numbers, so split it into two argv entries
        i = cmd.index("--center")
        cmd[i + 1 : i + 2] = cmd[i + 1].split()
        with (out / "render.err").open("w") as err:
            done = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err)
        frames = len(list((fly / "frames").glob("*.png")))
        (out / "commit.txt").write_text(f"{commit}\n")
        secs = time.time() - t0
        line = f"{biome} exit={done.returncode} frames={frames} seconds={secs:.0f} commit={commit}"
        print(line, flush=True)
        with progress.open("a") as fh:
            fh.write(line + "\n")
    with progress.open("a") as fh:
        fh.write("ALL_DONE\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
