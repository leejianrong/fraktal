"""Stage 0 (ADR-0010): the same tile in the 2.5D renderer and in Blender, scored against real chips.

Usage: python stage0_compare.py WORK_DIR [--size-m 500] [--gsd-m 0.5] [--samples 32] [--threads N]
       [--bpy-python PATH] [--real DIR ...]

Generates one dense_urban world, exports the tile (``fraktal.blender_export``), renders it with
``stage0.py`` (run with ``--bpy-python``, a Python that has ``bpy``), renders the same tile with
the 2.5D renderer, cuts both into 500 px chips, and writes ``stage0.json`` (timings and gauge
distances against each real chip directory) and ``stage0.png`` (a side-by-side strip).
Real chips are evaluation references only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

from fraktal import ablation, biomes, chips, palette, realism
from fraktal.blender_export import export_scene
from fraktal.kernels.sensor import sensor_chain
from fraktal.procedural import generate_world, read_buildings, read_fields, read_roads
from fraktal.render import RenderConfig, render_region
from fraktal.sensor import add_noise

CENTER = (32.55, 15.64)
CHIP = 500


def busiest_centre(world: Path, size_m: float) -> tuple[float, float]:
    """Lon/lat of the centre of the ``size_m`` window holding the most buildings."""
    from pyproj import Transformer

    from fraktal.render import utm_epsg

    epsg = utm_epsg(*CENTER)
    xy = np.array([(c.x, c.y) for c in read_buildings(world).to_crs(epsg).geometry.centroid])
    step = 50.0
    lo = xy.min(0)
    n = int(size_m / step)
    grid = np.zeros((int(np.ptp(xy[:, 0]) / step) + 2, int(np.ptp(xy[:, 1]) / step) + 2))
    for x, y in ((xy - lo) // step).astype(int):
        grid[x, y] += 1
    sums = np.array(
        [
            [grid[i : i + n, j : j + n].sum() for j in range(grid.shape[1] - n)]
            for i in range(grid.shape[0] - n)
        ]
    )
    i, j = np.unravel_index(sums.argmax(), sums.shape)
    cx, cy = lo[0] + (i + n / 2) * step, lo[1] + (j + n / 2) * step
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(cx, cy)
    return float(lon), float(lat)


def cut(image: np.ndarray, n: int = CHIP) -> list[np.ndarray]:
    """Non-overlapping n x n chips of a (3, H, W) image."""
    _, h, w = image.shape
    return [
        image[:, r : r + n, c : c + n]
        for r in range(0, h - n + 1, n)
        for c in range(0, w - n + 1, n)
    ]


def read_png_rgb(path: Path) -> np.ndarray:
    """The 8-bit RGB values of a PNG: with the Raw view transform these are the render itself."""
    with rasterio.open(path) as src:
        return src.read()[:3].astype("uint8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("--size-m", type=float, default=500.0)
    p.add_argument("--gsd-m", type=float, default=0.5)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--bpy-python", default=sys.executable)
    p.add_argument("--palette", type=Path)
    p.add_argument("--real", type=Path, action="append", default=[])
    p.add_argument("--biome", default="dense_urban")
    a = p.parse_args()
    work: Path = a.work
    work.mkdir(parents=True, exist_ok=True)
    world = work / "world"
    if not world.exists():
        generate_world(world, CENTER, 3, biomes.resolve(a.biome))
    centre = busiest_centre(world, a.size_m)
    result: dict = {
        "centre_lonlat": centre,
        "biome": a.biome,
        "size_m": a.size_m,
        "gsd_m": a.gsd_m,
        "samples": a.samples,
    }

    t = time.time()
    export_scene(world, work / "scene", centre, a.size_m, a.gsd_m)
    result["export_s"] = round(time.time() - t, 2)

    t = time.time()
    cmd = [
        a.bpy_python,
        str(Path(__file__).with_name("stage0.py")),
        str(work / "scene"),
        str(work / "blender.png"),
        "--samples",
        str(a.samples),
    ]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout[-1500:], proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"blender render failed with exit code {proc.returncode}")
    log = proc.stdout
    result["blender_wall_s"] = round(time.time() - t, 2)
    result["blender"] = json.loads([ln for ln in log.splitlines() if ln.startswith("{")][-1])

    px = round(a.size_m / a.gsd_m)
    base = RenderConfig(
        gsd_m=a.gsd_m,
        tile_px=px,
        size_m=a.size_m,
        seed=0,
        palette=palette.read(a.palette) if a.palette else None,
    )
    config, _ = ablation.variant_config("v8_objects", base)
    t = time.time()
    render_region(
        world,
        read_buildings(world),
        centre,
        work / "r25",
        config,
        roads=read_roads(world),
        fields=read_fields(world),
    )
    result["render25d_s"] = round(time.time() - t, 2)
    with rasterio.open(work / "r25" / "tiles" / "r00_c00.tif") as src:
        flat = src.read()[:3]
    blender = read_png_rgb(work / "blender.png")

    # The same sensor kernel the 2.5D renderer uses (FRK-50), with the same settings.
    lens = sensor_chain(
        blender,
        haze=(config.haze, config.sun_elevation_deg),
        tone=config.tone,
        contrast=config.contrast,
        psf_sigma_px=config.psf_sigma_px,
    )
    lens = add_noise(lens, config.noise_sigma, config.seed, "blender")
    np.save(work / "blender_sensor.npy", lens)
    sets = {"2.5d": cut(flat), "blender": cut(blender), "blender+sensor": cut(lens)}
    result["gauge"] = {}
    for real_dir in a.real:
        meta = json.loads((real_dir / "chips.json").read_text())
        real = chips.load_all(real_dir) if "tiles" in meta else chips.load_splits(real_dir)[0]
        real_chips = realism.sample([real.get(i)[0] for i in range(len(real))], 100)
        for name, synth in sets.items():
            result["gauge"][f"{name} vs {real_dir.name}"] = realism.compare(
                real_chips, [s for s in synth]
            )["distances"]
    (work / "stage0.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(work / "flat.npy", flat)
    np.save(work / "blender.npy", blender)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
