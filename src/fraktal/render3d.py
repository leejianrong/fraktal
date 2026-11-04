"""The 3D renderer's file contract and its Blender runner (ADR-0010, FRK-48).

Contract. In: a world directory (the same GeoParquet and GeoTIFF layers the 2.5D renderer reads),
a tile (centre, side, pixel size), a sun, and a seed. Out, in ``out_dir``:

- ``scene/``: the tile as data (``fraktal.blender_export``): ``scene.json``, ``ground.npy`` and the
  building meshes ``mesh_*.npy`` (``fraktal.building_grammar``) naming materials in ``assets/``.
  The same world, tile, sun and seed give byte-identical files, which is how determinism is
  defined for the 3D path (ADR-0010 decision 7): seeded scene equality, not pixel equality.
- ``image.png``: the Blender render, 8-bit RGB, north up, no tone curve.
- ``tile.tif``: the same pixels georeferenced in the tile's UTM zone.
- For a camera sequence (``trajectory``): ``scene/camera.json`` (the poses, part of the scene hash)
  and ``frames/frame_0000.png`` and on, instead of ``image.png`` and ``tile.tif``, with every
  frame's pose also in ``render.json``.
- ``render.json``: metadata, including ``scene_sha256`` (a hash of the scene files), timings,
  the Blender version, and every input needed to repeat the render.

Blender runs in a subprocess (``fraktal.blender_scene``, with a Python that has ``bpy``), so a hang
or crash cannot take the caller down: output is streamed as ``fraktal-log`` lines and the process
is killed at ``timeout_s``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import rasterio
from affine import Affine

from fraktal import camera as cameras
from fraktal import labels3d
from fraktal.blender_export import export_scene
from fraktal.log import emit

SCHEMA = 1
DEFAULT_ASSETS = Path(__file__).resolve().parents[2] / "assets"  # a source checkout
MESH_FILES = ("vertices", "face_sizes", "face_verts", "face_material", "face_building")
BLENDER_SCRIPT = Path(__file__).with_name("blender_scene.py")


class RenderError(RuntimeError):
    pass


def scene_sha256(scene_dir: Path) -> str:
    """One hash over the exported scene files, the thing that must repeat for the same seed."""
    digest = hashlib.sha256()
    names = ("scene.json", "ground.npy", "camera.json")  # camera.json only for sequences
    for name in (*names, *(f"mesh_{k}.npy" for k in MESH_FILES)):
        if (scene_dir / name).exists():
            digest.update((scene_dir / name).read_bytes())
    return digest.hexdigest()


def run_blender(
    python: str, script: Path, args: list[str], timeout_s: float
) -> tuple[dict[str, Any], float]:
    """Run ``script`` under ``python``; relay STAGE lines; return its last JSON line and seconds."""
    start = time.time()
    proc = subprocess.Popen(
        [python, str(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, kill)
    timer.start()
    tail: deque[str] = deque(maxlen=40)
    result: dict[str, Any] = {}
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if line.startswith("STAGE "):
                emit("render3d_blender", line=line[6:])
            elif line.startswith("{"):
                with contextlib.suppress(json.JSONDecodeError):
                    result = json.loads(line)
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
    if timed_out.is_set():
        raise RenderError(f"blender timed out after {timeout_s:.0f} s")
    if code:
        raise RenderError(f"blender exited with {code}; last output:\n" + "\n".join(tail))
    if not result:
        raise RenderError("blender printed no result line; last output:\n" + "\n".join(tail))
    return result, time.time() - start


def render(
    world_dir: Path,
    out_dir: Path,
    center_lonlat: tuple[float, float],
    size_m: float,
    gsd_m: float,
    *,
    sun: tuple[float, float] = (135.0, 50.0),
    seed: int = 0,
    samples: int = 32,
    threads: int = 0,
    bpy_python: str | None = None,
    timeout_s: float = 1800.0,
    script: Path = BLENDER_SCRIPT,
    trajectory: str | None = None,
    camera_options: dict[str, float] | None = None,
    frame_px: int = 512,
    assets_dir: Path | None = None,
    labels: bool = True,
    height_profile: str = "auto",
) -> dict[str, Any]:
    """Export the tile, render it in Blender, write the contract files, return ``render.json``."""
    python = bpy_python or sys.executable
    assets = assets_dir or Path(os.environ.get("FRAKTAL_ASSETS", DEFAULT_ASSETS))
    if not (assets / "manifest.json").is_file():
        raise RenderError(f"no assets directory at {assets}; pass --assets or set FRAKTAL_ASSETS")
    emit("render3d_start", world=str(world_dir), size_m=size_m, gsd_m=gsd_m, samples=samples)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scene = export_scene(
        world_dir, out_dir / "scene", center_lonlat, size_m, gsd_m, sun, seed,
        height_profile=height_profile,
    )  # fmt: skip
    poses = []
    if trajectory:  # a perspective sequence instead of the nadir tile (FRK-51)
        poses = cameras.trajectory(trajectory, scene["size_m"], seed, **(camera_options or {}))
        (out_dir / "scene" / "camera.json").write_text(
            json.dumps({"frames": [p.as_dict() for p in poses]}) + "\n"
        )
    digest = scene_sha256(out_dir / "scene")
    t1 = time.time()
    emit(
        "render3d_scene",
        sha256=digest,
        buildings=len(scene["buildings"]),
        trees=len(scene["trees"]),
        seconds=round(t1 - t0, 2),
    )
    target = out_dir / "frames" if poses else out_dir / "image.png"
    args = [str(out_dir / "scene"), str(target), "--samples", str(samples), "--assets", str(assets)]
    if poses:
        args += ["--px", str(frame_px)]
    if labels:
        args += ["--labels", str(out_dir / ("labels" if poses else "labels.png"))]
    if threads:
        args += ["--threads", str(threads)]
    blender, wall = run_blender(python, script, args, timeout_s)
    px = int(scene["px"])
    if not poses:
        west, south = scene["origin_sw"]
        transform = Affine(gsd_m, 0.0, west, 0.0, -gsd_m, south + px * gsd_m)
        with rasterio.open(out_dir / "image.png") as src:
            image = src.read()[:3]
        profile = {
            "driver": "GTiff", "count": 3, "dtype": "uint8", "height": px, "width": px,
            "crs": scene["crs"], "transform": transform, "compress": "deflate",
        }  # fmt: skip
        with rasterio.open(out_dir / "tile.tif", "w", **profile) as dst:
            dst.write(image)
        if labels:
            with rasterio.open(out_dir / "labels.png") as src:
                code = src.read()[:3]
            with rasterio.open(out_dir / "labels.tif", "w", **profile) as dst:
                dst.write(code)
    meta = {
        "schema": SCHEMA,
        "world": str(world_dir),
        "center_lonlat": list(center_lonlat),
        "size_m": scene["size_m"],
        "gsd_m": gsd_m,
        "px": px,
        "crs": scene["crs"],
        "origin_sw": scene["origin_sw"],
        "sun": scene["sun"],
        "seed": seed,
        "samples": samples,
        "scene_sha256": digest,
        "height_profile": height_profile,
        "assets": str(assets),
        "buildings": blender.get("buildings"),
        "trees": blender.get("trees"),
        "bpy": blender.get("bpy"),
        "seconds": {
            "export": round(t1 - t0, 2),
            "blender_wall": round(wall, 2),
            "blender_build": blender.get("build_s"),
            "blender_render": blender.get("render_s"),
        },
        "files": {"image": "image.png", "tile": "tile.tif", "scene": "scene/"},
    }
    if poses:  # frames are oblique, so there is no georeferenced tile
        meta["trajectory"] = trajectory
        meta["px"] = frame_px
        meta["frames"] = [p.as_dict() for p in poses]
        meta["files"] = {"frames": "frames/frame_NNNN.png", "scene": "scene/"}
    if labels:
        meta["labels"] = labels3d.meta([b["id"] for b in scene["buildings"]], len(scene["trees"]))
        meta["files"]["labels"] = "labels/frame_NNNN.png" if poses else "labels.png, labels.tif"
    (out_dir / "render.json").write_text(json.dumps(meta, indent=2) + "\n")
    emit("render3d_done", out=str(out_dir), seconds=round(time.time() - t0, 2))
    return meta


__all__ = ["BLENDER_SCRIPT", "RenderError", "render", "run_blender", "scene_sha256"]
