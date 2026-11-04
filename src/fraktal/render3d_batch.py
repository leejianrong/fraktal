"""Render many 3D tiles from one world in one process (FRK-68).

``plan`` draws tile centres, suns and seeds from named sub-streams, so the same ``(world, count,
seed)`` gives the same tiles. ``run`` renders them one after another into ``tile_NNNN/`` folders
and writes ``batch.json`` after every tile, so a run that stops can be started again: a tile whose
``render.json`` carries the same ``spec_sha256`` is skipped. A tile that fails is recorded and the
batch carries on; ``run`` returns the failures so the caller can set an exit code.

To keep a dataset small, each finished tile keeps ``image.png``, ``labels.png`` and ``render.json``
(which holds the scene hash, pose and georeference); the exported scene and the TIFFs are removed
unless asked for. The scene is rebuilt from the world, the tile spec and the seed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from pyproj import Transformer

from fraktal import render3d
from fraktal.ids import substream
from fraktal.log import emit
from fraktal.procedural import read_buildings

SCHEMA = 1
SUN_AZIMUTH_DEG = (
    100.0,
    260.0,
)  # an equatorial sun, assumed: it crosses the sky in the east or west
SUN_ELEVATION_DEG = (35.0, 70.0)
CENTRE_TRIES = 40
SLIM_REMOVE = ("scene", "tile.tif", "labels.tif")


class BatchError(ValueError):
    pass


def plan(
    world_dir: Path,
    count: int,
    seed: int,
    size_m: float,
    gsd_m: float,
    min_buildings: int = 1,
) -> list[dict[str, Any]]:
    """``count`` tile specs inside the world, each with at least ``min_buildings`` if it can."""
    manifest = json.loads((world_dir / "world.json").read_text())
    west, south, east, north = manifest["bounds"]
    half = size_m / 2
    if east - west < size_m or north - south < size_m:
        raise BatchError(f"the world is smaller than one {size_m:.0f} m tile")
    buildings = read_buildings(world_dir)
    index = buildings.sindex
    to_lonlat = Transformer.from_crs(manifest["crs"], 4326, always_xy=True)
    specs: list[dict[str, Any]] = []
    for i in range(count):
        rng = substream(seed, f"batch:tile:{i}")
        x = y = 0.0
        for _ in range(CENTRE_TRIES):
            x = (
                float(rng.uniform(west + half, east - half))
                if east - west > size_m
                else west + half
            )
            y = (
                float(rng.uniform(south + half, north - half))
                if north - south > size_m
                else south + half
            )
            hits = index.query(_window(x, y, half))
            if len(hits) >= min_buildings:
                break
        lon, lat = to_lonlat.transform(x, y)
        specs.append(
            {
                "index": i,
                "center": [round(lon, 7), round(lat, 7)],
                "size_m": size_m,
                "gsd_m": gsd_m,
                "sun": [
                    round(float(rng.uniform(*SUN_AZIMUTH_DEG)), 2),
                    round(float(rng.uniform(*SUN_ELEVATION_DEG)), 2),
                ],
                "seed": int(rng.integers(0, 2**31 - 1)),
            }
        )
    return specs


def _window(x: float, y: float, half: float) -> Any:
    from shapely.geometry import box

    return box(x - half, y - half, x + half, y + half)


def spec_sha256(spec: dict[str, Any], samples: int, world_id: str) -> str:
    payload = json.dumps({"spec": spec, "samples": samples, "world": world_id}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _done(tile_dir: Path, digest: str) -> dict[str, Any] | None:
    """The tile's ``render.json`` if it was finished for exactly this spec."""
    try:
        meta = json.loads((tile_dir / "render.json").read_text())
    except (OSError, ValueError):
        return None
    ok = meta.get("spec_sha256") == digest and (tile_dir / "image.png").is_file()
    return meta if ok else None


def run(
    world_dir: Path,
    out_dir: Path,
    specs: list[dict[str, Any]],
    *,
    samples: int = 16,
    threads: int = 0,
    keep_scene: bool = False,
    **render_options: Any,
) -> dict[str, Any]:
    """Render every spec; returns the batch record (also written as ``batch.json``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    world_id = json.loads((world_dir / "world.json").read_text()).get("world_id", "")
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "world": str(world_dir),
        "world_id": world_id,
        "samples": samples,
        "tiles": [],
    }
    started = time.time()
    total = len(specs)
    for spec in specs:
        i = int(spec["index"])
        tile_dir = out_dir / f"tile_{i:04d}"
        digest = spec_sha256(spec, samples, world_id)
        entry: dict[str, Any] = {
            "index": i,
            "dir": tile_dir.name,
            "spec": spec,
            "spec_sha256": digest,
        }
        t0 = time.time()
        meta = _done(tile_dir, digest)
        if meta is not None:
            entry.update(status="skipped", scene_sha256=meta.get("scene_sha256"))
            emit("render3d_batch_tile", index=i, of=total, status="skipped")
        else:
            shutil.rmtree(tile_dir, ignore_errors=True)
            try:
                meta = render3d.render(
                    world_dir, tile_dir, tuple(spec["center"]), spec["size_m"], spec["gsd_m"],
                    sun=tuple(spec["sun"]), seed=spec["seed"], samples=samples, threads=threads,
                    **render_options,
                )  # fmt: skip
            except (render3d.RenderError, OSError, ValueError) as err:
                entry.update(status="failed", error=str(err)[-600:])
                emit(
                    "render3d_batch_tile", index=i, of=total, status="failed", error=str(err)[-200:]
                )
            else:
                meta["spec_sha256"] = digest
                (tile_dir / "render.json").write_text(json.dumps(meta, indent=2) + "\n")
                if not keep_scene:
                    for name in SLIM_REMOVE:
                        target = tile_dir / name
                        shutil.rmtree(target) if target.is_dir() else target.unlink(missing_ok=True)
                entry.update(status="rendered", scene_sha256=meta["scene_sha256"])
                emit("render3d_batch_tile", index=i, of=total, status="rendered",
                     seconds=round(time.time() - t0, 1))  # fmt: skip
        entry["seconds"] = round(time.time() - t0, 2)
        record["tiles"].append(entry)
        _write(out_dir, record, started)
    return record


def _write(out_dir: Path, record: dict[str, Any], started: float) -> None:
    counts = {
        s: sum(t["status"] == s for t in record["tiles"]) for s in ("rendered", "skipped", "failed")
    }
    record["counts"] = counts
    record["wall_seconds"] = round(time.time() - started, 1)
    record["mean_seconds_per_rendered_tile"] = (
        round(
            sum(t["seconds"] for t in record["tiles"] if t["status"] == "rendered")
            / counts["rendered"],
            2,
        )
        if counts["rendered"]
        else None
    )
    tmp = out_dir / "batch.json.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out_dir / "batch.json")


__all__ = ["BatchError", "plan", "run", "spec_sha256"]
