"""Render many 3D tiles from one world in one process (FRK-68).

``plan`` draws tile centres, suns and seeds from named sub-streams, so the same ``(world, count,
seed)`` gives the same tiles. ``run`` renders them one after another into ``tile_NNNN/`` folders
and writes ``batch.json`` after every tile, so a run that stops can be started again: a tile whose
``render.json`` carries the same ``spec_sha256`` is skipped. A tile that fails is recorded and the
batch carries on; ``run`` returns the failures so the caller can set an exit code.

To keep a dataset small, each finished tile keeps ``image.png``, ``labels.png`` and ``render.json``
(which holds the scene hash, pose and georeference); the exported scene and the TIFFs are removed
unless asked for. The scene is rebuilt from the world, the tile spec and the seed.

With a sensor (FRK-72) each spec also carries a drawn grade, blur and noise (the tropical ranges
from ``fraktal.randomise``), applied to ``image.png`` to give ``sensor.png``; the raw render stays.
The draw is a named sub-stream per tile, so it is part of the plan and of the resume hash.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from pyproj import Transformer

from fraktal import randomise, render3d
from fraktal.ids import substream
from fraktal.log import emit
from fraktal.procedural import read_buildings
from fraktal.render import RenderConfig, tile_px_for

SCHEMA = 1
SUN_AZIMUTH_DEG = (
    100.0,
    260.0,
)  # an equatorial sun, assumed: it crosses the sky in the east or west
SUN_ELEVATION_DEG = (35.0, 70.0)
CENTRE_TRIES = 40
SLIM_REMOVE = ("scene", "tile.tif", "labels.tif")
SENSOR_NAME = "sensor.png"


class BatchError(ValueError):
    pass


EDGE_SLACK_PX = 3  # pixels kept between a tile and the world's edge


def plan(
    world_dir: Path,
    count: int,
    seed: int,
    size_m: float,
    gsd_m: float,
    min_buildings: int = 1,
    sensor: bool = False,
    sun_elevation: tuple[float, float] = SUN_ELEVATION_DEG,
) -> list[dict[str, Any]]:
    """``count`` tile specs inside the world, each with at least ``min_buildings`` if it can.

    With ``sensor`` each spec gets a ``sensor`` block drawn from the tropical ranges.
    """
    manifest = json.loads((world_dir / "world.json").read_text())
    west, south, east, north = manifest["bounds"]
    half = size_m / 2
    slack = EDGE_SLACK_PX * gsd_m  # the render grid snaps outward by up to a pixel on each side
    if east - west < size_m + 2 * slack or north - south < size_m + 2 * slack:
        raise BatchError(f"the world is smaller than one {size_m:.0f} m tile")
    buildings = read_buildings(world_dir)
    index = buildings.sindex
    to_lonlat = Transformer.from_crs(manifest["crs"], 4326, always_xy=True)
    specs: list[dict[str, Any]] = []
    for i in range(count):
        rng = substream(seed, f"batch:tile:{i}")
        x = y = 0.0
        for _ in range(CENTRE_TRIES):
            x = float(rng.uniform(west + half, east - half))
            y = float(rng.uniform(south + half, north - half))
            x = min(max(x, west + half + slack), east - half - slack)  # only an edge draw moves
            y = min(max(y, south + half + slack), north - half - slack)
            hits = index.query(_window(x, y, half))
            if len(hits) >= min_buildings:
                break
        lon, lat = to_lonlat.transform(x, y)
        sun = (
            round(float(rng.uniform(*SUN_AZIMUTH_DEG)), 2),
            round(float(rng.uniform(*sun_elevation)), 2),
        )
        spec: dict[str, Any] = {
            "index": i,
            "center": [round(lon, 7), round(lat, 7)],
            "size_m": size_m,
            "gsd_m": gsd_m,
            "sun": list(sun),
            "seed": int(rng.integers(0, 2**31 - 1)),
        }
        if sensor:
            spec["sensor"] = _draw_sensor(seed, i, size_m, gsd_m, sun[1], manifest.get("biome", ""))
        specs.append(spec)
    return specs


def _draw_sensor(
    seed: int, index: int, size_m: float, gsd_m: float, sun_elevation: float, biome: str = ""
) -> dict[str, Any]:
    """Haze, tone, contrast, blur and noise for one tile, from the biome or tropical ranges."""
    base = RenderConfig(gsd_m=gsd_m, tile_px=tile_px_for(size_m, gsd_m), size_m=size_m, seed=0)
    drawn = randomise.sample_config(
        base, randomise.BIOME_RANGES.get(biome, randomise.TROPICAL), seed, index
    )
    return {
        "haze": round(drawn.haze, 4),
        "sun_elevation_deg": sun_elevation,
        "tone": [round(float(t), 4) for t in drawn.tone],
        "contrast": round(drawn.contrast, 4),
        "psf_sigma_px": round(drawn.psf_sigma_px, 4),
        "noise_sigma": round(drawn.noise_sigma, 4),
        "noise_seed": drawn.seed,
    }


def apply_sensor(tile_dir: Path, sensor: dict[str, Any]) -> None:
    """Write ``sensor.png`` next to ``image.png``: the same grade, blur and noise as 2.5D."""
    import rasterio

    from fraktal.kernels.sensor import sensor_chain
    from fraktal.sensor import add_noise

    with rasterio.open(tile_dir / "image.png") as src:
        rgb = src.read()[:3].astype("uint8")
    graded = sensor_chain(
        rgb,
        haze=(sensor["haze"], sensor["sun_elevation_deg"]),
        tone=tuple(sensor["tone"]),
        contrast=sensor["contrast"],
        psf_sigma_px=sensor["psf_sigma_px"],
    )
    out = add_noise(graded, sensor["noise_sigma"], sensor["noise_seed"], "batch")
    with rasterio.open(
        tile_dir / SENSOR_NAME, "w", driver="PNG", width=out.shape[2], height=out.shape[1],
        count=3, dtype="uint8",
    ) as dst:  # fmt: skip
        dst.write(out)


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
    if ok and meta.get("sensor"):
        ok = (tile_dir / SENSOR_NAME).is_file()
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
                if spec.get("sensor"):
                    apply_sensor(tile_dir, spec["sensor"])
                    meta["sensor"] = {**spec["sensor"], "file": SENSOR_NAME}
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


__all__ = ["BatchError", "apply_sensor", "plan", "run", "spec_sha256"]
