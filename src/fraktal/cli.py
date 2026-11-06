"""Command-line entry point.

Every command returns a result envelope. With ``--json`` the envelope is printed to stdout
as ``{"ok": bool, "data": ..., "error": ...}``; without it, humans get plain text and errors
go to stderr. Exit codes are part of the contract (see ``ExitCode``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from fraktal import (
    __version__,
    ablation,
    agents,
    assets,
    benchmark,
    biomes,
    chips,
    evalset,
    heights,
    ingest,
    inspection,
    palette,
    procedural,
    realism,
    render,
    render3d,
    render3d_batch,
    report,
    skeleton,
    split,
    synth,
    terrain,
    world,
)
from fraktal import (
    camera as cameras,
)

PROCEDURAL_MARGIN_M = 200.0  # world border beyond the rendered region, so edge tiles stay inside


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2  # argparse exits with 2 on bad arguments
    NOT_IMPLEMENTED = 3
    FAILURE = 4


class CliError(Exception):
    """A failure a command wants reported with a specific exit code and error kind."""

    def __init__(self, code: ExitCode, kind: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class Result:
    data: dict[str, Any]
    text: str


Handler = Callable[[argparse.Namespace], Result]


def _version(_args: argparse.Namespace) -> Result:
    return Result({"version": __version__}, f"fraktal {__version__}")


def _not_implemented(name: str) -> Handler:
    def handler(_args: argparse.Namespace) -> Result:
        raise CliError(
            ExitCode.NOT_IMPLEMENTED, "not_implemented", f"`fraktal {name}` is not implemented yet"
        )

    return handler


def _ingest(args: argparse.Namespace) -> Result:
    data: dict[str, Any] = {"dataset": ingest.DATASET}
    lines: list[str] = []
    try:
        if args.what in ("imagery", "all"):
            result = ingest.ingest(args.data_dir, keep_archive=args.keep_archive)
            data.update(
                status=result.status,
                path=str(result.path),
                archive_bytes=result.bytes,
                resumed_from=result.resumed_from,
            )
            lines.append(f"imagery {result.status} {result.path}")
        if args.what in ("terrain", "all"):
            aoi = _aoi(args)
            layers = terrain.ingest_terrain(args.data_dir, aoi)
            data["terrain"] = {
                "status": layers.status,
                "path": str(layers.path),
                "shape": list(layers.shape),
                "bounds": list(layers.bounds),
            }
            lines.append(f"terrain {layers.status} {layers.path}")
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "ingest_failed", str(err)) from err
    return Result(data, "\n".join(lines))


def _aoi(args: argparse.Namespace) -> terrain.Bounds:
    """The AOI is the union of the ingested imagery tiles, unless given explicitly."""
    if args.bounds:
        left, bottom, right, top = args.bounds
        return (left, bottom, right, top)
    tiles, _crs = split.list_tiles(args.data_dir / "spacenet2" / "khartoum")
    return (
        min(t.bounds[0] for t in tiles),
        min(t.bounds[1] for t in tiles),
        max(t.bounds[2] for t in tiles),
        max(t.bounds[3] for t in tiles),
    )


def _data_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FRAKTAL_DATA_DIR", "data")),
        help="cache directory (default: $FRAKTAL_DATA_DIR or ./data, which git ignores)",
    )


def _ingest_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument(
        "--what", choices=["imagery", "terrain", "all"], default="all", help="which data to fetch"
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="terrain AOI in EPSG:4326 (default: union of the imagery tiles)",
    )
    parser.add_argument(
        "--keep-archive", action="store_true", help="keep the tarball after extraction"
    )


def _render(args: argparse.Namespace) -> Result:
    terrain_dir = args.data_dir / "terrain" / "khartoum"
    out = args.out or args.data_dir / "render"
    try:
        if not (terrain_dir / terrain.MARKER).is_file():
            raise FileNotFoundError(f"no terrain under {terrain_dir}; run `fraktal ingest`")
        buildings = world.read_layer(args.buildings, "building")
        center = tuple(args.center) if args.center else _terrain_center(terrain_dir)
        config = render.RenderConfig(
            size_m=args.size_m,
            sun_azimuth_deg=args.sun_azimuth,
            sun_elevation_deg=args.sun_elevation,
        )
        manifest = render.render_region(terrain_dir, buildings, (center[0], center[1]), out, config)
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "render_failed", str(err)) from err
    n = len(manifest["tiles"])
    data = {"out": str(out), "tiles": n, "crs": manifest["crs"]}
    return Result(data, f"rendered {n} tiles to {out}")


def _eval_set(args: argparse.Namespace) -> Result:
    out = args.out or args.data_dir / "eval" / "open-cities"
    try:
        scenes = [evalset.parse_scene(s) for s in args.scenes]
        path = evalset.build_eval_set(
            scenes,
            out,
            base=args.base,
            gsd_m=args.gsd_m,
            chip_px=args.chip_px,
            max_chips_per_scene=args.max_chips,
            seed=args.seed,
        )
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "eval_set_failed", str(err)) from err
    meta = json.loads((path / "chips.json").read_text())
    data = {"chips": str(path), "tiles": len(meta["tiles"]), "scenes": meta["scenes"]}
    return Result(data, f"wrote {len(meta['tiles'])} evaluation chips to {path} (eval only)")


def _eval_set_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument(
        "scenes",
        nargs="+",
        metavar="CITY/ID",
        help="Open Cities train_tier_1 scenes, e.g. dar/a017f9",
    )
    parser.add_argument("--base", default=evalset.BASE_URL, help="scene root (URL or directory)")
    parser.add_argument("--gsd-m", type=float, default=0.5, help="chip pixel size in metres")
    parser.add_argument("--chip-px", type=int, default=500)
    parser.add_argument("--max-chips", type=int, default=20, help="chips per scene")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default <data-dir>/eval/open-cities")


def _benchmark(args: argparse.Namespace) -> Result:
    manifest = args.split or args.data_dir / "spacenet2" / "khartoum-split.json"
    out = args.out or args.data_dir / "benchmark" / "report.json"
    try:
        chips_dir = args.chips or chips.build_chips(args.data_dir, manifest)
        train_set, test_set = chips.load_splits(chips_dir)
        recipe = benchmark.Recipe(steps=args.steps) if args.steps else benchmark.Recipe()
        synthetic = chips.load_all(args.synthetic) if args.synthetic else None
        report = benchmark.run_arms(
            train_set, test_set, recipe, args.seed, args.device, synthetic=synthetic
        )
    except (OSError, ValueError, RuntimeError) as err:
        raise CliError(ExitCode.FAILURE, "benchmark_failed", str(err)) from err
    benchmark.write_report(report, out)
    iou = {name: round(arm["iou"], 4) for name, arm in report["arms"].items()}
    data: dict[str, Any] = {"report": str(out), "iou": iou}
    if "comparison" in report:
        data["comparison"] = report["comparison"]
    return Result(data, f"wrote {out}: IoU {iou}")


def _benchmark_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, help="optimiser steps per arm (default: the recipe's)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--split", type=Path, help="split manifest from `fraktal split`")
    parser.add_argument("--chips", type=Path, help="prebuilt chips directory (skips building)")
    parser.add_argument("--synthetic", type=Path, help="synthetic chips dir (adds arm 3)")
    parser.add_argument("--out", type=Path, help="report path")


def _ablate(args: argparse.Namespace) -> Result:
    manifest = args.split or args.data_dir / "spacenet2" / "khartoum-split.json"
    out = args.out or args.data_dir / "ablation" / "ablation.json"
    names = args.variants or list(ablation.VARIANTS)
    root: Path = args.variants_dir or args.data_dir / "ablation"
    try:
        chips_dir = args.chips or chips.build_chips(args.data_dir, manifest)
        train_set, test_set = chips.load_splits(chips_dir)
        recipe = benchmark.Recipe(steps=args.steps) if args.steps else benchmark.Recipe()
        real = realism.sample([train_set.get(i)[0] for i in range(len(train_set))], 100)
        report = ablation.run_ablation(
            train_set,
            test_set,
            recipe,
            args.seeds,
            args.device,
            lambda name: chips.load_all(ablation.wait_for_variant(root, name, args.wait_secs)),
            names,
            out,
            real_chips=real,
            fraction=args.fraction,
        )
    except (OSError, ValueError, RuntimeError) as err:
        raise CliError(ExitCode.FAILURE, "ablation_failed", str(err)) from err
    return Result({"report": str(out), "runs": len(report["runs"])}, f"wrote {out}")


def _ablate_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--variants", nargs="+", choices=ablation.VARIANTS)
    parser.add_argument(
        "--variants-dir", type=Path, help="holds <variant>/chips (default ablation/)"
    )
    parser.add_argument(
        "--wait-secs", type=float, default=3600, help="wait per variant's synth job"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--fraction", type=float, default=0.1, help="share of real training tiles for arm 1 and 3"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", type=Path)
    parser.add_argument("--chips", type=Path)
    parser.add_argument("--out", type=Path, help="report path")


def _report(args: argparse.Namespace) -> Result:
    out: Path = args.out or args.data_dir / "reports"
    try:
        ablation_report = report.load_ablation(args.ablation)
        real: list[Any] = []
        variants: dict[str, list[Any]] = {}
        root: Path = args.variants_dir or args.data_dir / "ablation"
        real_dir = args.real or args.data_dir / "spacenet2" / "chips"
        if (real_dir / "chips.json").is_file():
            train = chips.load_splits(real_dir)[0]
            real = [train.get(i)[0] for i in range(len(train))]
            for name in ablation_report["variants"]:
                if (root / name / "chips" / "chips.json").is_file():
                    source = chips.load_all(root / name / "chips")
                    variants[name] = [source.get(i)[0] for i in range(len(source))]
        figures = report.build_figures(ablation_report, out / "figures", real, variants)
        pdf = report.build_pdf(ablation_report, figures, out / "realism-report.pdf")
    except (OSError, ValueError, KeyError, RuntimeError) as err:
        raise CliError(ExitCode.FAILURE, "report_failed", str(err)) from err
    data = {"pdf": str(pdf), "figures": {k: str(v) for k, v in figures.items()}}
    return Result(data, f"wrote {pdf} ({len(figures)} figures)")


def _report_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--ablation", type=Path, required=True, help="ablation.json from `ablate`")
    parser.add_argument("--real", type=Path, help="real chips dir (default spacenet2/chips)")
    parser.add_argument(
        "--variants-dir", type=Path, help="holds <variant>/chips (default ablation/)"
    )
    parser.add_argument("--out", type=Path, help="output directory (default <data-dir>/reports)")


def _palette(args: argparse.Namespace) -> Result:
    out: Path = args.out or args.data_dir / "spacenet2" / "palette.json"
    root = args.data_dir / "spacenet2" / "khartoum"
    try:
        chips_dir = args.chips or args.data_dir / "spacenet2" / "chips"
        train = chips.load_splits(chips_dir)[0]  # train tiles only: no test tile reaches a world
        paths = {t: chips.tile_path(root, t) for t in train.ids}
        landcover = args.data_dir / "terrain" / "khartoum" / "landcover.tif"
        if not landcover.is_file():
            raise FileNotFoundError(f"no land cover at {landcover}; run `fraktal ingest`")
        fitted = palette.fit(train, paths, landcover, tuple(args.tone), args.seed, args.max_tiles)
        palette.write(fitted, out, train.ids)
    except (OSError, ValueError, KeyError) as err:
        raise CliError(ExitCode.FAILURE, "palette_failed", str(err)) from err
    data = {
        "palette": str(out),
        "classes": sorted(fitted.classes),
        "roof_colours": len(fitted.roofs),
    }
    return Result(
        data, f"wrote {out}: {len(fitted.classes)} classes, {len(fitted.roofs)} roof colours"
    )


def _palette_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--chips", type=Path, help="chips dir (default spacenet2/chips)")
    parser.add_argument(
        "--tone", type=float, nargs=3, default=render.KHARTOUM_TONE, metavar=("R", "G", "B")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tiles", type=int, default=150, help="train tiles sampled")
    parser.add_argument("--out", type=Path, help="palette path")


def _inspect(args: argparse.Namespace) -> Result:
    try:
        if args.id:
            found = inspection.find_feature(args.path, args.id)
            return Result(
                found,
                f"{found['kind']} {found['id']}: {found['geometry']['type']}, "
                f"provenance {found['provenance']}, {found['properties']}",
            )
        summary = inspection.summarise(args.path)
    except (OSError, ValueError, KeyError) as err:
        raise CliError(ExitCode.FAILURE, "inspect_failed", str(err)) from err
    text = "; ".join(f"{k}: {v['features']}" for k, v in summary["layers"].items())
    return Result(summary, f"{args.path}: {text}")


def _inspect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path, help="a world directory or one layer file")
    parser.add_argument("--id", help="feature id (or a unique prefix of 6+ characters)")


def _waypoints(path: Path) -> list[tuple[float, float]]:
    """Waypoints in tile metres from a JSON file: a list of [x, y] or {"waypoints": [...]}."""
    try:
        data = json.loads(path.read_text())
        points = data["waypoints"] if isinstance(data, dict) else data
        return [(float(x), float(y)) for x, y in points]
    except (OSError, ValueError, KeyError, TypeError) as err:
        msg = f"cannot read waypoints from {path}: {err}"
        raise CliError(ExitCode.FAILURE, "bad_path", msg) from err


def _render3d(args: argparse.Namespace) -> Result:
    try:
        meta = render3d.render(
            args.world,
            args.out,
            (args.center[0], args.center[1]),
            args.size_m,
            args.gsd_m,
            sun=(args.sun_azimuth, args.sun_elevation),
            seed=args.seed,
            samples=args.samples,
            threads=args.threads,
            bpy_python=args.bpy_python or os.environ.get("FRAKTAL_BPY_PYTHON"),
            timeout_s=args.timeout_s,
            trajectory="path" if args.path else args.trajectory,
            waypoints=_waypoints(args.path) if args.path else None,
            make_video=args.video,
            with_depth=args.depth,
            with_flow=args.flow,
            follow=(
                {
                    "kind": args.follow,
                    "id": args.follow_id,
                    "mount_height_m": args.mount_height,
                    "pitch_deg": args.pitch,
                }
                if args.follow
                else None
            ),
            traffic=(
                {
                    "cars_per_km": args.cars_per_km,
                    "drive_side": args.drive_side,
                    "people_per_km": args.people_per_km,
                    "drones": args.drones,
                    "drone_scale": args.drone_scale,
                }
                if args.traffic
                else None
            ),
            camera_options={
                "altitude_m": args.altitude_m,
                "off_nadir_deg": args.off_nadir,
                "fov_deg": args.fov if args.fov is not None else (90.0 if args.follow else 40.0),
                "duration_s": args.duration_s,
                "fps": args.fps,
                "speed_mps": args.speed,
                "max_accel": args.max_accel,
                "max_yaw_rate_dps": args.max_yaw_rate,
            },
            frame_px=args.frame_px,
            assets_dir=args.assets,
            labels=not args.no_labels,
            height_profile=args.height_profile,
        )
    except (OSError, ValueError, render3d.RenderError) as err:
        raise CliError(ExitCode.FAILURE, "render3d_failed", str(err)) from err
    secs = meta["seconds"]["blender_render"]
    if "frames" in meta:
        n, kind = len(meta["frames"]), meta["trajectory"]
        text = f"rendered {n} {kind} frames to {args.out}, blender {secs} s"
        return Result(meta, text)
    text = (
        f"rendered {meta['px']} px ({meta['buildings']} buildings, {meta['trees']} trees) "
        f"to {args.out}, blender {secs} s"
    )
    return Result(meta, text)


def _render3d_batch(args: argparse.Namespace) -> Result:
    try:
        specs = render3d_batch.plan(
            args.world, args.count, args.seed, args.size_m, args.gsd_m, args.min_buildings,
            sensor=args.sensor,
        )  # fmt: skip
        record = render3d_batch.run(
            args.world,
            args.out,
            specs,
            samples=args.samples,
            threads=args.threads,
            keep_scene=args.keep_scene,
            bpy_python=args.bpy_python or os.environ.get("FRAKTAL_BPY_PYTHON"),
            timeout_s=args.timeout_s,
            assets_dir=args.assets,
        )
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "render3d_batch_failed", str(err)) from err
    counts = record["counts"]
    text = (
        f"{counts['rendered']} rendered, {counts['skipped']} skipped, {counts['failed']} failed "
        f"in {record['wall_seconds']} s ({args.out}/batch.json)"
    )
    if counts["failed"]:
        failed = [t["dir"] for t in record["tiles"] if t["status"] == "failed"]
        raise CliError(
            ExitCode.FAILURE, "render3d_batch_failed", f"{text}; failed: {', '.join(failed)}"
        )
    return Result(record, text)


def _traffic(args: argparse.Namespace) -> Result:
    try:
        roads, bounds, world_side = agents.tile_roads(
            args.world, (args.center[0], args.center[1]), args.size_m
        )
        config = agents.TrafficConfig(
            cars_per_km=args.cars_per_km, drive_side=args.drive_side or world_side
        )
        graph = agents.build_graph(roads, bounds, config)
        times = np.arange(0.0, args.duration_s + 1e-9, 1.0 / args.fps)
        tracks = agents.simulate_cars(graph, times, args.seed, config)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "agents.json").write_text(json.dumps(tracks.as_dict()) + "\n")
        agents.draw_tracks(graph, tracks, bounds[2], args.out / "traffic.png")
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "traffic_failed", str(err)) from err
    data = {
        "cars": len(tracks.ids),
        "frames": len(times),
        "road_km": round(graph.total_length_m() / 1000, 2),
        "drive_side": config.drive_side,
        "files": ["agents.json", "traffic.png"],
    }
    return Result(
        data, f"{data['cars']} cars on {data['road_km']} km of road, written to {args.out}"
    )


def _traffic_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("world", type=Path, help="a world directory (from `fraktal generate`)")
    parser.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--size-m", type=float, default=500.0, help="tile side in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--cars-per-km", type=float, default=6.0)
    parser.add_argument("--drive-side", choices=agents.DRIVE_SIDES, help="default: the world's")


def _demo(args: argparse.Namespace) -> Result:
    from fraktal import demo

    try:
        info = demo.build(
            args.world, args.out, tuple(args.center) if args.center else None,
            size_m=args.size_m, seed=args.seed, duration_s=args.duration_s,
            cars_per_km=args.cars_per_km, people_per_km=args.people_per_km, drones=args.drones,
        )  # fmt: skip
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "demo_failed", str(err)) from err
    return Result(info, f"wrote {info['out']} ({info['bytes'] // 1024} KiB); open it in a browser")


def _demo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("world", type=Path, help="a world directory (from `fraktal generate`)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"),
                        help="default: the world's centre")  # fmt: skip
    parser.add_argument("--size-m", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=60.0, help="length of the loop")
    parser.add_argument("--cars-per-km", type=float, default=12.0)
    parser.add_argument("--people-per-km", type=float, default=10.0)
    parser.add_argument("--drones", type=int, default=3)


def _render3d_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("world", type=Path, help="a world directory (from `fraktal generate`)")
    parser.add_argument("--out", type=Path, required=True, help="output directory (resumable)")
    parser.add_argument("--count", type=int, required=True, help="how many tiles")
    parser.add_argument(
        "--seed", type=int, default=0, help="seeds the tile centres, suns and scenes"
    )
    parser.add_argument("--size-m", type=float, default=500.0)
    parser.add_argument("--gsd-m", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0, help="0 means all cores")
    parser.add_argument("--min-buildings", type=int, default=1, help="skip empty tile windows")
    parser.add_argument("--bpy-python", help="a Python with bpy (default $FRAKTAL_BPY_PYTHON)")
    parser.add_argument("--timeout-s", type=float, default=1800.0, help="per tile")
    parser.add_argument("--assets", type=Path, help="assets directory (default $FRAKTAL_ASSETS)")
    parser.add_argument(
        "--keep-scene", action="store_true", help="keep each tile's scene and TIFFs"
    )
    parser.add_argument(
        "--sensor", action="store_true", help="also write sensor.png (tropical grade, blur, noise)"
    )


def _render3d_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("world", type=Path, help="a world directory (from `fraktal generate`)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"), required=True)
    parser.add_argument("--size-m", type=float, default=500.0, help="tile side in metres")
    parser.add_argument("--gsd-m", type=float, default=0.5)
    parser.add_argument("--sun-azimuth", type=float, default=135.0)
    parser.add_argument("--sun-elevation", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel")
    parser.add_argument("--threads", type=int, default=0, help="0 means all cores")
    parser.add_argument("--bpy-python", help="a Python with bpy (default $FRAKTAL_BPY_PYTHON)")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--trajectory", choices=cameras.KINDS, help="render a perspective sequence")
    parser.add_argument("--path", type=Path, help="fly through the waypoints in this JSON file")
    parser.add_argument("--video", action="store_true", help="also write video.mp4 (needs ffmpeg)")
    parser.add_argument("--depth", action="store_true", help="also write 16-bit depth images")
    parser.add_argument(
        "--flow", action="store_true", help="optical flow per frame (implies --depth)"
    )
    parser.add_argument("--traffic", action="store_true", help="moving cars, with tracks.json")
    parser.add_argument("--cars-per-km", type=float, default=6.0, help="with --traffic")
    parser.add_argument("--drones", type=int, default=0, help="flying drones, with --traffic")
    parser.add_argument(
        "--drone-scale",
        type=float,
        default=1.0,
        help="drone size relative to a 0.5 m quadcopter (6 is a 3 m delivery or fixed-wing UAV)",
    )
    parser.add_argument("--people-per-km", type=float, default=0.0, help="walkers, with --traffic")
    parser.add_argument("--drive-side", choices=agents.DRIVE_SIDES, help="default: the world's")
    parser.add_argument("--altitude-m", type=float, default=300.0)
    parser.add_argument("--off-nadir", type=float, default=35.0, help="degrees from straight down")
    parser.add_argument(
        "--fov", type=float, help="field of view in degrees (default 40, or 90 when following)"
    )
    parser.add_argument(
        "--follow", choices=("car", "person"), help="ride on a car or person (needs --traffic)"
    )
    parser.add_argument(
        "--follow-id", help="which one, e.g. car-0007 (default: the most travelled)"
    )
    parser.add_argument("--mount-height", type=float, help="camera height above ground, metres")
    parser.add_argument("--pitch", type=float, help="degrees below the horizon when following")
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=40.0, help="cruise speed, m/s")
    parser.add_argument("--max-accel", type=float, default=3.0, help="path flights, m/s^2")
    parser.add_argument("--max-yaw-rate", type=float, default=60.0, help="path flights, deg/s")
    parser.add_argument("--assets", type=Path, help="assets directory (default $FRAKTAL_ASSETS)")
    parser.add_argument("--frame-px", type=int, default=512, help="frame side for a sequence")
    parser.add_argument("--no-labels", action="store_true", help="skip the label images")
    parser.add_argument(
        "--height-profile",
        default="auto",
        choices=["auto", "world", *sorted(heights.PROFILES)],
        help="auto: the biome's storey mix; world: the world's own heights",
    )


def _assets(args: argparse.Namespace) -> Result:
    problems = assets.check(args.root)
    if problems:
        raise CliError(ExitCode.FAILURE, "asset_check_failed", "; ".join(problems))
    return Result({"problems": []}, f"{args.root}: manifest and files agree, all licences allowed")


def _assets_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("assets"), help="the assets directory")


def _split_manifest(args: argparse.Namespace) -> Path:
    path: Path = args.split or args.data_dir / "spacenet2" / "khartoum-split.json"
    return path


def _biome_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Settings for the biome: a JSON file first, then the explicit command-line flags."""
    overrides: dict[str, Any] = {}
    if args.biome_config:
        loaded = json.loads(args.biome_config.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("--biome-config must hold a JSON object")
        overrides.update(loaded)
    for flag, key in (("size_m", "size_m"), ("gsd_m", "gsd_m"), ("aridity", "aridity")):
        if getattr(args, flag, None) is not None:
            overrides[key] = getattr(args, flag)
    return overrides


def _generate_procedural(args: argparse.Namespace) -> Result:
    if not args.center:
        raise CliError(ExitCode.USAGE, "usage", "--fallback procedural needs --center LON LAT")
    out: Path = args.out or args.data_dir / "worlds" / f"procedural-{args.seed}"
    try:
        config = biomes.resolve(args.biome, _biome_overrides(args))
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.USAGE, "usage", str(err)) from err
    try:
        manifest = procedural.generate_world(
            out, (args.center[0], args.center[1]), args.seed, config, args.real_terrain
        )
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "generate_failed", str(err)) from err
    layers = {k: v["features"] for k, v in manifest["layers"].items()}
    text = f"wrote {out}: " + ", ".join(f"{n} {k}s" for k, n in layers.items())
    data = {"world": str(out), "world_id": manifest["world_id"], "biome": config.biome}
    return Result({**data, "layers": layers}, text)


def _generate(args: argparse.Namespace) -> Result:
    if args.fallback == "procedural":
        return _generate_procedural(args)
    out = args.out or args.data_dir / "worlds" / f"{skeleton.WORLD_ID}.parquet"
    try:
        layer = skeleton.build_skeleton(args.data_dir, _split_manifest(args), args.seed)
        world.write_layer(layer, "building", out)
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "generate_failed", str(err)) from err
    info = skeleton.describe(layer)
    return Result({"world": str(out), **info}, f"wrote {out}: {info['buildings']} buildings")


def _generate_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=Path, help="split manifest from `fraktal split`")
    parser.add_argument("--out", type=Path, help="world GeoParquet path")


def _generate_cli_args(parser: argparse.ArgumentParser) -> None:
    _generate_args(parser)
    parser.add_argument(
        "--fallback",
        choices=["none", "procedural"],
        default="none",
        help="procedural builds a whole world from the seed instead of using real footprints",
    )
    parser.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"))
    parser.add_argument(
        "--biome",
        default=procedural.DEFAULT_BIOME,
        help=f"biome preset: {', '.join(biomes.names())} (default {procedural.DEFAULT_BIOME})",
    )
    parser.add_argument(
        "--biome-config", type=Path, help="JSON of biome settings to override (unknown keys fail)"
    )
    parser.add_argument("--size-m", type=float, help="procedural region side (biome default)")
    parser.add_argument("--gsd-m", type=float, help="procedural grid cell size (biome default)")
    parser.add_argument("--aridity", type=float, help="0 lush to 1 barren (biome default)")
    parser.add_argument(
        "--real-terrain", type=Path, help="ingested terrain dir to keep real DEM and land cover"
    )


def _synth(args: argparse.Namespace) -> Result:
    terrain_dir = args.data_dir / "terrain" / "khartoum"
    out = args.out or args.data_dir / "synthetic"
    manifest = _split_manifest(args)
    config = render.RenderConfig(
        gsd_m=args.gsd_m,
        tile_px=args.tile_px,
        size_m=args.size_m,
        sun_azimuth_deg=args.sun_azimuth,
        sun_elevation_deg=args.sun_elevation,
        seed=args.seed,
        variation=args.variation,
        tone=tuple(args.tone),
        texture=args.texture,
        haze=args.haze,
        psf_sigma_px=args.psf,
        noise_sigma=args.noise,
        penumbra_m=args.penumbra,
        ambient=args.ambient,
        contrast=args.contrast,
        bright_roofs=args.bright_roofs,
        blank_prob=args.blank_prob,
        objects=args.objects,
        roof_detail=args.roof_detail,
        palette=palette.read(args.palette) if args.palette else None,
    )
    randomised = args.randomise
    if args.variant:
        config, randomised = ablation.variant_config(args.variant, config)
    try:
        if not (terrain_dir / terrain.MARKER).is_file():
            raise FileNotFoundError(f"no terrain under {terrain_dir}; run `fraktal ingest`")
        centers = synth.pick_centers(manifest, args.seed, args.size_m)
        layer = (
            None if args.procedural else skeleton.build_skeleton(args.data_dir, manifest, args.seed)
        )
        crs = str(layer.crs) if layer is not None else skeleton.UTM_CRS
        meta = synth.synthesize(
            terrain_dir,
            layer,
            centers,
            args.regions,
            skeleton.held_out_footprints(manifest, crs),
            None if layer is None else skeleton.train_coverage(manifest, crs),
            out,
            config,
            ranges=ablation.variant_ranges(args.variant) if randomised else None,
            seed=args.seed,
            procedural=(
                biomes.resolve(
                    args.biome,
                    {"size_m": args.size_m + 2 * PROCEDURAL_MARGIN_M, "gsd_m": args.world_gsd_m},
                )
                if args.procedural
                else None
            ),
            max_chips=args.max_chips,
        )
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "synth_failed", str(err)) from err
    n, dropped = len(meta["tiles"]), len(meta["dropped"])
    data = {"chips": str(out / "chips"), "tiles": n, "dropped": dropped}
    return Result(data, f"wrote {n} synthetic chips to {out / 'chips'} ({dropped} dropped)")


def _synth_args(parser: argparse.ArgumentParser) -> None:
    _generate_args(parser)
    parser.add_argument("--regions", type=int, default=3, help="regions to render")
    parser.add_argument(
        "--procedural",
        action="store_true",
        help="generate each region's buildings procedurally instead of using real footprints",
    )
    parser.add_argument(
        "--biome", default=procedural.DEFAULT_BIOME, help="biome preset for --procedural worlds"
    )
    parser.add_argument("--max-chips", type=int, help="thin kept chips evenly to about this many")
    parser.add_argument(
        "--world-gsd-m", type=float, default=5.0, help="procedural terrain and land cover cell size"
    )
    parser.add_argument("--size-m", type=float, default=2000.0)
    parser.add_argument("--gsd-m", type=float, default=0.3, help="match the real imagery GSD")
    parser.add_argument("--tile-px", type=int, default=650)
    parser.add_argument("--sun-azimuth", type=float, default=135.0)
    parser.add_argument("--sun-elevation", type=float, default=50.0)
    parser.add_argument(
        "--tone",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=render.KHARTOUM_TONE,
        help="per-channel gain (default matches the real Khartoum chips)",
    )
    parser.add_argument(
        "--variant",
        choices=ablation.VARIANTS,
        help="an ablation preset (overrides the realism flags below)",
    )
    parser.add_argument(
        "--randomise",
        action="store_true",
        help="draw sun, haze, noise and the rest per region from default ranges "
        "(overrides the fixed values above)",
    )
    parser.add_argument("--ambient", type=float, default=render.AMBIENT, help="light in shadow")
    parser.add_argument("--contrast", type=float, default=1.0, help="stretch about mid-grey")
    parser.add_argument(
        "--bright-roofs", type=float, default=0.0, help="share of very bright roofs"
    )
    parser.add_argument("--blank-prob", type=float, default=0.0, help="chance of a no-data edge")
    parser.add_argument("--palette", type=Path, help="palette from `fraktal palette`")
    parser.add_argument("--objects", type=float, default=0.0, help="tree and vehicle density")
    parser.add_argument("--roof-detail", type=float, default=0.0, help="roof stripe strength")
    parser.add_argument("--psf", type=float, default=0.8, help="sensor blur sigma in pixels")
    parser.add_argument("--noise", type=float, default=8.0, help="sensor noise std in DN")
    parser.add_argument("--haze", type=float, default=0.0, help="atmospheric optical depth")
    parser.add_argument("--penumbra", type=float, default=0.0, help="shadow softness in metres")
    parser.add_argument(
        "--texture", type=float, default=0.7, help="sub-10 m texture strength (0 is flat)"
    )
    parser.add_argument(
        "--variation", type=float, default=0.15, help="reflectance spread (0 is flat, V1 look)"
    )


def _compare(args: argparse.Namespace) -> Result:
    try:
        real_dir = args.real or args.data_dir / "spacenet2" / "chips"
        real = chips.load_splits(real_dir)[0]
        synthetic = chips.load_all(args.synthetic or args.data_dir / "synthetic" / "chips")
        report = realism.compare(
            realism.sample([real.get(i)[0] for i in range(len(real))], args.limit),
            realism.sample([synthetic.get(i)[0] for i in range(len(synthetic))], args.limit),
        )
    except (OSError, ValueError, KeyError) as err:
        raise CliError(ExitCode.FAILURE, "compare_failed", str(err)) from err
    d = report["distances"]
    text = (
        f"colour W1 {d['colour_w1']}, edge-energy gap {d['edge_energy_gap']}, "
        f"noise gap {d['noise_gap']} (0 is a perfect match)"
    )
    return Result(report, text)


def _compare_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--real", type=Path, help="real chips dir (default spacenet2/chips)")
    parser.add_argument("--synthetic", type=Path, help="synthetic chips dir")
    parser.add_argument("--limit", type=int, default=100, help="chips sampled per side")


def _terrain_center(terrain_dir: Path) -> tuple[float, float]:
    left, bottom, right, top = json.loads((terrain_dir / terrain.MARKER).read_text())["bounds"]
    return ((left + right) / 2, (bottom + top) / 2)


def _render_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--buildings", type=Path, required=True, help="building layer (GeoParquet)")
    parser.add_argument("--center", type=float, nargs=2, metavar=("LON", "LAT"))
    parser.add_argument("--size-m", type=float, default=2000.0, help="region side in metres")
    parser.add_argument("--sun-azimuth", type=float, default=135.0)
    parser.add_argument("--sun-elevation", type=float, default=50.0)
    parser.add_argument("--out", type=Path, help="output directory (default <data-dir>/render)")


def _split(args: argparse.Namespace) -> Result:
    root = args.data_dir / "spacenet2" / "khartoum"
    out = args.out or args.data_dir / "spacenet2" / "khartoum-split.json"
    try:
        tiles, crs = split.list_tiles(root)
        manifest = split.build_manifest(tiles, crs, args.seed, args.test_fraction)
    except (OSError, ValueError) as err:
        raise CliError(ExitCode.FAILURE, "split_failed", str(err)) from err
    split.write_manifest(manifest, out)
    counts = manifest["counts"]
    text = f"wrote {out}: {counts['train']} train, {counts['test']} test (seed {args.seed})"
    return Result({"manifest": str(out), "seed": args.seed, "counts": counts}, text)


def _split_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--seed", type=int, default=0, help="split seed (default 0)")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="held-out share")
    parser.add_argument("--out", type=Path, help="manifest path")


ARGS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "ingest": _ingest_args,
    "split": _split_args,
    "render": _render_args,
    "benchmark": _benchmark_args,
    "eval-set": _eval_set_args,
    "generate": _generate_cli_args,
    "synth": _synth_args,
    "compare": _compare_args,
    "ablate": _ablate_args,
    "report": _report_args,
    "palette": _palette_args,
    "inspect": _inspect_args,
    "assets": _assets_args,
    "render3d": _render3d_args,
    "render3d-batch": _render3d_batch_args,
    "traffic": _traffic_args,
    "demo": _demo_args,
}

COMMANDS: dict[str, tuple[str, Handler]] = {
    "version": ("Print the version", _version),
    "ingest": ("Fetch and cache real source data", _ingest),
    "split": ("Split the real tiles into train and held-out test", _split),
    "generate": ("Generate a world from a seed and config", _generate),
    "synth": ("Render synthetic pretraining chips from the world", _synth),
    "render": ("Render a world to imagery and labels", _render),
    "benchmark": ("Run the downstream benchmark", _benchmark),
    "eval-set": ("Build real evaluation chips from Open Cities AI (eval only)", _eval_set),
    "ablate": ("Run the realism ablation benchmark", _ablate),
    "report": ("Build the realism report PDF from an ablation", _report),
    "palette": ("Fit render colours from the real train chips", _palette),
    "compare": ("Measure how far synthetic chips are from real imagery", _compare),
    "inspect": ("Look up a feature by ID, or summarise a world", _inspect),
    "assets": ("Check the asset manifest and its licences", _assets),
    "render3d": ("Render a world tile with Blender (3D path)", _render3d),
    "render3d-batch": ("Render many tiles of a world, resumable", _render3d_batch),
    "traffic": ("Simulate cars on a tile's roads and map their paths", _traffic),
    "demo": ("Write a one-file browser demo you can drive around in", _demo),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraktal", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"fraktal {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print a JSON result envelope")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, (help_text, _handler) in COMMANDS.items():
        command = sub.add_parser(name, help=help_text, parents=[common])
        if name in ARGS:
            ARGS[name](command)
    return parser


def _emit(envelope: dict[str, Any], text: str, *, as_json: bool, ok: bool) -> None:
    if as_json:
        print(json.dumps(envelope))
    elif ok:
        print(text)
    else:
        print(f"error: {text}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    as_json: bool = args.json
    _help, handler = COMMANDS[args.command]
    try:
        result = handler(args)
    except CliError as err:
        error = {"kind": err.kind, "message": err.message}
        _emit({"ok": False, "data": None, "error": error}, err.message, as_json=as_json, ok=False)
        return int(err.code)
    except Exception as err:  # noqa: BLE001 - last line of defence at the process boundary
        message = f"{type(err).__name__}: {err}"
        error = {"kind": "unexpected", "message": message}
        _emit({"ok": False, "data": None, "error": error}, message, as_json=as_json, ok=False)
        return int(ExitCode.FAILURE)
    _emit({"ok": True, "data": result.data, "error": None}, result.text, as_json=as_json, ok=True)
    return int(ExitCode.OK)
