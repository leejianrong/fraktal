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

from fraktal import __version__, benchmark, chips, ingest, render, split, terrain, world


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


def _benchmark(args: argparse.Namespace) -> Result:
    manifest = args.split or args.data_dir / "spacenet2" / "khartoum-split.json"
    out = args.out or args.data_dir / "benchmark" / "report.json"
    try:
        chips_dir = args.chips or chips.build_chips(args.data_dir, manifest)
        train_set, test_set = chips.load_splits(chips_dir)
        recipe = benchmark.Recipe(steps=args.steps) if args.steps else benchmark.Recipe()
        report = benchmark.run_real_arms(train_set, test_set, recipe, args.seed, args.device)
    except (OSError, ValueError, RuntimeError) as err:
        raise CliError(ExitCode.FAILURE, "benchmark_failed", str(err)) from err
    benchmark.write_report(report, out)
    iou = {name: round(arm["iou"], 4) for name, arm in report["arms"].items()}
    return Result({"report": str(out), "iou": iou}, f"wrote {out}: IoU {iou}")


def _benchmark_args(parser: argparse.ArgumentParser) -> None:
    _data_dir_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, help="optimiser steps per arm (default: the recipe's)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--split", type=Path, help="split manifest from `fraktal split`")
    parser.add_argument("--chips", type=Path, help="prebuilt chips directory (skips building)")
    parser.add_argument("--out", type=Path, help="report path")


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
}

COMMANDS: dict[str, tuple[str, Handler]] = {
    "version": ("Print the version", _version),
    "ingest": ("Fetch and cache real source data", _ingest),
    "split": ("Split the real tiles into train and held-out test", _split),
    "generate": ("Generate a world from a seed and config", _not_implemented("generate")),
    "render": ("Render a world to imagery and labels", _render),
    "benchmark": ("Run the downstream benchmark", _benchmark),
    "inspect": ("Inspect a world, dataset or report", _not_implemented("inspect")),
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
