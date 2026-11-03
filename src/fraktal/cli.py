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

from fraktal import __version__, ingest


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
    try:
        result = ingest.ingest(args.data_dir, keep_archive=args.keep_archive)
    except OSError as err:
        raise CliError(ExitCode.FAILURE, "ingest_failed", str(err)) from err
    data = {
        "dataset": ingest.DATASET,
        "status": result.status,
        "path": str(result.path),
        "archive_bytes": result.bytes,
        "resumed_from": result.resumed_from,
    }
    return Result(data, f"{result.status} {result.path}")


def _ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FRAKTAL_DATA_DIR", "data")),
        help="cache directory (default: $FRAKTAL_DATA_DIR or ./data, which git ignores)",
    )
    parser.add_argument(
        "--keep-archive", action="store_true", help="keep the tarball after extraction"
    )


ARGS: dict[str, Callable[[argparse.ArgumentParser], None]] = {"ingest": _ingest_args}

COMMANDS: dict[str, tuple[str, Handler]] = {
    "version": ("Print the version", _version),
    "ingest": ("Fetch and cache real source data", _ingest),
    "generate": ("Generate a world from a seed and config", _not_implemented("generate")),
    "render": ("Render a world to imagery and labels", _not_implemented("render")),
    "benchmark": ("Run the downstream benchmark", _not_implemented("benchmark")),
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
