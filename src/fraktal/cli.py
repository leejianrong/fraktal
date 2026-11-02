"""Command-line entry point. Commands land here as slices are built."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from fraktal import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraktal", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fraktal {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
