"""The asset manifest and its licence check (ADR-0010 decision 5, FRK-47).

Every file under ``assets/`` needs one row in ``assets/manifest.json``. The default library holds
only assets we own, that are CC0, or that a model with permissive output terms generated for us
(``generated-permissive``, which needs a ``generator`` block saying how to make it again). An
attribution duty (CC-BY and the like) would flow on to dataset buyers, so those are out. ``check``
returns the problems it finds, and CI fails when there are any.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
ALLOWED_LICENCES = {
    "owned": "OWNED",
    "cc0-1.0": "CC0-1.0",
    "generated-permissive": "generated-permissive",
}  # lower case spelling -> canonical
GENERATED = "generated-permissive"
GENERATOR_KEYS = ("model", "version", "licence", "prompt", "seed", "date")
REQUIRED = ("id", "file", "kind", "source", "licence", "author", "url", "sha256", "verified")
IGNORED = {MANIFEST, "README.md"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(root: Path) -> list[dict[str, Any]]:
    loaded = json.loads((root / MANIFEST).read_text())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("assets"), list):
        raise ValueError(f'{MANIFEST} must be an object with an "assets" list')
    return [r for r in loaded["assets"]]


def check(root: Path) -> list[str]:
    """Problems with the manifest and the files under ``root``; an empty list means it is clean."""
    problems: list[str] = []
    try:
        rows = _rows(root)
    except (OSError, ValueError) as err:
        return [f"cannot read {MANIFEST}: {err}"]
    seen: set[str] = set()
    listed: set[str] = set()
    for i, row in enumerate(rows):
        name = str(row.get("id", f"row {i}")) if isinstance(row, dict) else f"row {i}"
        if not isinstance(row, dict):
            problems.append(f"{name}: not an object")
            continue
        for key in REQUIRED:
            if row.get(key) in (None, ""):
                problems.append(f"{name}: missing {key}")
        if name in seen:
            problems.append(f"{name}: duplicate id")
        seen.add(name)
        licence = str(row.get("licence", "")).lower()
        if licence and licence not in ALLOWED_LICENCES:
            problems.append(f"{name}: licence {row['licence']!r} is not allowed (owned or CC0-1.0)")
        if licence == GENERATED:
            generator = row.get("generator")
            if not isinstance(generator, dict):
                problems.append(f"{name}: a generated asset needs a generator block")
            else:
                for key in GENERATOR_KEYS:
                    if generator.get(key) in (None, ""):
                        problems.append(f"{name}: generator is missing {key}")
        if row.get("verified") is not True:
            problems.append(f"{name}: licence not verified against the source's licence page")
        file = row.get("file")
        if not file:
            continue
        listed.add(str(file))
        path = root / str(file)
        if not path.is_file():
            problems.append(f"{name}: file {file} does not exist")
        elif row.get("sha256") and sha256(path) != row["sha256"]:
            problems.append(f"{name}: sha256 does not match {file} (changed after review?)")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in IGNORED and rel not in listed:
            problems.append(f"{rel}: no manifest row (every asset needs a licence row)")
    return problems


__all__ = ["ALLOWED_LICENCES", "MANIFEST", "check", "sha256"]
