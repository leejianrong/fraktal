"""Look up features in a world: one by ID (with geometry and provenance), or a summary."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import mapping

MIN_PREFIX = 6


class InspectError(ValueError):
    """Raised for a missing path, an unknown ID or an ambiguous prefix."""


def _layer_files(path: Path) -> dict[str, Path]:
    """Layer files by kind: from a world directory's manifest, or a single layer file."""
    if path.is_dir():
        manifest = path / "world.json"
        if not manifest.is_file():
            raise InspectError(f"{path} has no world.json")
        layers = json.loads(manifest.read_text())["layers"]
        return {kind: path / info["file"] for kind, info in layers.items()}
    if path.is_file():
        return {"layer": path}
    raise InspectError(f"{path} does not exist")


def _plain(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def find_feature(path: Path, feature_id: str) -> dict[str, Any]:
    """The feature with this ID (or a unique prefix of at least six characters)."""
    matches: list[tuple[str, gpd.GeoDataFrame]] = []
    for file in _layer_files(path).values():
        frame = gpd.read_parquet(file)
        if len(feature_id) >= MIN_PREFIX:
            hit = frame[frame["id"].astype(str).str.startswith(feature_id)]
        else:
            hit = frame[frame["id"] == feature_id]
        if len(hit):
            matches.append((file.name, hit))
    total = sum(len(h) for _, h in matches)
    if total == 0:
        raise InspectError(f"no feature with id {feature_id!r} in {path}")
    if total > 1:
        raise InspectError(f"{feature_id!r} matches {total} features; give a longer id")
    file_name, hit = matches[0]
    row = hit.iloc[0]
    properties = {
        k: _plain(v) for k, v in row.items() if k not in ("geometry", "id", "kind", "provenance")
    }
    return {
        "id": str(row["id"]),
        "kind": str(row["kind"]),
        "provenance": str(row["provenance"]),
        "geometry": mapping(row.geometry),
        "crs": hit.crs.to_string(),
        "properties": properties,
        "layer_file": file_name,
    }


def summarise(path: Path) -> dict[str, Any]:
    """Feature counts and provenance per layer."""
    layers: dict[str, Any] = {}
    for kind, file in _layer_files(path).items():
        frame = gpd.read_parquet(file)
        layers[str(frame["kind"].iloc[0]) if kind == "layer" and len(frame) else kind] = {
            "file": file.name,
            "features": len(frame),
            "provenance": sorted(set(map(str, frame["provenance"]))),
            "crs": frame.crs.to_string(),
        }
    return {"path": str(path), "layers": layers}
