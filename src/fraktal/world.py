"""World model v0: feature layers with provenance and stable IDs (ADR-0003, ADR-0002).

A world is a set of vector layers. Each layer is a GeoDataFrame persisted as GeoParquet.
Every feature has a hashed ``id``, a ``kind``, and a ``provenance`` that says whether it came
from real data (``real:<source>``) or a generator (``procedural:<generator>``). Optional
columns exist now so later consumers (simulation, drone views) need no schema change.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from fraktal.ids import feature_id


class WorldError(ValueError):
    """A layer or world definition that breaks the world-model contract."""


PROVENANCE = re.compile(r"^(real|procedural):[a-z0-9][a-z0-9_.-]*$")
REQUIRED = ("id", "kind", "provenance", "geometry")


@dataclass(frozen=True)
class LayerSchema:
    geometry_types: frozenset[str]
    optional: Mapping[str, type]  # column name -> float or str


LAYERS: dict[str, LayerSchema] = {
    "building": LayerSchema(
        frozenset({"Polygon", "MultiPolygon"}),
        {"height_m": float, "land_use": str, "roof_type": str},
    ),
    "road": LayerSchema(
        frozenset({"LineString", "MultiLineString"}),
        {"road_class": str, "width_m": float, "bridge": float},
    ),
    "landcover": LayerSchema(
        frozenset({"Polygon", "MultiPolygon"}),
        {"landcover_class": str},
    ),
}


def world_id(seed: int, config: Mapping[str, Any], generator_version: str) -> str:
    """Identity of a world: a hash of (seed, config, generator_version)."""
    canonical = json.dumps(
        {"seed": seed, "config": config, "generator_version": generator_version},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def make_layer(
    kind: str, world: str, features: Iterable[Mapping[str, Any]], crs: str
) -> gpd.GeoDataFrame:
    """Build a validated layer. IDs derive from each geometry's centroid, not from order."""
    if kind not in LAYERS:
        raise WorldError(f"unknown layer kind {kind!r}")
    rows: list[dict[str, Any]] = []
    for feature in features:
        geom = feature["geometry"]
        centroid = geom.centroid
        row = dict(feature)
        row["kind"] = kind
        row["id"] = feature_id(world, kind, f"{centroid.x:.3f}", f"{centroid.y:.3f}")
        rows.append(row)
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if frame.empty:
        raise WorldError("a layer needs at least one feature")
    frame = frame.sort_values("id").reset_index(drop=True)
    validate_layer(frame, kind)
    return frame


def validate_layer(frame: gpd.GeoDataFrame, kind: str) -> None:
    """Raise WorldError if the layer breaks the contract."""
    if kind not in LAYERS:
        raise WorldError(f"unknown layer kind {kind!r}")
    schema = LAYERS[kind]
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise WorldError(f"missing required columns: {missing}")
    unknown = [c for c in frame.columns if c not in REQUIRED and c not in schema.optional]
    if unknown:
        raise WorldError(f"unknown columns for {kind!r}: {unknown}")
    if frame.crs is None:
        raise WorldError("layer has no CRS")
    if (frame["kind"] != kind).any():
        raise WorldError(f"every feature must have kind {kind!r}")
    if frame["id"].duplicated().any():
        raise WorldError("duplicate feature ids")
    bad = frame.loc[~frame["provenance"].astype(str).str.match(PROVENANCE), "provenance"]
    if not bad.empty:
        raise WorldError(f"invalid provenance values: {sorted(set(map(str, bad)))}")
    geometry = frame.geometry
    if geometry.isna().any() or geometry.is_empty.any():
        raise WorldError("null or empty geometry")
    wrong = set(geometry.geom_type) - schema.geometry_types
    if wrong:
        raise WorldError(f"geometry types not allowed for {kind!r}: {sorted(wrong)}")
    if not geometry.is_valid.all():
        raise WorldError("invalid geometry")
    for column, expected in schema.optional.items():
        if column in frame.columns:
            _check_column_type(frame[column], column, expected)


def _check_column_type(series: pd.Series, column: str, expected: type) -> None:
    values = series.dropna()
    if expected is float:
        ok = pd.api.types.is_numeric_dtype(values) or values.empty
    else:
        ok = all(isinstance(v, str) for v in values)
    if not ok:
        raise WorldError(f"column {column!r} must hold {expected.__name__} values")


def write_layer(frame: gpd.GeoDataFrame, kind: str, path: Path) -> None:
    """Write a layer as GeoParquet, sorted by id so identical worlds give identical bytes."""
    validate_layer(frame, kind)
    ordered = frame.sort_values("id").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_parquet(path, index=False, compression="zstd")


def read_layer(path: Path, kind: str) -> gpd.GeoDataFrame:
    frame = gpd.read_parquet(path)
    validate_layer(frame, kind)
    return frame
