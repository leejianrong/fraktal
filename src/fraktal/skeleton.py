"""The V1 world: real SpaceNet building footprints from the training tiles only (ADR-0006).

Footprints that touch a held-out test tile are dropped, and ``assert_no_test_leak`` then proves
none remain. Heights are not in the labels, so they are drawn from a named substream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box

from fraktal.chips import LABEL_DIR
from fraktal.ids import substream
from fraktal.split import assert_no_test_leak, read_manifest
from fraktal.world import make_layer

WORLD_ID = "khartoum-skeleton"
SOURCE = "real:spacenet2-khartoum"
HEIGHT_RANGE_M = (4.0, 12.0)
UTM_CRS = "EPSG:32636"


def build_skeleton(data_dir: Path, manifest_path: Path, seed: int) -> gpd.GeoDataFrame:
    manifest = read_manifest(manifest_path)
    label_dir = data_dir / "spacenet2" / "khartoum" / LABEL_DIR
    frames = []
    for tile_id in sorted(manifest["train"]):
        path = label_dir / f"buildings_AOI_5_Khartoum_{tile_id}.geojson"
        if path.is_file():
            frames.append(gpd.read_file(path)[["geometry"]].to_crs("EPSG:4326"))
    if not frames:
        raise FileNotFoundError(f"no building labels under {label_dir}; run `fraktal ingest`")
    geoms = pd.concat(frames, ignore_index=True).geometry
    geoms = gpd.GeoSeries(shapely.make_valid(geoms.to_numpy()), crs="EPSG:4326")
    geoms = geoms[geoms.geom_type.isin(["Polygon", "MultiPolygon"]) & ~geoms.is_empty]

    test_boxes = [box(*t["bounds"]) for t in manifest["test"]]
    tree = shapely.STRtree(test_boxes)
    hits = tree.query(geoms.to_numpy(), predicate="intersects")
    overlapping = {
        int(i) for i, j in zip(*hits, strict=True) if not geoms.iloc[i].touches(test_boxes[j])
    }
    keep = geoms.iloc[[i for i in range(len(geoms)) if i not in overlapping]]
    assert_no_test_leak(keep, manifest)

    local = keep.to_crs(UTM_CRS)
    centroids = local.centroid
    keys = pd.DataFrame(
        {"x": centroids.x.round(3).to_numpy(), "y": centroids.y.round(3).to_numpy()}
    )
    first = ~keys.duplicated()
    local = local[first.to_numpy()]
    layer = make_layer(
        "building",
        WORLD_ID,
        [{"geometry": g, "provenance": SOURCE} for g in local],
        UTM_CRS,
    )
    low, high = HEIGHT_RANGE_M
    layer["height_m"] = substream(seed, "building-heights").uniform(low, high, len(layer))
    layer.attrs["dropped_overlapping_test_tiles"] = len(overlapping)
    return layer


def describe(layer: gpd.GeoDataFrame) -> dict[str, Any]:
    return {
        "buildings": len(layer),
        "dropped_overlapping_test_tiles": layer.attrs.get("dropped_overlapping_test_tiles", 0),
        "crs": str(layer.crs),
    }


def held_out_footprints(manifest_path: Path, crs: str) -> list[Any]:
    """Held-out test tile footprints in ``crs``."""
    boxes = [box(*t["bounds"]) for t in read_manifest(manifest_path)["test"]]
    return list(gpd.GeoSeries(boxes, crs="EPSG:4326").to_crs(crs))


def train_coverage(manifest_path: Path, crs: str) -> Any:
    """Union of the training tile footprints in ``crs``: the only area whose labels we have."""
    manifest = read_manifest(manifest_path)
    train = set(manifest["train"])
    boxes = [box(*t["bounds"]) for t in manifest["tile_bounds"] if t["id"] in train]
    projected = gpd.GeoSeries(boxes, crs="EPSG:4326").to_crs(crs)
    return shapely.union_all(projected.to_numpy())
