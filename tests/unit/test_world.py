import hashlib
from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point, Polygon

from fraktal.ids import substream
from fraktal.world import (
    WorldError,
    make_layer,
    read_layer,
    validate_layer,
    world_id,
    write_layer,
)

CRS = "EPSG:32636"


def square(x: float, y: float, size: float = 10.0) -> Polygon:
    return Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])


def building(x: float, y: float, **attrs: Any) -> dict[str, Any]:
    return {"geometry": square(x, y), "provenance": "real:spacenet2", **attrs}


def layer(world: str = "w1") -> gpd.GeoDataFrame:
    return make_layer("building", world, [building(0, 0, height_m=6.0), building(50, 50)], CRS)


def test_world_id_is_stable_and_sensitive_to_each_input() -> None:
    base = world_id(1, {"a": 1, "b": 2}, "0.1.0")
    assert base == world_id(1, {"b": 2, "a": 1}, "0.1.0")
    assert base != world_id(2, {"a": 1, "b": 2}, "0.1.0")
    assert base != world_id(1, {"a": 1, "b": 3}, "0.1.0")
    assert base != world_id(1, {"a": 1, "b": 2}, "0.2.0")


def test_ids_do_not_depend_on_feature_order() -> None:
    a = make_layer("building", "w1", [building(0, 0), building(50, 50)], CRS)
    b = make_layer("building", "w1", [building(50, 50), building(0, 0)], CRS)
    assert_geodataframe_equal(a, b)


def test_ids_depend_on_world() -> None:
    assert set(layer("w1")["id"]).isdisjoint(layer("w2")["id"])


def test_optional_fields_may_be_absent_or_null() -> None:
    frame = layer()
    assert frame["height_m"].isna().sum() == 1
    validate_layer(frame, "building")


def test_duplicate_features_are_rejected() -> None:
    with pytest.raises(WorldError, match="duplicate"):
        make_layer("building", "w1", [building(0, 0), building(0, 0)], CRS)


@pytest.mark.parametrize("provenance", ["osm", "real:", "synthetic:x", "real:UPPER", ""])
def test_invalid_provenance_is_rejected(provenance: str) -> None:
    feature = {"geometry": square(0, 0), "provenance": provenance}
    with pytest.raises(WorldError, match="provenance"):
        make_layer("building", "w1", [feature], CRS)


@pytest.mark.parametrize("provenance", ["real:osm", "procedural:grid-v1", "real:esa.worldcover"])
def test_valid_provenance_is_accepted(provenance: str) -> None:
    make_layer("building", "w1", [{"geometry": square(0, 0), "provenance": provenance}], CRS)


def test_unknown_column_is_rejected() -> None:
    with pytest.raises(WorldError, match="unknown columns"):
        make_layer("building", "w1", [building(0, 0, colour="red")], CRS)


def test_wrong_geometry_type_is_rejected() -> None:
    feature = {"geometry": Point(1, 1), "provenance": "real:osm"}
    with pytest.raises(WorldError, match="geometry types"):
        make_layer("building", "w1", [feature], CRS)


def test_invalid_geometry_is_rejected() -> None:
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    with pytest.raises(WorldError, match="invalid geometry"):
        make_layer("building", "w1", [{"geometry": bowtie, "provenance": "real:osm"}], CRS)


def test_missing_crs_is_rejected() -> None:
    frame = layer().set_crs(None, allow_override=True)
    with pytest.raises(WorldError, match="CRS"):
        validate_layer(frame, "building")


def test_wrong_attribute_type_is_rejected() -> None:
    with pytest.raises(WorldError, match="height_m"):
        make_layer("building", "w1", [building(0, 0, height_m="tall")], CRS)


def test_landcover_layer_has_its_own_fields() -> None:
    feature = {"geometry": square(0, 0), "provenance": "real:worldcover", "landcover_class": "tree"}
    frame = make_layer("landcover", "w1", [feature], CRS)
    assert list(frame["kind"]) == ["landcover"]
    with pytest.raises(WorldError, match="unknown columns"):
        make_layer("landcover", "w1", [building(0, 0, height_m=3.0)], CRS)


def test_round_trip_preserves_features(tmp_path: Path) -> None:
    original = layer()
    write_layer(original, "building", tmp_path / "b.parquet")
    assert_geodataframe_equal(read_layer(tmp_path / "b.parquet", "building"), original)


def generate(seed: int, path: Path) -> str:
    """A tiny stand-in generator: random squares drawn from a named sub-stream."""
    rng = substream(seed, "buildings")
    xs = rng.uniform(0, 1000, size=20)
    ys = rng.uniform(0, 1000, size=20)
    features = [building(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
    w = world_id(seed, {"n": 20}, "0.0.1")
    write_layer(make_layer("building", w, features, CRS), "building", path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_gives_identical_file_hashes(tmp_path: Path) -> None:
    assert generate(42, tmp_path / "a.parquet") == generate(42, tmp_path / "b.parquet")


def test_different_seed_gives_different_file(tmp_path: Path) -> None:
    assert generate(42, tmp_path / "a.parquet") != generate(43, tmp_path / "b.parquet")
