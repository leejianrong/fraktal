"""The layer content hash follows the data, not the parquet file around it."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from fraktal import world
from fraktal.procedural import ProceduralConfig, generate_world

CRS = "EPSG:32636"


def roads(widths: tuple[float, float] = (6.0, 9.0), shift: float = 0.0) -> gpd.GeoDataFrame:
    features = [
        {
            "geometry": LineString([(0 + shift, 0), (100, 0)]),
            "provenance": "procedural:roads",
            "road_class": "arterial",
            "width_m": widths[0],
        },
        {
            "geometry": LineString([(0, 50), (100, 80)]),
            "provenance": "procedural:roads",
            "road_class": "local",
            "width_m": widths[1],
        },
    ]
    return world.make_layer("road", "w0", features, CRS)


def test_the_same_data_gives_the_same_hash_whatever_the_row_order() -> None:
    frame = roads()
    assert world.layer_content_sha256(frame, "road") == world.layer_content_sha256(
        frame.iloc[::-1], "road"
    )


@pytest.mark.parametrize(
    "changed",
    [roads(widths=(6.5, 9.0)), roads(shift=0.5)],
    ids=["a value", "a vertex"],
)
def test_a_changed_value_or_vertex_changes_the_hash(changed: gpd.GeoDataFrame) -> None:
    assert world.layer_content_sha256(changed, "road") != world.layer_content_sha256(
        roads(), "road"
    )


def test_a_missing_value_is_not_the_same_as_a_zero() -> None:
    frame = roads()
    holes = frame.copy()
    holes.loc[holes.index[0], "width_m"] = float("nan")
    zeros = frame.copy()
    zeros.loc[zeros.index[0], "width_m"] = 0.0
    assert world.layer_content_sha256(holes, "road") != world.layer_content_sha256(zeros, "road")


def test_the_crs_is_part_of_the_content() -> None:
    frame = roads()
    other = frame.set_crs("EPSG:32637", allow_override=True)
    assert world.layer_content_sha256(frame, "road") != world.layer_content_sha256(other, "road")


def test_the_hash_ignores_how_the_file_is_written_and_survives_a_round_trip(tmp_path: Path) -> None:
    frame = roads()
    one, two = tmp_path / "a.parquet", tmp_path / "b.parquet"
    world.write_layer(frame, "road", one)
    ordered = frame.sort_values("id").reset_index(drop=True)
    ordered.to_parquet(two, index=False, compression="snappy")  # other bytes, same data
    assert one.read_bytes() != two.read_bytes()
    expected = world.layer_content_sha256(frame, "road")
    assert world.layer_content_sha256(world.read_layer(one, "road"), "road") == expected
    assert world.layer_content_sha256(world.read_layer(two, "road"), "road") == expected


def test_the_manifest_records_the_content_hash_of_the_file_it_describes(tmp_path: Path) -> None:
    config = ProceduralConfig(size_m=600.0, gsd_m=5.0)
    manifest = generate_world(tmp_path, (32.55, 15.64), 7, config)
    for kind, entry in manifest["layers"].items():
        stored = world.read_layer(tmp_path / entry["file"], kind)
        assert entry["content_sha256"] == world.layer_content_sha256(stored, kind)
        assert entry["content_sha256"] != entry["sha256"]
