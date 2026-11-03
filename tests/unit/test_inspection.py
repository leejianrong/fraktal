import json
from pathlib import Path

import pytest

from fraktal import cli
from fraktal.inspection import InspectError, find_feature, summarise
from fraktal.procedural import ProceduralConfig, generate_world, read_buildings

CENTER = (32.55, 15.6)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("w")
    generate_world(out, CENTER, 11, ProceduralConfig(size_m=800.0))
    return out


def test_a_known_id_returns_geometry_and_provenance(world: Path) -> None:
    building = read_buildings(world).iloc[0]
    found = find_feature(world, building.id)
    assert found["id"] == building.id and found["kind"] == "building"
    assert found["provenance"] == "procedural:buildings"
    assert found["geometry"]["type"] == "Polygon" and len(found["geometry"]["coordinates"][0]) >= 4
    assert found["properties"]["height_m"] == pytest.approx(building.height_m)
    assert found["properties"]["parcel_id"] == building.parcel_id
    assert found["layer_file"] == "buildings.parquet" and found["crs"].startswith("EPSG:")
    json.dumps(found)  # plain JSON types only


def test_ids_are_found_in_any_layer_and_by_unique_prefix(world: Path) -> None:
    from fraktal.world import read_layer

    road = read_layer(world / "roads.parquet", "road").iloc[0]
    assert find_feature(world, road.id)["kind"] == "road"
    assert find_feature(world, road.id[:8])["id"] == road.id
    with pytest.raises(InspectError, match="no feature"):
        find_feature(world, "0" * 16)
    with pytest.raises(InspectError, match="no feature"):
        find_feature(world, "abc")  # too short to be a prefix, and not an id


def test_a_single_layer_file_works_and_a_summary_lists_layers(world: Path) -> None:
    b = read_buildings(world).iloc[0]
    assert find_feature(world / "buildings.parquet", b.id)["kind"] == "building"
    summary = summarise(world)
    assert set(summary["layers"]) == {"road", "parcel", "building"}
    assert summary["layers"]["building"]["provenance"] == ["procedural:buildings"]


def test_bad_paths_are_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(InspectError, match="does not exist"):
        summarise(tmp_path / "nope")
    with pytest.raises(InspectError, match="no world.json"):
        summarise(tmp_path)


def test_inspect_command_returns_the_feature_as_json(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    building = read_buildings(world).iloc[0]
    assert cli.main(["inspect", str(world), "--id", building.id, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["data"]["provenance"] == "procedural:buildings"


def test_inspect_command_fails_cleanly_on_an_unknown_id(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", str(world), "--id", "f" * 16, "--json"]) == 4
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["kind"] == "inspect_failed" and "no feature" in err["message"]


def test_inspect_without_an_id_summarises_the_world(
    world: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["inspect", str(world), "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["data"]["layers"]) == {
        "road",
        "parcel",
        "building",
    }
