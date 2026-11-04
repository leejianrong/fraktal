import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import Point

from fraktal import biomes, cli, world
from fraktal.biomes import BiomeError, resolve
from fraktal.procedural import (
    DEFAULT_BIOME,
    ProceduralConfig,
    generate_world,
    read_buildings,
    read_roads,
)

CENTER = (32.55, 15.64)
SMALL = {"size_m": 600.0, "gsd_m": 5.0}


def test_the_default_biome_is_the_plain_config_and_unknown_names_fail_loudly() -> None:
    assert resolve(DEFAULT_BIOME) == ProceduralConfig()
    assert DEFAULT_BIOME in biomes.names()
    with pytest.raises(BiomeError, match="unknown biome 'atlantis'.*khartoum"):
        resolve("atlantis")


def test_overrides_apply_to_nested_settings_and_keep_their_types() -> None:
    config = resolve(DEFAULT_BIOME, {"aridity": 1, "roads": {"local_spacing_m": 60}})
    assert config.aridity == 1.0 and isinstance(config.aridity, float)
    assert config.roads.local_spacing_m == 60.0
    assert config.roads.arterial_width_m == ProceduralConfig().roads.arterial_width_m
    tuple_set = resolve(DEFAULT_BIOME, {"parcels": {"frontage_m": [10, 20]}})
    assert tuple_set.parcels.frontage_m == (10.0, 20.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"aridty": 0.3}, "unknown biome setting 'aridty'"),
        ({"roads": {"local_spacin_m": 5}}, "'roads.local_spacin_m'"),
        ({"biome": "other"}, "unknown biome setting 'biome'"),
        ({"aridity": "dry"}, "aridity must be a number"),
        ({"roads": 3}, "roads must be an object"),
        ({"parcels": {"frontage_m": [1]}}, "list of 2 numbers"),
        ({"buildings": {"core_radius_m": True}}, "must be a number"),
    ],
)
def test_bad_settings_fail_with_the_key_named(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(BiomeError, match=message):
        resolve(DEFAULT_BIOME, overrides)


def digests(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_the_default_biome_world_is_unchanged_from_before_biomes_existed(tmp_path: Path) -> None:
    manifest = generate_world(tmp_path, CENTER, 7, resolve(DEFAULT_BIOME, SMALL))
    assert manifest["world_id"] == "1d6ca3c86f2729a6"  # pinned before this feature
    layers = {k: v["sha256"][:12] for k, v in manifest["layers"].items()}
    assert layers == {"road": "a6b753815513", "parcel": "d78ecedec9e2", "building": "8e572e61c253"}
    rasters = {k: v["sha256"][:12] for k, v in manifest["rasters"].items()}
    assert rasters == {"dem": "ecf34431b918", "landcover": "f498169bb554"}
    assert manifest["biome"] == DEFAULT_BIOME
    assert json.loads((tmp_path / "world.json").read_text())["biome"] == DEFAULT_BIOME


def test_a_registered_biome_is_routed_recorded_and_changes_the_world_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preset = dataclasses.replace(resolve(DEFAULT_BIOME, {"aridity": 0.9}), biome="dusty")
    monkeypatch.setitem(biomes.BIOMES, "dusty", biomes.Biome("dusty", "test", preset))
    config = resolve("dusty", SMALL)
    manifest = generate_world(tmp_path, CENTER, 7, config)
    assert manifest["biome"] == "dusty" and manifest["world_id"] != "1d6ca3c86f2729a6"


def test_generate_cli_takes_a_biome_and_rejects_a_bad_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"roads": {"local_spacing_m": 70}}))
    out = tmp_path / "w"
    code = cli.main(
        ["generate", "--fallback", "procedural", "--center", "32.55", "15.64", "--size-m", "600",
         "--biome", "khartoum", "--biome-config", str(good), "--out", str(out), "--json"]
    )  # fmt: skip
    assert code == 0 and json.loads((out / "world.json").read_text())["biome"] == "khartoum"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"aridty": 1}))
    code = cli.main(
        ["generate", "--fallback", "procedural", "--center", "32.55", "15.64",
         "--biome-config", str(bad), "--out", str(tmp_path / "x")]
    )  # fmt: skip
    assert code == 2 and not (tmp_path / "x").exists()
    assert "aridty" in capsys.readouterr().err


URBAN_CENTER = (100.50, 13.75)


def _urban_world(path: Path, seed: int) -> Any:
    config = resolve("dense_urban", {"size_m": 1200.0})
    return generate_world(path, URBAN_CENTER, seed, config)


def test_dense_urban_is_denser_taller_and_gridded(tmp_path: Path) -> None:
    arid = resolve(DEFAULT_BIOME, {"size_m": 1200.0})
    generate_world(tmp_path / "dry", URBAN_CENTER, 3, arid)
    _urban_world(tmp_path / "urban", 3)
    dry_b, urban_b = read_buildings(tmp_path / "dry"), read_buildings(tmp_path / "urban")
    area = 1200.0**2
    urban_cover = float(urban_b.geometry.area.sum() / area)
    assert 0.08 <= urban_cover <= 0.30  # configured range for a tropical core
    assert urban_cover > 3 * float(dry_b.geometry.area.sum() / area)
    assert 18.0 <= float(urban_b["height_m"].mean()) <= 40.0
    roads = read_roads(tmp_path / "urban")
    arterials = roads[roads["road_class"] == "arterial"]
    streets = roads[roads["road_class"] == "local"]
    square = 0
    for street in streets.geometry:
        c = list(street.coords)
        hook = Point(c[0])
        art = min(arterials.geometry, key=hook.distance)
        t = art.project(hook)
        a, b = art.interpolate(max(t - 8, 0)), art.interpolate(min(t + 8, art.length))
        turn = math.degrees(math.atan2(c[1][1] - c[0][1], c[1][0] - c[0][0]))
        turn -= math.degrees(math.atan2(b.y - a.y, b.x - a.x))
        square += abs((turn + 180) % 180 - 90) <= 25
    assert square / len(streets) >= 0.8  # streets leave the arterials at right angles


def test_dense_urban_is_byte_identical_per_seed(tmp_path: Path) -> None:
    one, two = _urban_world(tmp_path / "a", 5), _urban_world(tmp_path / "b", 5)
    assert one["world_id"] == two["world_id"]
    assert one["layers"] == two["layers"] and one["rasters"] == two["rasters"]
    assert one["biome"] == "dense_urban"


def _informal_world(path: Path, seed: int) -> Any:
    config = resolve("informal_settlement", {"size_m": 1200.0})
    return generate_world(path, URBAN_CENTER, seed, config)


def test_informal_settlement_has_small_close_buildings_with_tin_and_tile_roofs(
    tmp_path: Path,
) -> None:
    _informal_world(tmp_path, 3)
    b = read_buildings(tmp_path)
    assert 25.0 <= float(b.geometry.area.mean()) <= 90.0
    pts = [(g.centroid.x, g.centroid.y) for g in b.geometry]
    near = [
        min(math.hypot(x - u, y - v) for j, (u, v) in enumerate(pts) if j != i)
        for i, (x, y) in enumerate(pts[:300])
    ]
    assert 6.0 <= sum(near) / len(near) <= 16.0
    assert set(b["roof_material"]) == {"tin", "tile"}
    tin = float((b["roof_material"] == "tin").mean())
    assert 0.5 <= tin <= 0.8
    roads = read_roads(tmp_path)
    assert float(roads["width_m"].max()) <= 5.0  # footpaths and lanes, no streets


def test_informal_parcels_are_irregular_and_the_default_has_no_material_column(
    tmp_path: Path,
) -> None:
    _informal_world(tmp_path / "i", 3)
    generate_world(tmp_path / "d", URBAN_CENTER, 3, resolve(DEFAULT_BIOME, {"size_m": 1200.0}))
    parcels = world.read_layer(tmp_path / "i" / "parcels.parquet", "parcel")
    slanted = sum(
        abs(p.minimum_rotated_rectangle.area - p.area) > 0.02 * p.area for p in parcels.geometry
    )
    assert slanted > 0.5 * len(parcels)
    assert "roof_material" not in read_buildings(tmp_path / "d")


def test_informal_settlement_is_byte_identical_per_seed(tmp_path: Path) -> None:
    one, two = _informal_world(tmp_path / "a", 5), _informal_world(tmp_path / "b", 5)
    assert one["world_id"] == two["world_id"] and one["layers"] == two["layers"]
    assert one["biome"] == "informal_settlement"
