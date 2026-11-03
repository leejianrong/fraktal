import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from fraktal import biomes, cli
from fraktal.biomes import BiomeError, resolve
from fraktal.procedural import DEFAULT_BIOME, ProceduralConfig, generate_world

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
