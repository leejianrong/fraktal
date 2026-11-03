import hashlib
import json
from pathlib import Path

import pytest
import rasterio

from fraktal import cli
from fraktal.procedural import MANIFEST, ProceduralConfig, generate_world, read_buildings
from fraktal.render import RenderConfig, render_region
from fraktal.world import read_layer

CENTER = (32.55, 15.6)
SMALL = ProceduralConfig(size_m=800.0, gsd_m=5.0)


def digests(directory: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


def test_two_runs_of_a_seed_are_byte_identical(tmp_path: Path) -> None:
    a = generate_world(tmp_path / "a", CENTER, 42, SMALL)
    b = generate_world(tmp_path / "b", CENTER, 42, SMALL)
    assert a == b
    assert digests(tmp_path / "a") == digests(tmp_path / "b")
    generate_world(tmp_path / "c", CENTER, 43, SMALL)
    assert digests(tmp_path / "a")["roads.parquet"] != digests(tmp_path / "c")["roads.parquet"]


def test_a_world_without_real_data_is_procedural_everywhere(tmp_path: Path) -> None:
    manifest = generate_world(tmp_path, CENTER, 7, SMALL)
    assert manifest["layers"]["road"]["provenance"] == ["procedural:roads"]
    assert manifest["layers"]["building"]["provenance"] == ["procedural:buildings"]
    assert manifest["layers"]["parcel"]["provenance"] == ["procedural:parcels"]
    assert manifest["rasters"]["dem"]["origin"] == "procedural:fractal"
    assert manifest["layers"]["building"]["features"] > 20
    on_disk = json.loads((tmp_path / MANIFEST).read_text())
    assert on_disk == manifest and on_disk["world_id"] and on_disk["seed"] == 7
    for kind, name in (("road", "roads"), ("parcel", "parcels"), ("building", "buildings")):
        assert (
            len(read_layer(tmp_path / f"{name}.parquet", kind))
            == manifest["layers"][kind]["features"]
        )


def test_the_existing_renderer_renders_a_procedural_world(tmp_path: Path) -> None:
    world_dir = tmp_path / "world"
    generate_world(world_dir, CENTER, 5, SMALL)
    buildings = read_buildings(world_dir)
    cfg = RenderConfig(gsd_m=2.0, tile_px=100, size_m=400.0, seed=5, variation=0.15)
    out = render_region(world_dir, buildings, CENTER, tmp_path / "render", cfg)
    assert out["tiles"] and out["dropped"] == []
    assert sum(t["building_pixels"] for t in out["tiles"]) > 0
    with rasterio.open(tmp_path / "render" / "tiles" / f"{out['tiles'][0]['tile']}.tif") as src:
        assert src.count == 3


def test_real_terrain_is_kept_where_supplied(tmp_path: Path) -> None:
    from test_render import terrain_at

    real = terrain_at(tmp_path / "real", CENTER, land_class=40)
    manifest = generate_world(tmp_path / "w", CENTER, 9, SMALL, real_terrain=real)
    assert manifest["rasters"]["dem"]["origin"] == "real:copernicus-dem+procedural:detail"
    assert manifest["rasters"]["landcover"]["origin"] == "real:esa-worldcover+procedural:fill"
    with rasterio.open(tmp_path / "w" / "landcover.tif") as src:
        assert set(src.read(1).ravel().tolist()) == {40}  # the real class wins everywhere
    with rasterio.open(tmp_path / "w" / "dem.tif") as src:
        dem = src.read(1)
        assert 379.0 < dem.mean() < 381.0 and dem.std() > 0.05  # real 380 m plus fine detail


def test_generate_cli_builds_a_procedural_world_twice_identically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["generate", "--fallback", "procedural", "--seed", "42", "--center", "32.55", "15.6",
            "--size-m", "800", "--json"]  # fmt: skip
    outs = []
    for name in ("one", "two"):
        assert cli.main([*args, "--out", str(tmp_path / name)]) == 0
        outs.append(json.loads(capsys.readouterr().out))
    assert outs[0]["data"]["world_id"] == outs[1]["data"]["world_id"]
    assert digests(tmp_path / "one") == digests(tmp_path / "two")
    assert outs[0]["data"]["layers"]["road"] > 0


def test_generate_cli_needs_a_centre_for_the_procedural_fallback(tmp_path: Path) -> None:
    assert cli.main(["generate", "--fallback", "procedural", "--out", str(tmp_path / "x")]) == 2
