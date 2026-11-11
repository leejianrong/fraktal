import json
import textwrap
from pathlib import Path

import pytest

from fraktal import biomes, render3d_batch
from fraktal.cli import main
from fraktal.procedural import generate_world

CENTER = (104.0, 1.5)

FAKE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import rasterio
    scene = json.load(open(sys.argv[1] + "/scene.json"))
    px = scene["px"]
    labels = sys.argv[sys.argv.index("--labels") + 1]
    kw = dict(driver="PNG", count=3, dtype="uint8", height=px, width=px)
    for path in (sys.argv[2], labels):
        with rasterio.open(path, "w", **kw) as d:
            d.write(np.full((3, px, px), 7, dtype="uint8"))
    print(json.dumps({"build_s": 0.1, "render_s": 0.2, "px": px, "buildings": 1,
                      "trees": 1, "bpy": "fake"}))
    """
)
FAKE_FLAKY = FAKE.replace(
    "px = scene",
    "if 'tile_0001' in sys.argv[1]:\n    print('boom'); sys.exit(3)\npx = scene",
    1,
)


def fake(tmp_path: Path, body: str = FAKE) -> Path:
    script = tmp_path / "fake_blender.py"
    script.write_text(body)
    return script


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("bw") / "world"
    generate_world(path, CENTER, 3, biomes.resolve("dense_urban"))
    return path


def test_the_plan_is_deterministic_inside_the_world_and_varies_the_sun(world: Path) -> None:
    a = render3d_batch.plan(world, 6, 5, 200.0, 1.0)
    assert a == render3d_batch.plan(world, 6, 5, 200.0, 1.0)
    assert a != render3d_batch.plan(world, 6, 6, 200.0, 1.0)
    assert [s["index"] for s in a] == list(range(6)) and len({s["seed"] for s in a}) == 6
    assert len({tuple(s["sun"]) for s in a}) == 6
    for s in a:
        assert 100 <= s["sun"][0] <= 260 and 35 <= s["sun"][1] <= 70
    with pytest.raises(render3d_batch.BatchError):
        render3d_batch.plan(world, 1, 0, 50000.0, 1.0)


def test_a_low_sun_range_changes_only_the_elevation(world: Path) -> None:
    a = render3d_batch.plan(world, 4, 5, 200.0, 1.0)
    b = render3d_batch.plan(world, 4, 5, 200.0, 1.0, sun_elevation=(8.0, 20.0))
    assert [s["sun"][0] for s in a] == [s["sun"][0] for s in b]
    assert [s["seed"] for s in a] == [s["seed"] for s in b]
    assert all(8 <= s["sun"][1] <= 20 for s in b)


def test_a_batch_renders_every_tile_slims_them_and_writes_the_record(
    world: Path, tmp_path: Path
) -> None:
    specs = render3d_batch.plan(world, 3, 1, 150.0, 1.0)
    record = render3d_batch.run(world, tmp_path / "b", specs, script=fake(tmp_path))
    assert record["counts"] == {"rendered": 3, "skipped": 0, "failed": 0}
    on_disk = json.loads((tmp_path / "b" / "batch.json").read_text())
    assert on_disk["counts"] == record["counts"] and on_disk["mean_seconds_per_rendered_tile"] >= 0
    for i in range(3):
        tile = tmp_path / "b" / f"tile_{i:04d}"
        assert (tile / "image.png").is_file() and (tile / "labels.png").is_file()
        assert not (tile / "scene").exists() and not (tile / "tile.tif").exists()
        meta = json.loads((tile / "render.json").read_text())
        assert meta["spec_sha256"] == record["tiles"][i]["spec_sha256"] and meta["scene_sha256"]
    kept = render3d_batch.run(
        world, tmp_path / "k", specs[:1], script=fake(tmp_path), keep_scene=True
    )
    assert (tmp_path / "k" / "tile_0000" / "scene" / "scene.json").is_file()
    assert kept["counts"]["rendered"] == 1


def test_a_second_run_skips_finished_tiles_and_redoes_a_changed_one(
    world: Path, tmp_path: Path
) -> None:
    script = fake(tmp_path)
    specs = render3d_batch.plan(world, 3, 1, 150.0, 1.0)
    render3d_batch.run(world, tmp_path / "b", specs, script=script)
    again = render3d_batch.run(world, tmp_path / "b", specs, script=script)
    assert again["counts"] == {"rendered": 0, "skipped": 3, "failed": 0}
    (tmp_path / "b" / "tile_0001" / "image.png").unlink()  # a half-written tile is redone
    specs[2] = {**specs[2], "seed": specs[2]["seed"] + 1}  # and so is a tile whose spec changed
    third = render3d_batch.run(world, tmp_path / "b", specs, script=script)
    assert [t["status"] for t in third["tiles"]] == ["skipped", "rendered", "rendered"]


def test_a_failing_tile_is_recorded_and_the_rest_still_render(world: Path, tmp_path: Path) -> None:
    specs = render3d_batch.plan(world, 3, 1, 150.0, 1.0)
    record = render3d_batch.run(world, tmp_path / "b", specs, script=fake(tmp_path, FAKE_FLAKY))
    assert [t["status"] for t in record["tiles"]] == ["rendered", "failed", "rendered"]
    assert "boom" in record["tiles"][1]["error"]
    assert record["counts"]["failed"] == 1


def test_the_cli_exit_code_reports_failures_and_success(
    world: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["render3d-batch", str(world), "--out", str(tmp_path / "c"), "--count", "2",
         "--size-m", "150", "--gsd-m", "1", "--bpy-python", "/nonexistent/python"]
    )  # fmt: skip
    assert code == 4  # blender could not start, so both tiles fail
    capsys.readouterr()
    assert json.loads((tmp_path / "c" / "batch.json").read_text())["counts"]["failed"] == 2


def test_a_sensor_batch_draws_per_tile_and_writes_sensor_png(world: Path, tmp_path: Path) -> None:
    import numpy as np
    import rasterio

    specs = render3d_batch.plan(world, 2, 3, 150.0, 1.0, sensor=True)
    assert specs == render3d_batch.plan(world, 2, 3, 150.0, 1.0, sensor=True)
    assert "sensor" not in render3d_batch.plan(world, 1, 3, 150.0, 1.0)[0]
    a, b = (s["sensor"] for s in specs)
    assert a != b and 0.12 <= a["haze"] <= 0.35 and 0.9 <= a["contrast"] <= 1.3
    record = render3d_batch.run(world, tmp_path / "s", specs, script=fake(tmp_path))
    assert record["counts"]["rendered"] == 2
    tile = tmp_path / "s" / "tile_0000"
    meta = json.loads((tile / "render.json").read_text())
    assert meta["sensor"]["file"] == "sensor.png" and meta["sensor"]["haze"] == a["haze"]
    with rasterio.open(tile / "sensor.png") as src, rasterio.open(tile / "image.png") as raw:
        graded, plain = src.read(), raw.read()
    assert graded.shape == plain.shape and not np.array_equal(graded, plain)
    again = render3d_batch.run(world, tmp_path / "s", specs, script=fake(tmp_path))
    assert again["counts"] == {"rendered": 0, "skipped": 2, "failed": 0}
    (tile / "sensor.png").unlink()
    redo = render3d_batch.run(world, tmp_path / "s", specs, script=fake(tmp_path))
    assert redo["counts"]["rendered"] == 1


def test_a_tile_never_reaches_past_the_worlds_raster_edge(world: Path) -> None:
    """The render grid snaps outward by up to a pixel; a tile at the edge used to fail to render."""
    from fraktal.render import RenderConfig, region_grid, tile_px_for

    manifest = json.loads((world / "world.json").read_text())
    west, south, east, north = manifest["bounds"]
    size = min(east - west, north - south) - 4.0  # nearly the whole world: tiles sit at the edge
    for spec in render3d_batch.plan(world, 40, 2, size, 0.3):
        px = tile_px_for(size, 0.3)
        _, transform, (_, side) = region_grid(
            tuple(spec["center"]), RenderConfig(gsd_m=0.3, tile_px=px, size_m=size)
        )
        x0, y0 = transform.c, transform.f
        extent = side * 0.3
        assert west <= x0 and x0 + extent <= east and south <= y0 - extent and y0 <= north


def test_a_size_that_is_not_a_whole_number_of_pixels_is_one_tile_not_four() -> None:
    from fraktal.render import RenderConfig, region_grid, tile_px_for

    for size, gsd in ((200.0, 0.15), (100.0, 0.3), (300.0, 0.7), (250.0, 0.3)):
        px = tile_px_for(size, gsd)
        assert px * gsd >= size and (px - 1) * gsd < size
        _, _, (tiles, side) = region_grid(
            (32.55, 15.6), RenderConfig(gsd_m=gsd, tile_px=px, size_m=size)
        )
        assert tiles == 1 and side == px


def test_lens_effects_are_drawn_per_tile_and_each_can_be_left_out(world: Path) -> None:
    every = render3d_batch.LENS_EFFECTS
    both = render3d_batch.plan(world, 3, 4, 150.0, 1.0, sensor=True, lens=every)
    assert both == render3d_batch.plan(
        world, 3, 4, 150.0, 1.0, sensor=True, lens=render3d_batch.LENS_EFFECTS
    )
    plain = render3d_batch.plan(world, 3, 4, 150.0, 1.0, sensor=True)
    assert all("lens" not in s["sensor"] for s in plain)
    lenses = [s["sensor"]["lens"] for s in both]
    assert len({lens["vignette"] for lens in lenses}) == 3
    assert all(0.15 <= lens["vignette"] <= 0.45 and lens["jpeg_quality"] >= 55 for lens in lenses)
    only = render3d_batch.plan(world, 3, 4, 150.0, 1.0, sensor=True, lens=("vignette",))
    for a, b in zip(both, only, strict=True):
        assert b["sensor"]["lens"] == {"vignette": a["sensor"]["lens"]["vignette"],
                                       "ca_px": 0.0, "jpeg_quality": 0}  # fmt: skip
    with pytest.raises(render3d_batch.BatchError):
        render3d_batch.plan(world, 1, 4, 150.0, 1.0, sensor=True, lens=("bokeh",))


def test_tiles_of_a_field_world_land_on_planted_ground(tmp_path: Path) -> None:
    from shapely.geometry import box

    from fraktal.procedural import read_fields

    path = tmp_path / "estate"
    generate_world(path, CENTER, 3, biomes.resolve("oil_palm_plantation"))
    fields = read_fields(path)
    planted = fields[fields["field_type"] == "oil_palm"].union_all()
    specs = render3d_batch.plan(path, 8, 5, 300.0, 0.3)
    assert specs == render3d_batch.plan(path, 8, 5, 300.0, 0.3)
    manifest = json.loads((path / "world.json").read_text())
    from pyproj import Transformer

    to_world = Transformer.from_crs(4326, manifest["crs"], always_xy=True)
    for s in specs:
        x, y = to_world.transform(*s["center"])
        window = box(x - 150, y - 150, x + 150, y + 150)
        assert window.intersection(planted).area / window.area >= render3d_batch.FIELD_SHARE - 0.01
