import json
import textwrap
from pathlib import Path

import pytest
import rasterio

from fraktal import biomes, render3d
from fraktal.cli import main
from fraktal.procedural import generate_world

CENTER = (104.0, 1.5)

FAKE_BLENDER = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import rasterio
    scene = json.load(open(sys.argv[1] + "/scene.json"))
    px = scene["px"]
    nb, nt = len(scene["buildings"]), len(scene["trees"])
    print("STAGE scene_built buildings=%d trees=%d seconds=0.1" % (nb, nt))
    rgb = np.full((3, px, px), 90, dtype="uint8")
    kw = dict(driver="PNG", count=3, dtype="uint8", height=px, width=px)
    with rasterio.open(sys.argv[2], "w", **kw) as d:
        d.write(rgb)
    print(json.dumps({"build_s": 0.1, "render_s": 0.2, "px": px, "samples": 4,
                      "buildings": nb, "trees": nt, "bpy": "fake"}))
    """
)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("w") / "world"
    generate_world(path, CENTER, 3, biomes.resolve("dense_urban"))
    return path


def fake(tmp_path: Path, body: str = FAKE_BLENDER) -> Path:
    script = tmp_path / "fake_blender.py"
    script.write_text(body)
    return script


def test_the_contract_files_are_written(world: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    meta = render3d.render(world, out, CENTER, 100.0, 1.0, seed=2, script=fake(tmp_path))
    for name in ("scene/scene.json", "scene/ground.npy", "image.png", "tile.tif", "render.json"):
        assert (out / name).is_file(), name
    assert json.loads((out / "render.json").read_text()) == meta
    assert meta["px"] == 100 and meta["seed"] == 2 and meta["bpy"] == "fake"
    assert meta["scene_sha256"] == render3d.scene_sha256(out / "scene")
    with rasterio.open(out / "tile.tif") as tif:
        assert tif.count == 3 and tif.width == 100 and str(tif.crs).startswith("EPSG:")
        assert tif.transform.a == 1.0 and tif.transform.e == -1.0
        assert float(tif.read().mean()) == 90.0


def test_the_same_seed_gives_the_same_scene_and_another_seed_does_not(
    world: Path, tmp_path: Path
) -> None:
    script = fake(tmp_path)
    a = render3d.render(world, tmp_path / "a", CENTER, 150.0, 1.0, seed=1, script=script)
    b = render3d.render(world, tmp_path / "b", CENTER, 150.0, 1.0, seed=1, script=script)
    c = render3d.render(world, tmp_path / "c", CENTER, 150.0, 1.0, seed=2, script=script)
    assert a["scene_sha256"] == b["scene_sha256"]
    for name in ("scene.json", "ground.npy"):
        one, two = tmp_path / "a/scene" / name, tmp_path / "b/scene" / name
        assert one.read_bytes() == two.read_bytes()
    assert a["scene_sha256"] != c["scene_sha256"]  # the seed moves the trees and the ground texture


def test_a_hung_blender_is_killed(world: Path, tmp_path: Path) -> None:
    script = fake(tmp_path, "import time\nprint('STAGE hang', flush=True)\ntime.sleep(60)\n")
    with pytest.raises(render3d.RenderError, match="timed out"):
        render3d.render(world, tmp_path / "o", CENTER, 50.0, 1.0, timeout_s=2.0, script=script)


def test_a_failing_blender_reports_its_last_output(world: Path, tmp_path: Path) -> None:
    script = fake(tmp_path, "import sys\nprint('Traceback boom')\nsys.exit(3)\n")
    with pytest.raises(render3d.RenderError, match=r"(?s)exited with 3.*Traceback boom"):
        render3d.render(world, tmp_path / "o", CENTER, 50.0, 1.0, script=script)


def test_the_cli_reports_a_failure_without_a_traceback(
    world: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["render3d", str(world), "--out", str(tmp_path / "o"), "--center", "104", "1.5",
         "--size-m", "50", "--gsd-m", "1", "--bpy-python", "/nonexistent/python"]
    )  # fmt: skip
    assert code != 0 and "render3d_failed" not in capsys.readouterr().out
