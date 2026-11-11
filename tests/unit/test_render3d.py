import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio

from fraktal import biomes, render3d
from fraktal.blender_export import terrain_height
from fraktal.cli import main
from fraktal.procedural import generate_world

CENTER = (104.0, 1.5)

FAKE_BLENDER = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import rasterio
    import os
    scene = json.load(open(sys.argv[1] + "/scene.json"))
    px = scene["px"]
    if "--px" in sys.argv:
        px = int(sys.argv[sys.argv.index("--px") + 1])
    labels = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else None
    cam = sys.argv[1] + "/camera.json"
    if os.path.exists(cam):
        frames = json.load(open(cam))["frames"]
        os.makedirs(sys.argv[2], exist_ok=True)
        for f in frames:
            kw = dict(driver="PNG", count=3, dtype="uint8", height=px, width=px)
            name = "%s/frame_%04d.png" % (sys.argv[2], f["index"])
            with rasterio.open(name, "w", **kw) as d:
                d.write(np.full((3, px, px), 60, dtype="uint8"))
        if labels:
            os.makedirs(labels, exist_ok=True)
            for f in frames:
                kw = dict(driver="PNG", count=3, dtype="uint8", height=px, width=px)
                with rasterio.open("%s/frame_%04d.png" % (labels, f["index"]), "w", **kw) as d:
                    d.write(np.full((3, px, px), 1, dtype="uint8"))
        print("STAGE frame done")
        print(json.dumps({"build_s": 0.1, "render_s": 0.2, "px": px, "frames": len(frames),
                          "buildings": 1, "trees": 1, "bpy": "fake"}))
        sys.exit(0)
    nb, nt = len(scene["buildings"]), len(scene["trees"])
    print("STAGE scene_built buildings=%d trees=%d seconds=0.1" % (nb, nt))
    rgb = np.full((3, px, px), 90, dtype="uint8")
    kw = dict(driver="PNG", count=3, dtype="uint8", height=px, width=px)
    with rasterio.open(sys.argv[2], "w", **kw) as d:
        d.write(rgb)
    if labels:
        with rasterio.open(labels, "w", **kw) as d:
            d.write(np.full((3, px, px), 1, dtype="uint8"))
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
    assert (out / "labels.png").is_file() and (out / "labels.tif").is_file()
    assert meta["labels"]["classes"]["building"] == 2 and len(meta["labels"]["buildings"]) > 0
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


def test_a_camera_sequence_writes_frames_and_poses(world: Path, tmp_path: Path) -> None:
    out = tmp_path / "seq"
    meta = render3d.render(
        world, out, CENTER, 300.0, 1.0, seed=4, script=fake(tmp_path), trajectory="flight",
        camera_options={"duration_s": 2.0, "fps": 12.0}, frame_px=64,
    )  # fmt: skip
    assert len(meta["frames"]) == 24 and meta["px"] == 64 and meta["trajectory"] == "flight"
    assert [f["index"] for f in meta["frames"]] == list(range(24))
    assert {"x", "y", "z", "heading_deg", "off_nadir_deg", "fov_deg", "t_s"} <= set(
        meta["frames"][0]
    )
    assert len(list((out / "frames").glob("frame_*.png"))) == 24
    assert not (out / "tile.tif").exists()
    again = render3d.render(
        world, tmp_path / "seq2", CENTER, 300.0, 1.0, seed=4, script=fake(tmp_path),
        trajectory="flight", camera_options={"duration_s": 2.0, "fps": 12.0}, frame_px=64,
    )  # fmt: skip
    assert again["scene_sha256"] == meta["scene_sha256"] and again["frames"] == meta["frames"]
    nadir = render3d.render(
        world, tmp_path / "n", CENTER, 300.0, 1.0, seed=4, script=fake(tmp_path)
    )
    assert nadir["scene_sha256"] != meta["scene_sha256"]  # the camera path is part of the scene


def test_labels_can_be_skipped(world: Path, tmp_path: Path) -> None:
    out = tmp_path / "nolabels"
    meta = render3d.render(world, out, CENTER, 60.0, 1.0, script=fake(tmp_path), labels=False)
    assert "labels" not in meta and not (out / "labels.png").exists()


TRAFFIC_BLENDER = textwrap.dedent(
    """
    import json, os, sys
    import numpy as np
    import rasterio
    scene = sys.argv[1]
    px = int(sys.argv[sys.argv.index("--px") + 1])
    labels = sys.argv[sys.argv.index("--labels") + 1]
    frames = json.load(open(scene + "/camera.json"))["frames"]
    agents = json.load(open(scene + "/agents.json"))
    first = len(json.load(open(scene + "/scene.json"))["vehicles"])
    kw = dict(driver="PNG", count=3, dtype="uint8", height=px, width=px)
    os.makedirs(sys.argv[2], exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    for f in frames:
        with rasterio.open("%s/frame_%04d.png" % (sys.argv[2], f["index"]), "w", **kw) as d:
            d.write(np.full((3, px, px), 60, dtype="uint8"))
        img = np.zeros((3, px, px), dtype="uint8")
        img[0, 5:9, 5:9] = 4                   # class vehicle
        img[2, 5:9, 5:9] = first + 1           # the first moving car, every frame
        with rasterio.open("%s/frame_%04d.png" % (labels, f["index"]), "w", **kw) as d:
            d.write(img)
    print(json.dumps({"build_s": 0.1, "render_s": 0.2, "px": px, "frames": len(frames),
                      "buildings": 1, "trees": 1, "bpy": "fake"}))
    """
)


def test_a_render_with_traffic_writes_agents_and_tracks(world: Path, tmp_path: Path) -> None:
    out = tmp_path / "t"
    meta = render3d.render(
        world, out, CENTER, 1000.0, 0.5, seed=4, trajectory="orbit", frame_px=64,
        camera_options={"fps": 4.0, "duration_s": 1.0},
        script=fake(tmp_path, TRAFFIC_BLENDER), traffic={"cars_per_km": 40.0},
    )  # fmt: skip
    agent_file = json.loads((out / "scene" / "agents.json").read_text())
    n = len(agent_file["ids"])
    assert n > 5 and len(agent_file["x"][0]) == len(meta["frames"]) == 4
    scene = json.loads((out / "scene" / "scene.json").read_text())
    assert scene["vehicles"] == []  # moving cars replace the parked ones
    table = json.loads((out / "tracks.json").read_text())
    assert [t["instance"] for t in table["tracks"]][:2] == [1, 2]
    seen = [t for t in table["tracks"] if t["detections"]]
    assert [t["id"] for t in seen] == ["car-0000"] and len(seen[0]["detections"]) == 4
    assert seen[0]["detections"][0]["bbox"] == [5, 5, 8, 8]
    assert meta["traffic"]["cars"] == n and meta["traffic"]["first_instance"] == 0
    assert meta["labels"]["vehicles"] == n
    again = render3d.render(
        world, tmp_path / "t2", CENTER, 1000.0, 0.5, seed=4, trajectory="orbit", frame_px=64,
        camera_options={"fps": 4.0, "duration_s": 1.0},
        script=fake(tmp_path, TRAFFIC_BLENDER), traffic={"cars_per_km": 40.0},
    )  # fmt: skip
    assert again["scene_sha256"] == meta["scene_sha256"]  # the tracks are part of the scene hash
    plain = render3d.render(
        world, tmp_path / "p", CENTER, 1000.0, 0.5, seed=4, trajectory="orbit", frame_px=64,
        camera_options={"fps": 4.0, "duration_s": 1.0}, script=fake(tmp_path),
    )  # fmt: skip
    assert plain["scene_sha256"] != meta["scene_sha256"] and "traffic" not in plain


def test_people_are_written_beside_the_cars_and_listed_in_tracks(
    world: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pp"
    meta = render3d.render(
        world, out, CENTER, 1000.0, 0.5, seed=4, trajectory="orbit", frame_px=64,
        camera_options={"fps": 4.0, "duration_s": 1.0}, script=fake(tmp_path, TRAFFIC_BLENDER),
        traffic={"cars_per_km": 20.0, "people_per_km": 30.0},
    )  # fmt: skip
    walkers = json.loads((out / "scene" / "people.json").read_text())
    assert walkers["kind"] == "person" and len(walkers["ids"]) > 5
    assert meta["traffic"]["people"] == len(walkers["ids"])
    assert meta["labels"]["people"] == len(walkers["ids"])
    kinds = [t["kind"] for t in json.loads((out / "tracks.json").read_text())["tracks"]]
    assert kinds.count("person") == len(walkers["ids"]) and kinds[0] == "car"
    plain = render3d.render(
        world, tmp_path / "np", CENTER, 1000.0, 0.5, seed=4, trajectory="orbit", frame_px=64,
        camera_options={"fps": 4.0, "duration_s": 1.0}, script=fake(tmp_path, TRAFFIC_BLENDER),
        traffic={"cars_per_km": 20.0},
    )  # fmt: skip
    assert (
        not (tmp_path / "np" / "scene" / "people.json").exists()
        and "people" not in plain["traffic"]
    )
    assert plain["scene_sha256"] != meta["scene_sha256"]  # the walkers are part of the scene


def _ride(
    world: Path, tmp_path: Path, name: str, follow: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    return render3d.render(
        world, tmp_path / name, CENTER, 1000.0, 0.5, seed=4, frame_px=64,
        camera_options={"fps": 4.0, "duration_s": 2.0}, script=fake(tmp_path, TRAFFIC_BLENDER),
        follow=follow, **extra,
    )  # fmt: skip


def test_a_camera_can_ride_on_a_car_and_the_car_is_hidden_from_itself(
    world: Path, tmp_path: Path
) -> None:
    traffic = {"cars_per_km": 40.0, "people_per_km": 30.0}
    meta = _ride(world, tmp_path, "car", {"kind": "car"}, traffic=traffic)
    assert meta["trajectory"] == "follow" and len(meta["frames"]) == 8
    assert meta["follow"]["kind"] == "car" and meta["follow"]["id"].startswith("car-")
    cam = json.loads((tmp_path / "car" / "scene" / "camera.json").read_text())
    assert cam["follow"] == {"agent": "traffic", "index": meta["follow"]["index"]}
    scene_dir = tmp_path / "car" / "scene"
    scene_json = json.loads((scene_dir / "scene.json").read_text())
    nodes = np.load(scene_dir / "ground_height.npy") if "terrain" in scene_json else None
    for f in cam["frames"]:  # 1.25 m above the highest ground within 1.5 m of the camera
        ground = (
            0.0
            if nodes is None
            else terrain_height(nodes, scene_json["terrain"]["cell_m"], f["x"], f["y"])
        )
        assert ground + 1.25 - 0.01 <= f["z"] <= ground + 1.25 + 1.5  # never under, slope-bounded
    agents_file = json.loads((tmp_path / "car" / "scene" / "agents.json").read_text())
    k = meta["follow"]["index"]  # the camera is just ahead of its car, on its track
    first = cam["frames"][0]
    assert (
        abs(first["x"] - agents_file["x"][k][0]) < 1.0
        and abs(first["y"] - agents_file["y"][k][0]) < 1.0
    )
    walk = _ride(world, tmp_path, "walk", {"kind": "person"}, traffic=traffic)
    assert walk["follow"]["id"].startswith("person-")
    cam2 = json.loads((tmp_path / "walk" / "scene" / "camera.json").read_text())
    assert cam2["follow"]["agent"] == "people" and cam2["frames"][0]["off_nadir_deg"] == 90.0
    again = _ride(world, tmp_path, "car2", {"kind": "car"}, traffic=traffic)
    assert again["scene_sha256"] == meta["scene_sha256"] and again["follow"] == meta["follow"]


def test_riding_needs_something_to_ride_on(world: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs traffic"):
        _ride(world, tmp_path, "a", {"kind": "car"})
    with pytest.raises(ValueError, match="no car called"):
        _ride(
            world, tmp_path, "b", {"kind": "car", "id": "car-9999"}, traffic={"cars_per_km": 40.0}
        )
    with pytest.raises(ValueError, match="nothing to follow"):
        _ride(world, tmp_path, "c", {"kind": "person"}, traffic={"cars_per_km": 40.0})
