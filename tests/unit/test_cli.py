import argparse
import json
from pathlib import Path

import pytest

from fraktal import __version__, cli
from fraktal.cli import ExitCode, main

STAGES = ["version"]  # a stand-in: every real command is implemented


def test_exit_codes_are_a_stable_contract() -> None:
    assert [
        int(c) for c in (ExitCode.OK, ExitCode.USAGE, ExitCode.NOT_IMPLEMENTED, ExitCode.FAILURE)
    ] == [
        0,
        2,
        3,
        4,
    ]


def test_version_json_envelope(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version", "--json"]) == ExitCode.OK
    envelope = json.loads(capsys.readouterr().out)
    assert envelope == {"ok": True, "data": {"version": __version__}, "error": None}


def test_version_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == ExitCode.OK
    assert capsys.readouterr().out.strip() == f"fraktal {__version__}"


@pytest.mark.parametrize("stage", STAGES)
def test_unimplemented_stage_exits_3_with_json_error(
    stage: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(cli.COMMANDS, stage, ("stub", cli._not_implemented(stage)))
    assert main([stage, "--json"]) == ExitCode.NOT_IMPLEMENTED
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is False
    assert envelope["data"] is None
    assert envelope["error"]["kind"] == "not_implemented"
    assert stage in envelope["error"]["message"]


def test_human_errors_go_to_stderr(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(cli.COMMANDS, "version", ("stub", cli._not_implemented("version")))
    assert main(["version"]) == ExitCode.NOT_IMPLEMENTED
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not implemented" in captured.err


def test_usage_errors_exit_2() -> None:
    with pytest.raises(SystemExit) as info:
        main(["no-such-command"])
    assert info.value.code == ExitCode.USAGE
    with pytest.raises(SystemExit) as info:
        main([])
    assert info.value.code == ExitCode.USAGE


def test_unexpected_exception_exits_4(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_args: argparse.Namespace) -> cli.Result:
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(cli.COMMANDS, "version", ("Print the version", boom))
    assert main(["version", "--json"]) == ExitCode.FAILURE
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["error"]["kind"] == "unexpected"
    assert "disk on fire" in envelope["error"]["message"]


def test_scene_exports_a_tile_without_blender(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fraktal import biomes
    from fraktal.procedural import generate_world

    world = tmp_path / "w"
    generate_world(world, (104.0, 1.5), 3, biomes.resolve("dense_urban"))
    out = tmp_path / "s"
    assert main(["scene", str(world), "--out", str(out), "--size-m", "200", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["ok"] and info["data"]["buildings"] > 0
    assert (out / "start.json").exists()
    assert (out / "scene.json").exists() and (out / "ground.npy").exists()


def test_scene_can_write_the_tracks_the_viewer_replays(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from fraktal import biomes
    from fraktal.procedural import generate_world

    world = tmp_path / "w"
    generate_world(world, (104.0, 1.5), 3, biomes.resolve("dense_urban"))
    base = ["scene", str(world), "--out", "", "--size-m", "300", "--json", "--duration-s", "20"]
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        argv = [*base]
        argv[argv.index("--out") + 1] = str(out)
        assert main([*argv, "--agents", "--drones", "2", "--seed", "4"]) == 0
        outs.append(out)
    info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert info["data"]["agents"] == ["agents.json", "drones.json", "people.json"]
    cars = json.loads((outs[0] / "agents.json").read_text())
    assert cars["kind"] == "car" and len(cars["t"]) == 80  # 20 s at 0.25 s steps
    assert all(len(row) == len(cars["t"]) for row in (cars["x"] + cars["y"] + cars["heading"]))
    assert json.loads((outs[0] / "drones.json").read_text())["kind"] == "drone"
    assert len(json.loads((outs[0] / "drones.json").read_text())["ids"]) == 2
    for name in (
        "agents.json",
        "people.json",
        "drones.json",
    ):  # the same seed replays the same loop
        assert (outs[0] / name).read_text() == (outs[1] / name).read_text()
    plain = tmp_path / "plain"
    assert main(["scene", str(world), "--out", str(plain), "--size-m", "300", "--json"]) == 0
    assert not (plain / "agents.json").exists()  # off unless asked for
