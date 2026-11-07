"""The Godot viewer loads an exported tile and a scripted drive is reproducible (FRK-91).

Needs a Godot 4 binary: set FRAKTAL_GODOT, or have `godot` on PATH, or unpack one under
~/.local/opt/godot-*/. Skipped otherwise. Runs headless, so no display or GPU is needed.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fraktal import biomes
from fraktal.cli import main
from fraktal.procedural import generate_world

pytestmark = pytest.mark.integration

PROJECT = Path(__file__).resolve().parents[2] / "godot"


def _godot() -> str | None:
    found = os.environ.get("FRAKTAL_GODOT") or shutil.which("godot")
    if found:
        return found
    local = sorted((Path.home() / ".local/opt").glob("godot-*/Godot_v*linux.x86_64"))
    return str(local[-1]) if local else None


def _run(godot: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [godot, "--headless", "--path", str(PROJECT), *args],
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip


def test_the_viewer_loads_a_tile_and_a_scripted_drive_is_reproducible(tmp_path: Path) -> None:
    godot = _godot()
    if godot is None:
        pytest.skip("no Godot 4 binary (set FRAKTAL_GODOT)")
    world = tmp_path / "w"
    generate_world(world, (104.0, 1.5), 3, biomes.resolve("dense_urban"))
    scene = tmp_path / "scene"
    assert main(["scene", str(world), "--out", str(scene), "--size-m", "400"]) == 0
    _run(godot, "--import")  # builds the class cache a fresh checkout does not have
    runs = [_run(godot, "--", "--scene", str(scene), "--selftest") for _ in range(2)]
    for r in runs:
        assert r.returncode == 0, r.stderr
        assert "SCRIPT ERROR" not in r.stdout + r.stderr, r.stdout + r.stderr
        assert "RAMMED true" in r.stdout
    states = [[x for x in r.stdout.splitlines() if x.startswith("STATE")] for r in runs]
    assert states[0] and states[0] == states[1]
