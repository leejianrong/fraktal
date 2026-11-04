"""Needs a Python with bpy: set FRAKTAL_BPY_PYTHON to run it (CI does not)."""

import json
import os
from pathlib import Path

import pytest
import rasterio

from fraktal import biomes
from fraktal.cli import main
from fraktal.procedural import generate_world

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.environ.get("FRAKTAL_BPY_PYTHON"), reason="needs a Python with bpy")
def test_a_tiny_world_renders_in_blender(tmp_path: Path) -> None:
    world = tmp_path / "world"
    generate_world(world, (104.0, 1.5), 3, biomes.resolve("dense_urban"))
    out = tmp_path / "real"
    code = main(
        ["render3d", str(world), "--out", str(out), "--center", "104", "1.5", "--size-m", "120",
         "--gsd-m", "0.5", "--samples", "4", "--timeout-s", "600"]
    )  # fmt: skip
    assert code == 0
    meta = json.loads((out / "render.json").read_text())
    assert meta["px"] == 240 and meta["bpy"] and meta["buildings"] > 0
    with rasterio.open(out / "tile.tif") as tif:
        assert tif.read().std() > 5  # not a black or blank frame
