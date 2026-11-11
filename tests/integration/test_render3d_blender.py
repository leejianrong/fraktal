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


@pytest.mark.skipif(not os.environ.get("FRAKTAL_BPY_PYTHON"), reason="needs a Python with bpy")
def test_nadir_building_labels_match_the_footprints(tmp_path: Path) -> None:
    import numpy as np
    from affine import Affine
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    from fraktal import labels3d

    world = tmp_path / "world"
    generate_world(world, (104.0, 1.5), 3, biomes.resolve("dense_urban"))
    out = tmp_path / "real"
    args = ["render3d", str(world), "--out", str(out), "--center", "104", "1.5", "--size-m", "120",
            "--gsd-m", "0.5", "--samples", "4", "--timeout-s", "600"]  # fmt: skip
    assert main(args) == 0
    scene = json.loads((out / "scene" / "scene.json").read_text())
    px, gsd = scene["px"], scene["gsd_m"]
    footprints = rasterize(
        [(Polygon(b["ring"]), 1) for b in scene["buildings"]],
        (px, px),
        transform=Affine(gsd, 0, 0, 0, -gsd, px * gsd),
        dtype="uint8",
    ).astype(bool)
    with rasterio.open(out / "labels.png") as src:
        semantic, instance = labels3d.decode(src.read()[:3])
    mask = semantic == labels3d.CLASSES["building"]
    assert (mask & footprints).sum() / (mask | footprints).sum() > 0.9
    assert set(np.unique(semantic)) <= set(labels3d.CLASSES.values())
    assert 0 < len(np.unique(instance[mask])) <= len(scene["buildings"])


@pytest.mark.skipif(not os.environ.get("FRAKTAL_BPY_PYTHON"), reason="needs a Python with bpy")
def test_a_tile_with_no_buildings_renders_on_sloping_ground(tmp_path: Path) -> None:
    from fraktal import render3d_batch

    world = tmp_path / "estate"
    generate_world(world, (104.0, 1.5), 3, biomes.resolve("oil_palm_plantation"))
    specs = render3d_batch.plan(world, 12, 5, 100.0, 1.0, min_buildings=0)
    out = tmp_path / "batch"
    record = render3d_batch.run(world, out, specs, samples=4, timeout_s=600.0)
    assert record["counts"]["failed"] == 0, record["tiles"]
