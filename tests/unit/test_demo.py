import json
import re
from pathlib import Path
from typing import Any

import pytest

from fraktal import biomes, demo
from fraktal.procedural import generate_world

CENTER = (32.5, 15.6)


def _data(html: Path) -> dict[str, Any]:
    found = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html.read_text(), re.S
    )
    assert found is not None
    data: dict[str, Any] = json.loads(found.group(1))
    return data


def test_the_demo_is_one_offline_page_with_the_tile_and_its_agents(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("dense_urban", {"size_m": 3000.0}))
    info = demo.build(world, tmp_path / "out", CENTER, size_m=400.0, duration_s=6.0, drones=2)
    html = tmp_path / "out" / "index.html"
    assert info["cars"] > 0 and info["drones"] == 2 and info["roads"] > 0
    text = html.read_text()
    assert "__DATA__" not in text and "src=" not in text and "http" not in text.split("<script")[0]
    d = _data(html)
    assert (
        d["frames"] == 60 and len(d["cars"]["x"]) == info["cars"] and len(d["cars"]["x"][0]) == 60
    )
    assert len(d["drones"]["z"]) == 2 and d["size"] == pytest.approx(400.0, abs=1.0)
    assert all(0 <= v <= 400.5 for row in d["drones"]["x"] for v in row)
    sx, sy, _ = d["start"]
    assert 0 <= sx <= 400.5 and 0 <= sy <= 400.5


def test_a_tile_without_roads_is_refused(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("dense_urban", {"size_m": 3000.0}))
    with pytest.raises(ValueError):
        demo.build(world, tmp_path / "out", (CENTER[0] + 0.5, CENTER[1] + 0.5), size_m=300.0)
