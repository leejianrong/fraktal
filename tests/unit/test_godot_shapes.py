"""The Godot viewer draws cars, people and drones from godot/shapes/agent_shapes.json, which is
exported from the part tables in blender_scene.py. This keeps the two from drifting apart."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]


def _exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_shapes", ROOT / "scripts/godot/export_shapes.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_shapes_are_what_the_blender_script_defines() -> None:
    committed = json.loads((ROOT / "godot/shapes/agent_shapes.json").read_text())
    assert committed == json.loads(json.dumps(_exporter().shapes())), (
        "run scripts/godot/export_shapes.py and commit godot/shapes/agent_shapes.json"
    )


def test_every_part_has_a_role_the_viewer_can_colour() -> None:
    agents = (ROOT / "godot/scripts/agents.gd").read_text()
    shapes = json.loads((ROOT / "godot/shapes/agent_shapes.json").read_text())
    assert set(shapes) == {"car", "person", "drone"}
    for kind, parts in shapes.items():
        assert parts, kind
        for part in parts:
            assert f'"{part["role"]}"' in agents, (kind, part["role"])
            assert len(part["box"]) == 5, part
