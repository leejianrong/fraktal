"""FRK-156: a crown gets enough leaf cards to cover its surface once, within a cap.

``blender_scene`` imports bpy, so the budget function is pulled out of its source and run here.
"""

import ast
import math
from pathlib import Path
from typing import Any

import pytest

SOURCE = (Path(__file__).parents[2] / "src/fraktal/blender_scene.py").read_text()
WANTED = {"card_budget", "CARD_SIZE", "CARD_COVER", "CARD_MAX"}


def scene() -> dict[str, Any]:
    keep: list[ast.stmt] = []
    for node in ast.parse(SOURCE).body:
        if isinstance(node, ast.FunctionDef):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if names & WANTED:
            keep.append(node)
    scope: dict[str, Any] = {"math": math}
    exec(compile(ast.Module(keep, []), "blender_scene.py", "exec"), scope)  # noqa: S102
    return scope


def test_a_crown_is_covered_about_once() -> None:
    sc = scene()
    n, size = sc["card_budget"](3.0, 2.0)
    area = 4 * math.pi * 3.0 * 2.0
    assert n * (2 * size) ** 2 / area == pytest.approx(1.0, rel=0.1)


def test_big_crowns_hit_the_cap_and_small_ones_keep_a_floor() -> None:
    sc = scene()
    assert sc["card_budget"](14.0, 12.0)[0] == sc["CARD_MAX"]
    assert sc["card_budget"](0.4, 0.3)[0] == 14


def test_card_size_stays_in_range() -> None:
    sc = scene()
    for rx in (0.5, 3.0, 8.0):
        size = sc["card_budget"](rx, rx)[1]
        assert 0.6 <= size <= 1.3
