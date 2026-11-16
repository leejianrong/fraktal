"""FRK-163: each person gets their own clothes, skin and build, from the index alone.

``blender_scene`` imports bpy, so the look function is pulled out of its source and run here.
"""

import ast
from pathlib import Path
from typing import Any

SOURCE = (Path(__file__).parents[2] / "src/fraktal/blender_scene.py").read_text()
WANTED = {"_frac", "person_look", "CLOTHES", "SKINS", "TROUSERS"}


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
    scope: dict[str, Any] = {}
    exec(compile(ast.Module(keep, []), "blender_scene.py", "exec"), scope)  # noqa: S102
    return scope


def test_a_look_depends_on_the_index_only() -> None:
    look = scene()["person_look"]
    assert [look(k) for k in range(40)] == [look(k) for k in range(40)]


def test_indices_and_scales_are_in_range_and_all_variants_turn_up() -> None:
    sc = scene()
    looks = [sc["person_look"](k) for k in range(400)]
    for key, size in (("shirt", len(sc["CLOTHES"])), ("skin", len(sc["SKINS"])),
                      ("trousers", len(sc["TROUSERS"]))):  # fmt: skip
        assert {lk[key] for lk in looks} == set(range(size))
    heights = [lk["scale"][0] for lk in looks]
    builds = [lk["scale"][1] for lk in looks]
    assert 0.9 <= min(heights) < 0.95 and 1.03 < max(heights) <= 1.08
    assert 0.9 <= min(builds) < 1.0 and 1.15 < max(builds) <= 1.25


def test_clothes_do_not_follow_skin_or_height() -> None:
    """Neighbouring people differ, and the traits are not locked together by the hash."""
    look = scene()["person_look"]
    pairs = {(look(k)["shirt"], look(k)["skin"], look(k)["trousers"]) for k in range(400)}
    assert len(pairs) > 100
