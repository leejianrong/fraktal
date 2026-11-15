"""FRK-163: moving cars and people are rounded-off boxes, built by ``_box_vertices``.

``blender_scene`` imports bpy, so the geometry code is pulled out of its source and run here.
"""

import ast
from pathlib import Path
from typing import Any

import numpy as np

SOURCE = (Path(__file__).parents[2] / "src/fraktal/blender_scene.py").read_text()
WANTED = {"CORNERS", "CAR_PARTS", "CAR_BOXES", "PERSON_PARTS", "PERSON_BOXES", "_box_vertices"}


def _scene() -> dict[str, Any]:
    keep = []
    for node in ast.parse(SOURCE).body:
        names = (
            {t.id for t in node.targets if isinstance(t, ast.Name)}
            if isinstance(node, ast.Assign)
            else {node.name}
            if isinstance(node, ast.FunctionDef)
            else set()
        )
        if names & WANTED:
            keep.append(node)
    scope: dict[str, Any] = {"np": np}
    exec(compile(ast.Module(keep, []), "blender_scene.py", "exec"), scope)  # noqa: S102
    return scope


def _one(boxes: Any, heading: float = 0.0) -> Any:
    fn = _scene()["_box_vertices"]
    return fn(np.array([10.0]), np.array([5.0]), np.array([heading]), boxes)[0]


def test_a_plain_box_is_unchanged_by_the_new_options() -> None:
    v = _one(((1.0, 2.0, 0.5, 0.0, 1.0),))
    assert v.shape == (8, 3)
    assert np.allclose(v[:4, 2], 0.0) and np.allclose(v[4:, 2], 1.0)
    assert np.allclose(v[:4, :2], v[4:, :2])  # a prism: the top sits over the bottom
    assert np.allclose(v[:, 0].min(), 10.0 + 1.0 - 2.0) and np.allclose(v[:, 0].max(), 13.0)


def test_a_taper_narrows_the_top_and_a_front_height_slopes_it() -> None:
    v = _one(((0.0, 2.0, 1.0, 0.0, 1.0, {"top_l": 1.0, "top_w": 0.5, "z1_front": 0.6}),))
    assert np.ptp(v[:4, 0]) == 4.0 and np.ptp(v[4:, 0]) == 2.0  # length 4 below, 2 above
    assert np.ptp(v[:4, 1]) == 2.0 and np.ptp(v[4:, 1]) == 1.0
    front = v[4:][v[4:, 0] > 10.0]
    back = v[4:][v[4:, 0] < 10.0]
    assert np.allclose(front[:, 2], 0.6) and np.allclose(back[:, 2], 1.0)


def test_side_moves_a_box_to_the_left_of_the_heading() -> None:
    for heading in (0.0, np.pi / 2, 1.0):
        v = _one(((0.0, 1.0, 1.0, 0.0, 1.0, {"side": 2.0}),), heading)
        left = np.array([-np.sin(heading), np.cos(heading)])
        assert np.allclose(v[:, :2].mean(axis=0) - [10.0, 5.0], 2.0 * left)


def test_cars_and_people_use_the_new_shapes_and_every_part_has_a_known_role() -> None:
    scope = _scene()
    car, person = scope["CAR_PARTS"], scope["PERSON_PARTS"]
    assert len(scope["CAR_BOXES"]) == len(car) and len(scope["PERSON_BOXES"]) == len(person)
    assert {r for r, _ in car} == {"body", "glass", "tyre"}
    assert {r for r, _ in person} == {"trousers", "shirt", "skin"}
    wheels = [b for r, b in car if r == "tyre" and len(b) > 5 and "side" in b[5]]
    assert len(wheels) == 4  # four wheels, not one block across each axle
    assert sum(1 for _, b in car if len(b) > 5 and "z1_front" in b[5]) >= 2  # a sloping bonnet
    legs = [b for r, b in person if r == "trousers"]
    assert len(legs) == 2 and legs[0][5]["side"] == -legs[1][5]["side"]


def test_a_car_stays_within_a_normal_footprint() -> None:
    scope = _scene()
    v = scope["_box_vertices"](
        np.array([0.0]), np.array([0.0]), np.array([0.0]), scope["CAR_BOXES"]
    )[0]
    assert 4.0 < np.ptp(v[:, 0]) < 5.0 and 1.6 < np.ptp(v[:, 1]) < 2.1
    assert 1.3 < v[:, 2].max() < 1.7 and v[:, 2].min() == 0.0
