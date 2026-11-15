"""FRK-178: cars from a palette-texture OBJ take one flat colour per face from the palette image.

``blender_scene`` imports bpy, so the parser is pulled out of its source and run here.
"""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]
SOURCE = (ROOT / "src/fraktal/blender_scene.py").read_text()
WANTED = {
    "palette_colour", "parse_obj_car", "model_variants", "CAR_MESHES", "KENNEY_MESHES",
    "MODEL_CYCLE",
}  # fmt: skip


def scene() -> dict[str, Any]:
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


# A 2 x 2 palette, row 0 at the bottom: red, green / blue, white (display values).
PALETTE = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]], dtype="float64")

# Two triangles of a slab 2 long (Z), 1 wide, 1 tall: one face on the red cell, one on the blue.
OBJ = """\
v -0.5 0 -1
v 0.5 0 -1
v 0.5 1 -1
v -0.5 0 1
v 0.5 0 1
v 0.5 1 1
vt 0.1 0.1
vt 0.2 0.2
vt 0.1 0.2
vt 0.1 0.8
vt 0.2 0.9
vt 0.1 0.9
usemtl colormap
f 1/1 2/2 3/3
f 4/4 5/5 6/6
"""


def test_each_face_takes_the_palette_colour_at_its_uvs() -> None:
    car = scene()["parse_obj_car"](OBJ, "", PALETTE, 1.0)
    assert car["poly_mat"] == ["pal-ff0000", "pal-0000ff"]  # red at the bottom, blue above


def test_row_zero_is_the_bottom_of_the_image() -> None:
    fn = scene()["palette_colour"]
    assert fn(PALETTE, np.array([0.1, 0.1])) == (255, 0, 0)  # low v: the bottom row
    assert fn(PALETTE, np.array([0.9, 0.9])) == (255, 255, 255)
    assert fn(PALETTE, np.array([1.0, 1.0])) == (255, 255, 255)  # the edge stays inside


def test_the_face_colours_are_listed_as_display_rgb() -> None:
    car = scene()["parse_obj_car"](OBJ, "", PALETTE, 1.0)
    assert car["colours"]["pal-ff0000"] == (1.0, 0.0, 0.0)
    assert set(car["poly_mat"]) <= set(car["colours"])


def test_the_car_is_put_in_the_car_frame_wheels_on_the_ground() -> None:
    car = scene()["parse_obj_car"](OBJ, "", PALETTE, 1.0)
    v = car["verts"]
    assert v[:, 2].min() == 0.0  # on the ground
    assert np.isclose(v[:, 0].min(), -v[:, 0].max())  # centred, length along x
    assert np.isclose(v[:, 0].max() - v[:, 0].min(), 2.0)


def test_forward_can_be_set_by_hand_and_flips_the_car() -> None:
    parse = scene()["parse_obj_car"]
    a, b = parse(OBJ, "", PALETTE, 1.0)["verts"], parse(OBJ, "", PALETTE, -1.0)["verts"]
    assert np.allclose(a[:, 0], -b[:, 0])


def test_a_palette_car_without_uvs_is_refused() -> None:
    with pytest.raises(ValueError, match="UV"):
        scene()["parse_obj_car"]("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", "", PALETTE, 1.0)


def test_the_existing_cars_parse_as_before() -> None:
    mesh = ROOT / "assets/meshes/quaternius-cars"
    car = scene()["parse_obj_car"]((mesh / "Taxi.obj").read_text(), (mesh / "Taxi.mtl").read_text())
    assert "Headlights" in car["colours"] and len(car["polys"]) > 1000
    assert not any(m.startswith("pal-") for m in car["poly_mat"])


def test_kenney_vehicles_are_a_quarter_of_the_traffic_and_cars_alone_cycle_by_index() -> None:
    sc = scene()
    models = len(sc["CAR_MESHES"]) + len(sc["KENNEY_MESHES"])
    v = sc["model_variants"](1200, models)
    assert set(v.tolist()) == set(range(models))
    assert (v >= len(sc["CAR_MESHES"])).mean() == pytest.approx(0.25, abs=0.01)
    assert sc["model_variants"](12, 6).tolist() == [0, 1, 2, 3, 4, 5] * 2
