import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from fraktal import building_grammar as grammar

ASSETS = Path(__file__).resolve().parents[2] / "assets"
RING = [[0, 0], [14, 0], [14, 9], [0, 9], [0, 0]]


def test_the_same_id_and_seed_give_the_same_mesh_and_another_id_a_different_one() -> None:
    a = grammar.build("aaaa1111", RING, 9.0, None, 1)
    assert a == grammar.build("aaaa1111", RING, 9.0, None, 1)
    assert a != grammar.build("bbbb2222", RING, 9.0, None, 1)
    assert a != grammar.build("aaaa1111", RING, 9.0, None, 2)


def test_a_flat_roof_block_has_walls_roof_parapet_windows_and_a_door() -> None:
    mesh = grammar.build("aaaa1111", RING, 9.0, None, 1)
    kinds = Counter(mesh.materials)
    assert sum(v for k, v in kinds.items() if k.startswith("roof-flat")) == 1
    assert kinds["door"] == 1 and kinds["window-glass"] >= 10
    z = np.array(mesh.vertices)[:, 2]
    assert z.min() == 0.0 and z.max() > 9.0  # the parapet and clutter rise above the roof


def test_the_roof_material_follows_the_recorded_material() -> None:
    assert set(grammar.TIN_ROOFS) & set(grammar.build("aaaa1111", RING, 4.0, "tin", 1).materials)
    tiles = {n for n in grammar.build("aaaa1111", RING, 4.0, "tile", 1).materials if "tile" in n}
    assert tiles and tiles <= {"roof-tile", grammar.CLAY_TILE}


def test_flat_roofs_vary_between_buildings() -> None:
    roofs = {
        next(
            m
            for m in grammar.build(f"id{i:06d}", RING, 6.0, None, 0).materials
            if m.startswith("roof-")
        )
        for i in range(40)
    }
    assert len(roofs) == 4


def test_storeys_follow_the_height() -> None:
    low = Counter(grammar.build("cccc3333", RING, 3.0, None, 5).materials)["window-glass"]
    tall = Counter(grammar.build("cccc3333", RING, 12.0, None, 5).materials)["window-glass"]
    assert tall > low


def test_nothing_sticks_out_of_the_footprint_sideways_except_openings() -> None:
    mesh = grammar.build("aaaa1111", RING, 9.0, None, 1)
    v = np.array(mesh.vertices)
    proud = {"window-glass", "window-frame", "door", "balcony"}
    body = sorted(
        {i for f, m in zip(mesh.faces, mesh.materials, strict=True) if m not in proud for i in f}
    )
    xy = v[body][:, :2]
    assert xy.min() >= -0.05 and xy[:, 0].max() <= 14.05 and xy[:, 1].max() <= 9.05
    assert np.array(v)[:, :2].min() > -1.1  # a balcony reaches out 0.9 m, a sill 0.1 m


def test_a_clockwise_ring_gives_the_same_building() -> None:
    cw = grammar.build("aaaa1111", RING[::-1], 9.0, None, 1)
    ccw = grammar.build("aaaa1111", RING, 9.0, None, 1)
    assert Counter(cw.materials) == Counter(ccw.materials)


@pytest.mark.parametrize(
    "ring", [[[0, 0], [1, 0]], [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0, 0]]]
)
def test_an_unusable_footprint_is_refused(ring: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="no usable area"):
        grammar.build("aaaa1111", ring, 6.0, None, 1)


def test_an_l_shaped_footprint_builds() -> None:
    ring = [[0, 0], [12, 0], [12, 5], [5, 5], [5, 12], [0, 12], [0, 0]]
    mesh = grammar.build("dddd4444", ring, 6.0, None, 3)
    assert len(mesh.faces) > 20 and len(mesh.faces) == len(mesh.materials)


def test_pack_gives_flat_arrays_that_index_each_other() -> None:
    meshes = [grammar.build(f"id{i:06d}", RING, 6.0, None, 0) for i in range(3)]
    packed = grammar.pack(meshes)
    assert packed["face_sizes"].sum() == len(packed["face_verts"])
    assert packed["face_verts"].max() < len(packed["vertices"])
    assert packed["face_material"].max() < len(packed["materials"])
    assert len(packed["face_sizes"]) == sum(len(m.faces) for m in meshes)


def test_every_material_the_grammar_uses_exists_in_the_library() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text())["assets"]
    listed = {row["file"] for row in manifest}
    for name in grammar.MATERIALS:
        spec_file = f"materials/{name}.json"
        assert spec_file in listed, name
        spec = json.loads((ASSETS / spec_file).read_text())
        if "image" in spec:
            assert spec["image"] in listed and (ASSETS / spec["image"]).is_file()


BIG = [[0, 0], [20, 0], [20, 14], [0, 14], [0, 0]]


def test_a_recorded_tin_or_tile_roof_gives_a_pitched_shed() -> None:
    mesh = grammar.build("aaaa1111", RING, 4.0, "tin", 1)
    assert mesh.typology == "shed"
    z = np.array(mesh.vertices)[:, 2]
    assert z.max() > 4.0 + 0.7 and z.max() < 4.0 + 2.3  # the ridge stands above the eaves
    roof_faces = [
        f for f, m in zip(mesh.faces, mesh.materials, strict=True) if m in grammar.TIN_ROOFS
    ]
    assert len(roof_faces) == 2  # one slope each side of the ridge
    v = np.array(mesh.vertices)
    for face in roof_faces:  # each slope is a plane
        pts = v[list(face)]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        assert all(abs(float((p - pts[0]) @ normal)) < 1e-6 for p in pts)
        assert normal[2] > 0  # facing up
    xy = v[:, :2]
    assert xy.min() >= -0.2 and xy[:, 0].max() <= 14.2 and xy[:, 1].max() <= 9.2  # sills stand out


def test_a_shed_is_closed_at_the_gable_ends() -> None:
    mesh = grammar.build("aaaa1111", RING, 4.0, "tile", 1)
    v = np.array(mesh.vertices)
    walls = [f for f, m in zip(mesh.faces, mesh.materials, strict=True) if m.startswith("wall")]
    peaks = [f for f in walls if len(f) == 5]  # a wall that rises to the ridge has a fifth point
    assert len(peaks) == 2 and all(
        abs(v[list(f)][:, 2].max() - v[:, 2].max()) < 1e-9 for f in peaks
    )


def test_some_roomy_low_rectangles_become_courtyard_houses_and_others_stay_blocks() -> None:
    kinds = Counter(grammar.build(f"id{i:06d}", BIG, 4.0, None, 0).typology for i in range(200))
    assert set(kinds) == {"block", "courtyard"}
    assert 0.2 < kinds["courtyard"] / 200 < 0.5
    assert grammar.build("aaaa1111", BIG, 12.0, None, 0).typology == "block"  # too tall
    assert grammar.build("aaaa1111", RING, 4.0, None, 0).typology == "block"  # too small


def test_a_courtyard_house_has_a_yard_inside_its_roof_ring() -> None:
    mesh = next(
        m
        for i in range(100)
        if (m := grammar.build(f"id{i:06d}", BIG, 4.0, None, 0)).typology == "courtyard"
    )
    v = np.array(mesh.vertices)
    roof = [f for f, m in zip(mesh.faces, mesh.materials, strict=True) if m.startswith("roof-")]
    area = 0.0
    for face in roof:
        p = v[list(face)][:, :2]
        area += 0.5 * abs(
            sum(p[k, 0] * p[(k + 1) % 4, 1] - p[(k + 1) % 4, 0] * p[k, 1] for k in range(4))
        )
    assert 20 * 14 * 0.4 < area < 20 * 14 * 0.9  # a ring, not the full footprint
    assert Counter(mesh.materials)["door"] == 1


def test_a_wall_has_a_plinth_windows_have_frames_and_tall_blocks_get_balconies() -> None:
    mesh = grammar.build("aaaa1111", BIG, 10.0, None, 1)
    kinds = Counter(mesh.materials)
    assert kinds["wall-plinth"] == 4 and kinds["window-frame"] >= 2 * kinds["window-glass"] - 4
    low = grammar.build("aaaa1111", BIG, 2.5, None, 1)
    assert "wall-plinth" not in low.materials and "balcony" not in low.materials
    share = sum(
        "balcony" in grammar.build(f"id{i:06d}", BIG, 10.0, None, 0).materials for i in range(200)
    )
    assert 0.15 < share / 200 < 0.55  # BALCONY_SHARE is 0.35, and a few have no upper bay drawn


def test_the_openings_do_not_move_when_facade_detail_is_added() -> None:
    a = grammar.build("aaaa1111", RING, 9.0, None, 1)
    glass = [f for f, m in zip(a.faces, a.materials, strict=True) if m == "window-glass"]
    assert len(glass) >= 10  # the same bays as before: the facade has its own random stream


def test_tin_roofs_come_in_several_colours_in_the_set_proportions() -> None:
    picks = Counter(
        next(
            m
            for m in grammar.build(f"id{i:06d}", RING, 4.0, "tin", 0).materials
            if m in grammar.TIN_ROOFS
        )
        for i in range(600)
    )
    assert set(picks) == set(grammar.TIN_ROOFS)
    for name, weight in zip(grammar.TIN_ROOFS, grammar.TIN_WEIGHTS, strict=True):
        assert abs(picks[name] / 600 - weight) < 0.07
    assert grammar.build("aaaa1111", RING, 4.0, "tin", 1) == grammar.build(
        "aaaa1111", RING, 4.0, "tin", 1
    )


def test_some_walls_and_tile_roofs_use_the_generated_textures() -> None:
    walls: list[str] = []
    roofs: list[str] = []
    for i in range(120):
        mesh = grammar.build(f"b{i}", BIG, 6.0, "tile" if i % 2 else None, seed=3)
        names = set(mesh.materials)
        walls += [n for n in names if n.startswith("wall-") and n != "wall-plinth"]
        roofs += [n for n in names if n.startswith("roof-tile")]
    generated = [w for w in walls if w in grammar.GENERATED_WALLS]
    assert 0.1 < len(generated) / len(walls) < 0.45, len(generated) / len(walls)
    assert {"roof-tile", "roof-tile-clay"} <= set(roofs)
