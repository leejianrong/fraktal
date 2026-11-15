from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from fraktal import biomes, camera, markings
from fraktal.blender_export import export_scene
from fraktal.procedural import generate_world

NONE = camera.obstacles([], [])


def _road(x0: float, y0: float, x1: float, y1: float, w: float, kind: str = "local") -> Any:
    return markings.RoadLine([(x0, y0), (x1, y1)], w, kind)


def _centre(quad: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(v[0] for v in quad) / 4, sum(v[1] for v in quad) / 4


def _count(parts: list[markings.Part], material: str) -> int:
    return sum(1 for p in parts if p[3] == material)


def test_a_biome_without_a_style_gets_no_marks() -> None:
    assert markings.build([_road(0, 0, 100, 0, 10)], NONE, "khartoum") == []
    assert markings.build([_road(0, 0, 100, 0, 10)], NONE, "dense_urban") == []


def test_a_local_street_gets_a_dashed_centre_and_kerbs_on_both_sides() -> None:
    parts = markings.build([_road(0, 0, 100, 0, 10)], NONE, "manila_quezon")
    assert _count(parts, "line-white") == int((100 - markings.DASH_M) // markings.PERIOD_M) + 1
    kerb_y = [_centre(q)[1] for q, _, _, m in parts if m == "kerb"]
    assert min(kerb_y) < -5 and max(kerb_y) > 5  # a kerb at plus and minus half the width
    assert all(top == markings.KERB_HEIGHT_M for _, _, top, m in parts if m == "kerb")
    assert all(top == markings.PAINT_M for _, _, top, m in parts if m == "line-white")
    assert _count(parts, "pavement") == 0  # Manila-Quezon: kerbs, no raised pavement


def test_a_wide_arterial_gets_lane_lines_and_a_solid_yellow_centre() -> None:
    parts = markings.build([_road(0, 0, 100, 0, 24, "arterial")], NONE, "manila_quezon")
    lane_y = {round(_centre(q)[1], 1) for q, _, _, m in parts if m == "line-white"}
    assert lane_y == {-6.0, 6.0}  # a quarter of the width either side of the centre
    assert _count(parts, "line-yellow") > 40


def test_nairobi_has_raised_pavements_and_narrow_roads_get_nothing() -> None:
    parts = markings.build([_road(0, 0, 60, 0, 12)], NONE, "nairobi_centre")
    assert _count(parts, "pavement") > 0 and _count(parts, "kerb") > 0
    path = markings.RoadLine([(0, 0), (60, 0)], 1.6, "footpath")
    assert markings.build([path], NONE, "nairobi_centre") == []
    assert markings.build([_road(0, 0, 60, 0, 4)], NONE, "nairobi_centre") == []


def test_marks_are_cut_at_a_junction_and_kerbs_stop_at_the_crossing_road() -> None:
    roads = [_road(0, 0, 100, 0, 10), _road(50, -50, 50, 50, 10)]
    parts = markings.build(roads, NONE, "manila_quezon")
    for quad, _, _, material in parts:
        cx, cy = _centre(quad)
        if material == "line-white":  # no dash in the junction box and its margin
            assert not (abs(cx - 50) < 5 and abs(cy) < 5)
        if material == "kerb":  # no kerb across the side street's carriageway
            assert not (abs(cx - 50) < 4.5 and abs(cy) < 6)


def test_a_building_keeps_the_raised_edges_out() -> None:
    house = Polygon([(40, 6), (60, 6), (60, 14), (40, 14)])
    hazards = camera.obstacles([{"ring": list(house.exterior.coords)}], [])
    road = [_road(0, 0, 100, 0, 10)]
    free = markings.build(road, NONE, "nairobi_centre")
    blocked = markings.build(road, hazards, "nairobi_centre")
    assert _count(blocked, "pavement") < _count(free, "pavement")
    raised = [q for q, _, _, m in blocked if m in ("kerb", "pavement")]
    assert not any(house.contains(Point(*_centre(q))) for q in raised)


def test_arterial_junctions_get_zebra_stripes_and_khartoum_gets_no_paint() -> None:
    roads = [_road(0, 0, 200, 0, 24, "arterial"), _road(100, -80, 100, 80, 10)]
    parts = markings.build(roads, NONE, "manila_quezon")
    near = [
        q for q, _, _, m in parts
        if m == "line-white" and 6 < abs(_centre(q)[0] - 100) < 14 and abs(_centre(q)[1]) < 12
    ]  # fmt: skip
    assert len(near) >= 20  # stripes across the arterial, either side of the side street
    khartoum = markings.build(roads, NONE, "khartoum_city")
    assert khartoum and not any(m.startswith("line") for _, _, _, m in khartoum)


def test_pack_gives_a_box_per_part_and_is_repeatable() -> None:
    parts = markings.build([_road(0, 0, 60, 0, 10)], NONE, "manila_quezon")
    a, b = markings.pack(parts), markings.pack(parts)
    assert a["vertices"].shape == (8 * len(parts), 3) and a["faces"].shape == (12 * len(parts), 3)
    assert a["faces"].max() == len(a["vertices"]) - 1
    assert a["face_material"].max() < len(a["materials"])
    assert all((a[k] == b[k]).all() for k in ("vertices", "faces", "face_material"))
    empty = markings.pack([])
    assert empty["faces"].shape == (0, 3) and len(empty["materials"]) == len(markings.MATERIALS)


@pytest.mark.slow
def test_a_city_export_writes_the_marks_inside_the_tile_and_other_biomes_none(
    tmp_path: Path,
) -> None:
    centre = (32.55, 15.64)
    manila = tmp_path / "m"
    generate_world(manila, centre, 3, biomes.resolve("manila_quezon", {"size_m": 1000.0}))
    scene = export_scene(manila, tmp_path / "a", centre, 300.0, 1.0)
    assert scene["markings"]["parts"] > 100 and len(scene["markings"]["materials"]) == len(
        markings.MATERIALS
    )
    verts = np.load(tmp_path / "a" / "markings_vertices.npy")
    assert len(verts) == 8 * scene["markings"]["parts"]
    assert (
        markings.KERB_HEIGHT_M <= verts[:, 2].max() <= markings.KERB_HEIGHT_M + 0.01
    )  # wear on top of it
    margin = 10.0 + 13.0  # the clip margin and half the widest road
    assert verts[:, :2].min() >= -margin and verts[:, :2].max() <= 300.0 + margin
    wires = scene["cables"][
        "wires"
    ]  # overhead cables ride along with the same city export (FRK-147)
    assert wires >= 5 and len(np.load(tmp_path / "a" / "cables_faces.npy")) == 2 * 10 * wires
    again = export_scene(manila, tmp_path / "b", centre, 300.0, 1.0)
    assert again["cables"] == scene["cables"]
    assert (verts == np.load(tmp_path / "b" / "markings_vertices.npy")).all()
    assert again["markings"] == scene["markings"]
    plain = tmp_path / "p"
    generate_world(plain, centre, 3, biomes.resolve("dense_urban", {"size_m": 1000.0}))
    other = export_scene(plain, tmp_path / "c", centre, 300.0, 1.0)
    assert other["markings"]["parts"] == 0 and other["cables"]["wires"] == 0


ROAD_WEAR = {"road-patch", "road-patch-pale", "road-stain", "road-polish", "road-crack", "manhole"}
EDGE_WEAR = {"kerb-grime", "kerb-scuff"}
PAVEMENT_WEAR = {"pavement-stain", "pavement-crack", "pavement-hole", "weed"}
WEAR = ROAD_WEAR | EDGE_WEAR | PAVEMENT_WEAR


def _worn(seed: int | None) -> list[markings.Part]:
    return markings.build([_road(0, 0, 300, 0, 10)], NONE, "manila_quezon", seed)


def test_wear_is_off_without_a_seed() -> None:
    assert not any(m in WEAR for _, _, _, m in _worn(None))


def test_a_seeded_road_gets_every_kind_of_wear_and_it_stays_under_the_paint() -> None:
    parts = _worn(7)
    kinds = {m for _, _, _, m in parts if m in WEAR}
    assert kinds == ROAD_WEAR | EDGE_WEAR  # Manila-Quezon has kerbs and no raised pavement
    assert all(top < markings.PAINT_M for _, _, top, m in parts if m in ROAD_WEAR)
    marks = [p for p in parts if p[3] not in WEAR]
    assert marks == _worn(None)  # the marks and kerbs are exactly what they were


def test_wear_is_deterministic_and_changes_with_the_seed() -> None:
    assert _worn(7) == _worn(7)
    assert _worn(7) != _worn(8)


def test_wear_stays_on_the_carriageway_and_out_of_buildings() -> None:
    ys = [y for q, _, _, m in _worn(7) if m in ROAD_WEAR for _, y in q]
    assert max(ys) <= 5.0 + 0.1 and min(ys) >= -5.0 - 0.1  # within half the road width
    ring = [(100.0, -20.0), (160.0, -20.0), (160.0, 20.0), (100.0, 20.0), (100.0, -20.0)]
    blocked = markings.build(
        [_road(0, 0, 300, 0, 10)], camera.obstacles([{"ring": ring}], []), "manila_quezon", 7
    )
    inside = [q for q, _, _, m in blocked if m in WEAR and 100 < _centre(q)[0] < 160]
    assert inside == []


def _nairobi(seed: int | None) -> list[markings.Part]:
    return markings.build([_road(0, 0, 300, 0, 10)], NONE, "nairobi_centre", seed)


def test_a_raised_pavement_gets_stains_cracks_missing_pavers_and_weeds() -> None:
    parts = _nairobi(7)
    kinds = {m for _, _, _, m in parts if m in WEAR}
    assert kinds == WEAR  # every kind, on a biome with kerbs and a raised pavement
    for quad, base, top, m in parts:
        if m in PAVEMENT_WEAR:
            assert base == markings.KERB_HEIGHT_M and top > base  # lying on top of the pavement
            assert top - base < 0.07  # a weed is the tallest, at 6 cm
            ys = [abs(y) for _, y in quad]  # inside the pavement band, beyond the kerb
            assert min(ys) >= 5.0 + markings.KERB_M - 0.01 and max(ys) <= 5.0 + 0.2 + 2.5 + 0.05
    marks = [p for p in parts if p[3] not in WEAR]
    assert marks == _nairobi(None)  # kerbs, pavements and paint are what they were


def test_kerb_wear_sits_on_the_kerb_and_is_seeded() -> None:
    parts = _worn(7)
    for quad, base, top, m in parts:
        if m == "kerb-grime":
            assert base == 0.0 and top >= markings.KERB_HEIGHT_M  # up the face
            assert all(4.9 <= abs(y) <= 5.01 for _, y in quad)  # the road side of the kerb
        if m == "kerb-scuff":
            assert base == markings.KERB_HEIGHT_M  # on the kerb's top
            assert all(5.0 < abs(y) < 5.0 + markings.KERB_M for _, y in quad)
    assert _nairobi(7) == _nairobi(7) and _nairobi(7) != _nairobi(8)
    assert not any(
        m in EDGE_WEAR | PAVEMENT_WEAR for _, _, _, m in _worn(None)
    )  # off without a seed
