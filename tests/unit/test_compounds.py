from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, box

from fraktal import compounds, labels3d

CRS = "EPSG:32636"


def _layers(n: int = 6) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    road = gpd.GeoDataFrame({"id": ["r1"]}, geometry=[LineString([(0, 0), (400, 0)])], crs=CRS)
    parcels = gpd.GeoDataFrame(
        {"id": [f"p{i}" for i in range(n)], "road_id": ["r1"] * n},
        geometry=[box(10 + i * 40, 4, 40 + i * 40, 34) for i in range(n)],  # 30 x 30 m lots
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(
        {"id": [f"b{i}" for i in range(n)], "parcel_id": [f"p{i}" for i in range(n)]},
        geometry=[box(15 + i * 40, 10, 30 + i * 40, 25) for i in range(n)],
        crs=CRS,
    )
    return parcels, buildings, road


def test_walls_enclose_a_share_of_built_parcels_with_a_gate_gap() -> None:
    parcels, buildings, roads = _layers()
    out = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=1.0, irregular=0.0)
    assert len(out) == 6
    for c in out:
        total = sum(np.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in c["segments"])
        perimeter = 4 * (30 - 2 * compounds.INSET_M)
        assert perimeter - compounds.GATE_M - 0.5 < total < perimeter - compounds.GATE_M + 0.5
        assert len(c["segments"]) == 5  # four sides, one cut in two by the gate
        assert 1.8 <= c["height_m"] <= 2.4


def test_the_gate_faces_the_road() -> None:
    parcels, buildings, roads = _layers(1)
    (c,) = compounds.build(
        parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=1.0, irregular=0.0
    )
    ends = [(s[0], s[1]) for s in c["segments"]] + [(s[2], s[3]) for s in c["segments"]]
    near = [p for p in ends if abs(p[1] - 4.4) < 0.1]  # the road side: y is the inset parcel edge
    xs = sorted(p[0] for p in near)
    gaps = np.diff(xs)
    assert gaps.max() > 2.5  # the gate gap sits on that side


def test_share_zero_none_and_unbuilt_parcels_get_no_wall() -> None:
    parcels, buildings, roads = _layers()
    assert compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=0.0) == []
    empty = buildings.iloc[:0]
    assert compounds.build(parcels, empty, roads, (0.0, 0.0), 400.0, 1, share=1.0) == []
    some = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=0.5)
    assert 0 < len(some) < 6


def test_deterministic_and_seeded() -> None:
    parcels, buildings, roads = _layers()
    a = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=0.5)
    assert a == compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=0.5)
    assert a != compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 2, share=0.5)


def test_pack_boxes_face_outwards_and_every_compound_is_owned() -> None:
    parcels, buildings, roads = _layers(3)
    out = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=1.0)
    packed = compounds.pack(out)
    assert set(packed["face_owner"].tolist()) == {0, 1, 2}
    v, f = packed["vertices"].astype("float64"), packed["faces"]
    volume = np.einsum("ij,ij->", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])) / 6
    assert volume > 0  # one outward-facing closed surface per box
    assert compounds.pack([])["faces"].shape[0] == 0


def test_wall_is_label_class_8_and_every_biome_has_a_share() -> None:
    assert labels3d.CLASSES["wall"] == 8 and labels3d.CLASSES["yard_object"] == 9
    assert compounds.share_for("khartoum") > compounds.share_for("paddy_terraces")
    assert compounds.share_for("unknown") == compounds.DEFAULT_SHARE


def _many(biome: str) -> list[dict[str, Any]]:
    parcels, buildings, roads = _layers(6)
    out: list[dict[str, Any]] = []
    for seed in range(40):  # different seeds stand in for different tiles
        out += compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, seed, 1.0, biome)
    return out


def test_wall_kinds_follow_the_biome_and_sheet_fences_are_thin_and_metal() -> None:
    informal, khartoum = _many("informal_settlement"), _many("khartoum")

    def sheet(cs: list[dict[str, Any]]) -> float:
        return float(sum(c["kind"] == "sheet" for c in cs)) / len(cs)

    assert sheet(informal) > 0.4 > 0.2 > sheet(khartoum) > 0.0
    assert {c["kind"] for c in informal + khartoum} == set(compounds.KINDS)
    for c in informal:
        assert (c["material"] in compounds.SHEET_MATERIALS) == (c["kind"] == "sheet")
    assert (
        len({c["material"] for c in khartoum if c["kind"] != "sheet"}) >= 5
    )  # more than 4 colours


def test_piered_walls_add_posts_and_sheet_walls_are_thinner() -> None:
    parcels, buildings, roads = _layers(1)
    base = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, 1.0, irregular=0.0)[0]
    sizes = {}
    for kind in compounds.KINDS:
        sizes[kind] = len(compounds.pack([{**base, "kind": kind}])["faces"])
    assert sizes["piered"] > sizes["block"] > sizes["sheet"]  # posts and a cap on top
    thin = compounds.pack([{**base, "kind": "sheet"}])["vertices"]
    thick = compounds.pack([{**base, "kind": "block"}])["vertices"]
    assert np.ptp(thin[:8], axis=0).min() < np.ptp(thick[:8], axis=0).min()


def test_the_yard_point_is_inside_the_compound_and_clear_of_the_building() -> None:
    parcels, buildings, roads = _layers(6)
    for c in compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, 1.0):
        i = int(c["id"][1:])
        x, y = c["yard"]
        assert 10 + i * 40 < x < 40 + i * 40 and 4 < y < 34
        assert (
            not box(15 + i * 40, 10, 30 + i * 40, 25)
            .buffer(1.9)
            .contains(__import__("shapely.geometry", fromlist=["Point"]).Point(x, y))
        )


def _angle_set(c: dict[str, Any]) -> set[int]:
    return {
        round(np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 90) % 90
        for x0, y0, x1, y1 in c["segments"]
        if np.hypot(x1 - x0, y1 - y0) > 1.0
    }


def test_walls_come_in_at_angles_and_not_only_rectangles() -> None:
    parcels, buildings, roads = _layers(6)
    crooked = 0
    total = 0
    for seed in range(40):
        for c in compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, seed, 1.0):
            total += 1
            crooked += len(_angle_set(c) - {0}) > 0
    assert 0.25 < crooked / total < 0.8  # many walls lean, the rest stay square or notched


def test_irregular_walls_stay_inside_the_lot_and_clear_of_the_house() -> None:
    parcels, buildings, roads = _layers(6)
    for seed in range(40):
        for c in compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, seed, 1.0):
            i = int(c["id"][1:])
            lot, house = box(10 + i * 40, 4, 40 + i * 40, 34), box(15 + i * 40, 10, 30 + i * 40, 25)
            walls = MultiLineString([[(s[0], s[1]), (s[2], s[3])] for s in c["segments"]])
            assert lot.buffer(0.01).contains(walls)
            assert not walls.intersects(house)


def test_a_gate_is_still_cut_on_irregular_compounds() -> None:
    parcels, buildings, roads = _layers(6)
    for seed in range(20):
        for c in compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, seed, 1.0):
            starts = {(round(s[0], 1), round(s[1], 1)) for s in c["segments"]}
            ends = {(round(s[2], 1), round(s[3], 1)) for s in c["segments"]}
            assert len(starts - ends) == 1  # one wall end is not met by another: the gate


def test_some_walls_curve_and_a_curve_is_many_short_runs() -> None:
    parcels, buildings, roads = _layers(6)
    counts = [
        len(c["segments"])
        for seed in range(40)
        for c in compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, seed, 1.0)
    ]
    assert max(counts) >= 12 > 5  # a square has five runs; a bowed or rounded wall many more
    assert sum(n > 8 for n in counts) / len(counts) > 0.1


def test_wall_colours_map_to_a_texture_with_a_tint_and_sheet_metal_stays_flat() -> None:
    table = compounds.textures()
    assert set(table) == set(compounds.TEXTURE_BASE) == set(compounds.WALL_MATERIALS)
    for name, (base, tint) in table.items():
        assert base.startswith("wall-") and len(tint) == 3 and max(tint) <= compounds.TINT_MAX, name
    assert not set(table) & set(compounds.SHEET_MATERIALS)


def test_block_walls_get_a_coping_stone_and_sheet_fences_do_not() -> None:
    parcels, buildings, roads = _layers(1)
    base = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, 1.0, irregular=0.0)[0]
    faces = {k: len(compounds.pack([{**base, "kind": k}])["faces"]) for k in compounds.KINDS}
    n_segments = len(base["segments"])
    assert faces["block"] == 24 * n_segments  # a wall box and a cap box, 12 triangles each
    assert faces["sheet"] == 12 * n_segments
    top = compounds.pack([{**base, "kind": "block"}])["vertices"][:, 2].max()
    assert top == pytest.approx(base["height_m"] + compounds.CAP_M, abs=1e-4)


def test_a_textured_wall_averages_the_colour_it_was_given() -> None:
    from fraktal.propgen import MATERIALS

    for colour, (base, tint) in compounds.textures().items():
        mean = compounds.TEXTURE_MEAN[base]
        for want, t, m in zip(MATERIALS[colour][0], tint, mean, strict=True):
            assert t * m == pytest.approx(min(want, t * m), abs=0.01)  # capped tints fall short
            if t < compounds.TINT_MAX:
                assert t * m == pytest.approx(want, abs=0.005)


def test_the_recorded_texture_means_match_the_image_files() -> None:
    import json
    from pathlib import Path

    import rasterio

    assets = Path(__file__).resolve().parents[2] / "assets"
    for base, recorded in compounds.TEXTURE_MEAN.items():
        spec = json.loads((assets / "materials" / f"{base}.json").read_text())
        with rasterio.open(assets / spec["image"]) as src:
            measured = src.read().astype("float64").mean(axis=(1, 2)) / 255.0
        assert measured == pytest.approx(recorded, abs=0.01), base
