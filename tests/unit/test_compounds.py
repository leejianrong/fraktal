import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, box

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
    out = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=1.0)
    assert len(out) == 6
    for c in out:
        total = sum(np.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in c["segments"])
        perimeter = 4 * (30 - 2 * compounds.INSET_M)
        assert perimeter - compounds.GATE_M - 0.5 < total < perimeter - compounds.GATE_M + 0.5
        assert len(c["segments"]) == 5  # four sides, one cut in two by the gate
        assert 1.8 <= c["height_m"] <= 2.4


def test_the_gate_faces_the_road() -> None:
    parcels, buildings, roads = _layers(1)
    (c,) = compounds.build(parcels, buildings, roads, (0.0, 0.0), 400.0, 1, share=1.0)
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
