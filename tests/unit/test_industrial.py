from typing import Any

import numpy as np
from affine import Affine
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from fraktal import building_grammar as grammar
from fraktal import props_scene
from fraktal.industrial import LAND_USE, IndustrialConfig, carve
from fraktal.labels3d import CLASSES
from fraktal.roads import build_roads
from fraktal.settlement import ParcelConfig, arterial_centres, build_buildings, build_parcels
from fraktal.terrain_gen import fractal_terrain
from fraktal.world import validate_layer

GSD = 5.0
TRANSFORM = Affine(GSD, 0, 500000.0, 0, -GSD, 1700000.0)
CRS = "EPSG:32636"
SEED = 3
CFG = IndustrialConfig(sites_per_km=6.0)


def _world() -> tuple[np.ndarray, Any, Any]:
    dem = fractal_terrain((300, 300), GSD, 1, relief_m=12.0, largest_cell_m=600.0)
    land = np.full((300, 300), 40, dtype="uint8")
    roads = build_roads(dem, land, GSD, TRANSFORM, CRS, SEED)
    return land, roads, build_parcels(roads, land, TRANSFORM, SEED, ParcelConfig())


def test_no_sites_by_default_and_the_parcels_come_back_unchanged() -> None:
    land, roads, parcels = _world()
    same, halls = carve(parcels, roads, land, TRANSFORM, SEED, IndustrialConfig())
    assert same is parcels and halls == []


def test_sites_replace_parcels_stay_off_roads_and_hold_long_halls() -> None:
    land, roads, parcels = _world()
    out, halls = carve(parcels, roads, land, TRANSFORM, SEED, CFG)
    validate_layer(out, "parcel")
    again = carve(parcels, roads, land, TRANSFORM, SEED, CFG)
    assert out.equals(again[0])
    assert [h["geometry"].wkt for h in halls] == [h["geometry"].wkt for h in again[1]]
    sites = out[out["land_use"] == LAND_USE]
    assert len(sites) >= 2 and len(out) < len(parcels) + len(sites)  # houses made way
    geoms = list(out.geometry)
    index = STRtree(geoms)
    for i, g in enumerate(geoms):  # nothing overlaps: houses, sites and sites
        assert all(j <= i or g.intersection(geoms[j]).area < 1e-6 for j in index.query(g))
    carriageways = [r.geometry.buffer(r.width_m / 2) for _, r in roads.iterrows()]
    for site in sites.geometry:
        assert site.area > 1500
        assert all(site.intersection(c).area < 0.03 * site.area for c in carriageways)
    by_id = dict(zip(sites.id, sites.geometry, strict=True))
    assert halls and {h["parcel_id"] for h in halls} <= set(by_id)
    for h in halls:
        g = h["geometry"]
        assert by_id[h["parcel_id"]].buffer(1e-6).contains(g) and h["land_use"] == LAND_USE
        rect = np.asarray(g.minimum_rotated_rectangle.exterior.coords[:4])
        a, b = np.hypot(*(rect[1] - rect[0])), np.hypot(*(rect[2] - rect[1]))
        assert g.area >= 300 and max(a, b) / min(a, b) >= 1.8 and 6.5 <= h["height_m"] <= 11.0
    homes = out[out["land_use"] != LAND_USE]
    layer = build_buildings(homes, arterial_centres(roads), SEED, extra=halls)
    validate_layer(layer, "building")
    assert (layer["land_use"] == LAND_USE).sum() == len(halls)


def test_a_long_flat_footprint_is_an_industrial_hall_and_a_house_is_not() -> None:
    hall = [[0, 0], [40, 0], [40, 16], [0, 16], [0, 0]]
    house = [[0, 0], [10, 0], [10, 8], [0, 8], [0, 0]]
    assert grammar.build("hall0001", hall, 8.0, None, 1).typology == "industrial_hall"
    assert grammar.build("home0001", house, 8.0, None, 1).typology != "industrial_hall"


def test_yard_props_fill_the_apron_clear_of_the_halls_with_yard_objects() -> None:
    yard = Polygon([(0, 0), (60, 0), (60, 18), (0, 18)])
    site = {"id": "s1", "yard": yard, "yaw": 0.0}
    props = props_scene.yard_props([site], 5)
    assert props == props_scene.yard_props([site], 5) and len(props) >= 8
    assert {p["class"] for p in props} == {CLASSES["yard_object"]}
    assert {p["kind"] for p in props} <= {"container", "water_tank", "silo"}
    for p in props:
        assert yard.contains(Point(p["x"], p["y"]))
    assert props_scene.yard_props([{**site, "yard": Polygon()}], 5) == []
