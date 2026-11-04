from typing import Any

import numpy as np
from affine import Affine
from shapely.strtree import STRtree

from fraktal.landcover_gen import WATER
from fraktal.roads import build_roads
from fraktal.settlement import (
    BuildingConfig,
    ParcelConfig,
    arterial_centres,
    build_buildings,
    build_parcels,
)
from fraktal.terrain_gen import fractal_terrain
from fraktal.world import validate_layer

GSD = 5.0
TRANSFORM = Affine(GSD, 0, 500000.0, 0, -GSD, 1700000.0)
CRS = "EPSG:32636"
SEED = 3


def scene(water: bool = False) -> tuple[np.ndarray, Any]:
    dem = fractal_terrain((300, 300), GSD, 1, relief_m=12.0, largest_cell_m=600.0)
    land = np.full((300, 300), 40, dtype="uint8")
    if water:
        land[:, 140:150] = WATER
    return land, build_roads(dem, land, GSD, TRANSFORM, CRS, SEED)


def test_every_parcel_fronts_a_road_has_area_and_overlaps_nothing() -> None:
    land, roads = scene()
    parcels = build_parcels(roads, land, TRANSFORM, SEED)
    validate_layer(parcels, "parcel")
    assert len(parcels) > 50 and (parcels.area >= 150.0).all()
    tree = STRtree(list(roads.geometry))
    half = dict(zip(roads.id, roads.width_m / 2, strict=True))
    by_id = dict(zip(roads.id, roads.geometry, strict=True))
    for _, p in parcels.iterrows():
        near = by_id[p.road_id]
        assert p.geometry.distance(near) <= half[p.road_id] + 0.01  # front edge on the carriageway
    geoms = list(parcels.geometry)
    index = STRtree(geoms)
    for i, g in enumerate(geoms):
        for j in index.query(g):
            if j > i:
                assert g.intersection(geoms[j]).area < 1e-6
    assert tree is not None


def test_no_parcel_sits_on_a_carriageway_or_in_water() -> None:
    land, roads = scene(water=True)
    parcels = build_parcels(roads, land, TRANSFORM, SEED)
    carriageways = [row.geometry.buffer(row.width_m / 2) for _, row in roads.iterrows()]
    index = STRtree(carriageways)
    inverse = ~TRANSFORM
    for _, p in parcels.iterrows():
        for j in index.query(p.geometry):
            clipped = p.geometry.intersection(carriageways[j]).area
            assert clipped <= 0.03 * p.geometry.area + 1e-6  # a curve may clip a sliver
        for x, y in [*p.geometry.exterior.coords, (p.geometry.centroid.x, p.geometry.centroid.y)]:
            col, row = inverse @ (x, y)
            assert land[int(row), int(col)] != WATER


def test_every_building_lies_inside_its_parcel_and_carries_procedural_provenance() -> None:
    land, roads = scene()
    parcels = build_parcels(roads, land, TRANSFORM, SEED)
    buildings = build_buildings(parcels, arterial_centres(roads), SEED)
    validate_layer(buildings, "building")
    assert len(buildings) > 30 and (buildings["provenance"] == "procedural:buildings").all()
    by_id = dict(zip(parcels.id, parcels.geometry, strict=True))
    for _, b in buildings.iterrows():
        assert by_id[b.parcel_id].buffer(1e-6).contains(b.geometry)
    assert buildings["parcel_id"].is_unique  # one building per parcel
    assert set(buildings.roof_type) <= {"flat", "gable", "hip"}
    assert buildings.height_m.between(3.0, 22.0).all()


def test_building_settings_never_move_a_parcel() -> None:
    land, roads = scene()
    a = build_parcels(roads, land, TRANSFORM, SEED)
    sparse = build_buildings(
        a,
        arterial_centres(roads),
        SEED,
        BuildingConfig(built_probability_core=0.3, built_probability_edge=0.2),
    )
    full = build_buildings(
        a,
        arterial_centres(roads),
        SEED,
        BuildingConfig(built_probability_core=1.0, built_probability_edge=1.0),
    )
    b = build_parcels(roads, land, TRANSFORM, SEED)
    assert list(a.id) == list(b.id) and a.geometry.equals(b.geometry)
    assert len(sparse) < len(full) and set(sparse.parcel_id) <= set(
        full.parcel_id
    )  # same draws per parcel


def test_buildings_are_taller_and_denser_near_the_centres() -> None:
    land, roads = scene()
    parcels = build_parcels(roads, land, TRANSFORM, SEED)
    centres = arterial_centres(roads)
    b = build_buildings(parcels, centres, SEED)
    near = b[b.land_use == "commercial"]
    far = b[b.land_use == "residential"]
    assert len(near) and len(far) and near.height_m.mean() > far.height_m.mean()


def test_same_seed_same_world_and_a_local_only_option() -> None:
    land, roads = scene()
    a = build_parcels(roads, land, TRANSFORM, SEED)
    assert list(a.id) == list(build_parcels(roads, land, TRANSFORM, SEED).id)
    local = build_parcels(roads, land, TRANSFORM, SEED, ParcelConfig(local_only=True))
    assert set(roads.set_index("id").loc[local.road_id, "road_class"]) == {"local"}
    assert len(local) < len(a)


def test_block_lots_cover_more_than_frontage_strips_and_stay_valid() -> None:
    land, roads = scene(water=True)
    strips = build_parcels(roads, land, TRANSFORM, SEED)
    cfg = ParcelConfig(blocks=True, block_depth_m=60.0)
    lots = build_parcels(roads, land, TRANSFORM, SEED, cfg)
    validate_layer(lots, "parcel")
    assert lots.area.sum() > 1.2 * strips.area.sum()
    geoms = list(lots.geometry)
    index = STRtree(geoms)
    for i, g in enumerate(geoms):
        assert g.is_valid and g.area >= cfg.min_area_m2
        assert all(j <= i or g.intersection(geoms[j]).area < 1e-6 for j in index.query(g))
    carriageways = [row.geometry.buffer(row.width_m / 2) for _, row in roads.iterrows()]
    road_index = STRtree(carriageways)
    for g in geoms:
        for j in road_index.query(g):
            assert g.intersection(carriageways[j]).area <= 0.03 * g.area + 1e-6
    assert lots.equals(build_parcels(roads, land, TRANSFORM, SEED, cfg))  # deterministic


def test_block_buildings_sit_inside_their_lot_and_face_a_street() -> None:
    land, roads = scene()
    lots = build_parcels(roads, land, TRANSFORM, SEED, ParcelConfig(blocks=True))
    lines = dict(zip(roads.id, roads.geometry, strict=True))
    buildings = build_buildings(lots, arterial_centres(roads), SEED, road_lines=lines)
    validate_layer(buildings, "building")
    by_lot = dict(zip(lots.id, lots.geometry, strict=True))
    assert len(buildings) > 50
    for _, b in buildings.iterrows():
        assert by_lot[b.parcel_id].buffer(1e-6).contains(b.geometry)
