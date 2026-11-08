from pathlib import Path
from typing import Any

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Point

from fraktal.ids import substream
from fraktal.landcover_gen import WATER
from fraktal.roads import RoadConfig, RoadError, build_roads
from fraktal.terrain_gen import fractal_terrain
from fraktal.world import read_layer, write_layer

GSD = 5.0
TRANSFORM = Affine(GSD, 0, 500000.0, 0, -GSD, 1700000.0)
CRS = "EPSG:32636"


def cell_of(x: float, y: float) -> tuple[int, int]:
    col, row = ~TRANSFORM @ (x, y)
    return int(row), int(col)


def gentle(size: int = 300, seed: int = 1) -> np.ndarray:
    out: np.ndarray = fractal_terrain((size, size), GSD, seed, relief_m=12.0, largest_cell_m=600.0)
    return out


def dry(shape: tuple[int, int]) -> np.ndarray:
    out: np.ndarray = np.full(shape, 40, dtype="uint8")
    return out


def components(roads: Any) -> int:
    parent: dict[tuple[float, float], tuple[float, float]] = {}

    def find(p: tuple[float, float]) -> tuple[float, float]:
        parent.setdefault(p, p)
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for line in roads.geometry:
        pts = [(round(x, 3), round(y, 3)) for x, y in line.coords]
        for a, b in zip(pts, pts[1:], strict=False):
            parent[find(a)] = find(b)
    return len({find(p) for p in list(parent)})


def make(dem: np.ndarray, land: np.ndarray, seed: int = 1, **kwargs: Any) -> Any:
    return build_roads(dem, land, GSD, TRANSFORM, CRS, seed, **kwargs)


def test_the_network_is_one_connected_piece_with_both_road_classes() -> None:
    roads = make(gentle(), dry((300, 300)))
    assert set(roads["road_class"]) == {"arterial", "local"}
    assert components(roads) == 1
    assert (roads.loc[roads.road_class == "arterial", "width_m"] == 12.0).all()
    assert (roads["provenance"] == "procedural:roads").all()


def test_no_road_enters_water_without_a_bridge_feature() -> None:
    dem, land = gentle(), dry((300, 300))
    land[:, 140:150] = WATER  # a river across the whole region
    roads = make(dem, land, centres=[(150, 40), (150, 260), (60, 80)])
    bridges = roads[roads.bridge == 1.0]
    assert len(bridges) >= 1 and (bridges.road_class == "arterial").all()
    for _, road in roads.iterrows():
        in_water = [bool(land[cell_of(x, y)] == WATER) for x, y in road.geometry.coords]
        if road.bridge == 1.0:
            assert any(in_water) and not in_water[0] and not in_water[-1]  # banks at both ends
        else:
            assert not any(in_water)  # dry roads never touch water
    assert components(roads) == 1


def test_local_streets_never_enter_water() -> None:
    dem, land = gentle(), dry((300, 300))
    land[100:200, 120:180] = WATER
    roads = make(dem, land, centres=[(30, 30), (270, 270)])
    for _, road in roads[roads.road_class == "local"].iterrows():
        assert not any(land[cell_of(x, y)] == WATER for x, y in road.geometry.coords)


def test_same_seed_gives_identical_bytes_and_other_streams_do_not_matter(tmp_path: Path) -> None:
    dem, land = gentle(), dry((300, 300))
    a = make(dem, land)
    substream(1, "buildings").uniform(size=1000)  # building draws cannot move a road
    b = make(dem, land)
    write_layer(a, "road", tmp_path / "a.parquet")
    write_layer(b, "road", tmp_path / "b.parquet")
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()
    assert read_layer(tmp_path / "a.parquet", "road").equals(a)
    other = make(dem, land, seed=2)
    assert list(other.id) != list(a.id)


def arterial_ids(roads: Any) -> list[str]:
    return sorted(roads.loc[roads.road_class == "arterial", "id"])


def test_a_different_local_config_leaves_the_arterials_alone() -> None:
    dem, land = gentle(), dry((300, 300))
    base = make(dem, land)
    sparse = make(dem, land, config=RoadConfig(local_probability=0.2, local_spacing_m=200.0))
    assert arterial_ids(base) == arterial_ids(sparse)
    assert (sparse.road_class == "local").sum() < (base.road_class == "local").sum()


def test_centres_sit_on_dry_ground_and_failure_is_loud() -> None:
    dem = gentle()
    roads = make(dem, dry((300, 300)), centres=[(50, 50), (250, 250)])
    assert roads.geometry.distance(Point(*(TRANSFORM @ (50.5, 50.5)))).min() < 1e-6
    flooded = np.full((300, 300), WATER, dtype="uint8")
    with pytest.raises(RoadError, match="centres"):
        make(dem, flooded)
    with pytest.raises(RoadError, match="same shape"):
        make(dem, dry((10, 10)))


def turning_length(roads: Any) -> float:
    """Total absolute heading change along the arterials: large for staircases, small when fair."""
    total = 0.0
    for line in roads.loc[roads.road_class == "arterial", "geometry"]:
        pts = np.array(line.coords)
        d = np.diff(pts, axis=0)
        d = d[np.hypot(d[:, 0], d[:, 1]) > 1e-6]
        heading = np.arctan2(d[:, 1], d[:, 0])
        total += float(np.abs(np.diff(np.unwrap(heading))).sum())
    return total


def test_smoothing_keeps_the_network_connected_and_off_the_water_and_cuts_turning() -> None:
    dem, land = gentle(), dry((300, 300))
    land[:, 140:150] = WATER
    raw = make(dem, land, config=RoadConfig(smooth_tolerance_cells=0.0))
    fair = make(dem, land)
    assert components(fair) == 1
    assert len(raw) == len(fair)  # same segments, only their shape changes
    assert turning_length(fair) < turning_length(raw)
    for _, road in fair.iterrows():
        if road.bridge == 0.0:
            for x, y in road.geometry.coords:
                r, c = cell_of(x, y)
                assert land[r, c] != WATER


def test_smoothing_keeps_arterial_end_points() -> None:
    dem, land = gentle(), dry((300, 300))
    raw = make(dem, land, config=RoadConfig(smooth_tolerance_cells=0.0))
    fair = make(dem, land)

    def ends(roads: Any) -> set[tuple[tuple[float, float], tuple[float, float]]]:
        lines = roads.loc[roads.road_class == "arterial", "geometry"]
        return {(g.coords[0], g.coords[-1]) for g in lines}

    assert ends(raw) == ends(fair)


ORGANIC = RoadConfig(local_wander_deg=25.0, local_curl_deg=45.0, branch_rate=1.0, footpath_rate=1.5)


def _arterials(roads: Any) -> list[str]:
    return [g.wkt for g in roads[roads.road_class == "arterial"].geometry]


def _bow(line: Any) -> float:
    """How far a street bows away from its chord, as a share of its length."""
    chord = LineString([line.coords[0], line.coords[-1]])
    if chord.length == 0:
        return 0.0
    return float(max(chord.distance(Point(c)) for c in line.coords) / line.length)


def test_organic_streets_curve_branch_and_dead_end_while_arterials_stay_put() -> None:
    dem, land = gentle(), dry((300, 300))
    straight, organic = make(dem, land), make(dem, land, config=ORGANIC)
    assert _arterials(straight) == _arterials(organic)  # growth never touches the arterials
    assert set(organic["road_class"]) == {"arterial", "local", "footpath"}
    assert (organic.loc[organic.road_class == "footpath", "width_m"] == 1.6).all()
    locals_ = organic[organic.road_class == "local"]
    bend = np.mean([_bow(g) for g in locals_.geometry])
    flat = np.mean([_bow(g) for g in straight[straight.road_class == "local"].geometry])
    assert bend > 2 * flat and flat < 0.03  # streets bend where they used to be straight
    assert len(locals_) > (straight.road_class == "local").sum()  # side streets were added


def test_footpaths_and_side_streets_hang_off_the_network_and_one_piece_remains() -> None:
    organic = make(gentle(), dry((300, 300)), config=ORGANIC)
    assert components(organic) == 1
    feet = organic[organic.road_class == "footpath"]
    assert len(feet) > 3
    open_ends = 0
    for i, foot in feet.iterrows():
        others = organic.drop(index=i).geometry
        start, end = Point(foot.geometry.coords[0]), Point(foot.geometry.coords[-1])
        assert others.distance(start).min() < 1e-6  # it starts on a road
        open_ends += others.distance(end).min() > 1.0  # and most of them stop short: a dead end
    assert open_ends >= len(feet) // 2


def test_organic_growth_is_deterministic_and_seeded() -> None:
    dem, land = gentle(), dry((300, 300))
    a = make(dem, land, config=ORGANIC)
    assert [g.wkt for g in a.geometry] == [g.wkt for g in make(dem, land, config=ORGANIC).geometry]
    other = make(dem, land, seed=2, config=ORGANIC)
    assert [g.wkt for g in a.geometry] != [g.wkt for g in other.geometry]


def test_defaults_leave_the_world_id_alone_and_biomes_that_grow_change_it() -> None:
    from fraktal import procedural

    default = procedural._jsonable(procedural.ProceduralConfig())
    assert not {"local_wander_deg", "branch_rate", "footpath_rate"} & set(default["roads"])
    grown = procedural._jsonable(procedural.ProceduralConfig(roads=ORGANIC))
    assert grown["roads"]["branch_rate"] == 1.0
