import dataclasses
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from fraktal import agents, biomes, procedural
from fraktal.procedural import generate_world

CENTER = (32.55, 15.64)
BOUNDS = (0.0, 0.0, 500.0, 500.0)


def _roads(
    lines: list[tuple[str, float, list[tuple[float, float]]]], bridge: float = 0.0
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "road_class": [c for c, _, _ in lines],
            "width_m": [w for _, w, _ in lines],
            "bridge": [bridge] * len(lines),
        },
        geometry=[LineString(xy) for _, _, xy in lines],
    )


CROSS = _roads([
    ("arterial", 12.0, [(0, 250), (500, 250)]),
    ("local", 6.0, [(250, 0), (250, 250)]),  # a T-junction from the south ...
    ("local", 6.0, [(250, 250), (250, 500)]),  # ... and the road goes on to the north
])  # fmt: skip
TIMES = np.arange(0.0, 30.0, 0.5)


def test_roads_are_noded_where_they_meet() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    assert len(g.edges) == 4 and len(g.nodes) == 5
    assert sorted(len(h) for h in g.out) == [1, 1, 1, 1, 4]  # one four-way junction
    assert g.total_length_m() == pytest.approx(500 + 250 + 250, abs=1.0)


def test_a_t_junction_and_a_bridge_are_handled() -> None:
    t = _roads(
        [("arterial", 12.0, [(0, 250), (500, 250)]), ("local", 6.0, [(250, 0), (250, 249.2)])]
    )
    g = agents.build_graph(t, BOUNDS)
    assert sorted(len(h) for h in g.out) == [1, 1, 1, 3]  # the end within a metre joins the line
    bridged = _roads([("arterial", 12.0, [(0, 250), (500, 250)])], bridge=1.0)
    assert agents.build_graph(bridged, BOUNDS).edges == []


def test_roads_are_clipped_to_the_tile() -> None:
    long = _roads([("local", 6.0, [(-300, 100), (800, 100)])])
    g = agents.build_graph(long, BOUNDS)
    assert g.total_length_m() == pytest.approx(500.0, abs=0.5)


def test_cars_keep_to_the_drive_side_and_travel_both_ways() -> None:
    road = _roads([("arterial", 12.0, [(0, 250), (500, 250)])])
    g = agents.build_graph(road, BOUNDS)
    for side, sign in (("right", -1.0), ("left", 1.0)):
        cfg = agents.TrafficConfig(cars_per_km=20.0, drive_side=side)
        tr = agents.simulate_cars(g, TIMES, 1, cfg)
        east = np.cos(tr.heading[:, 10]) > 0  # driving east: the right-hand lane is south
        assert east.any() and (~east).any()
        mid = tr.y[:, 10] - 250.0
        assert (np.sign(mid[east]) == sign).all() and (np.sign(mid[~east]) == -sign).all()
        inside = (tr.x[:, 10] > 30) & (tr.x[:, 10] < 470)  # not turning round at a tile edge
        assert np.abs(np.abs(mid[inside]) - 3.0).max() < 0.5  # a quarter of the width off centre


def test_cars_stay_on_the_roads_and_move_plausibly() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    cfg = agents.TrafficConfig(cars_per_km=15.0)
    tr = agents.simulate_cars(g, TIMES, 2, cfg)
    assert len(tr.ids) == 15 and tr.x.shape == (15, len(TIMES))
    centre = CROSS.geometry.union_all()
    from shapely.geometry import Point

    inside = (
        (tr.x > 0) & (tr.x < 500) & (tr.y > 0) & (tr.y < 500)
    )  # cars may drive off the tile edge
    far = max(centre.distance(Point(x, y)) for x, y in zip(tr.x[inside], tr.y[inside], strict=True))
    assert far < 5.0  # within a lane and a bend of a road centre line
    step = np.hypot(np.diff(tr.x, axis=1), np.diff(tr.y, axis=1))
    assert step.max() < 12.5 * 1.2 * 0.5 + 0.5  # no teleports: at most speed times the 0.5 s step
    assert tr.speed.max() < cfg.arterial_speed_mps * (1 + cfg.speed_spread) + 0.01
    assert tr.speed.mean() > 2.0  # they actually drive


def test_a_sharp_corner_is_taken_slowly() -> None:
    bend = _roads([("arterial", 12.0, [(50, 50), (250, 50), (250, 250)])])
    g = agents.build_graph(bend, BOUNDS)
    tr = agents.simulate_cars(
        g, np.arange(0.0, 40.0, 0.5), 4, agents.TrafficConfig(cars_per_km=20.0)
    )
    nearest = np.hypot(tr.x - 250.0, tr.y - 50.0)
    at_corner = tr.speed[nearest < 6.0]
    assert len(at_corner) and at_corner.mean() < 0.7 * 11.0


def test_the_same_seed_gives_the_same_tracks_and_another_seed_does_not() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    a = agents.simulate_cars(g, TIMES, 7)
    b = agents.simulate_cars(g, TIMES, 7)
    c = agents.simulate_cars(g, TIMES, 8)
    assert a.ids == b.ids and np.array_equal(a.x, b.x) and np.array_equal(a.heading, b.heading)
    assert not np.array_equal(a.x, c.x)


def test_cars_do_not_start_on_top_of_each_other() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    tr = agents.simulate_cars(g, TIMES, 5, agents.TrafficConfig(cars_per_km=12.0))
    pts = np.stack([tr.x[:, 0], tr.y[:, 0]], axis=1)
    d = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1)) + np.eye(len(pts)) * 1e9
    assert d.min() > 2.0


def test_no_roads_or_no_cars_give_empty_tracks_and_bad_input_is_refused() -> None:
    empty = agents.build_graph(_roads([("local", 6.0, [(10, 10), (12, 10)])]), BOUNDS)
    tr = agents.simulate_cars(empty, TIMES, 0)
    assert tr.ids == [] and tr.x.shape == (0, len(TIMES))
    g = agents.build_graph(CROSS, BOUNDS)
    with pytest.raises(agents.AgentError, match="drive_side"):
        agents.simulate_cars(g, TIMES, 0, agents.TrafficConfig(drive_side="up"))
    with pytest.raises(agents.AgentError, match="ascending"):
        agents.simulate_cars(g, [3.0, 1.0], 0)


def test_the_tracks_serialise_to_json() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    tr = agents.simulate_cars(g, TIMES[:5], 1, agents.TrafficConfig(cars_per_km=8.0))
    data = json.loads(json.dumps(tr.as_dict()))
    assert data["kind"] == "car" and len(data["ids"]) == len(data["x"]) == len(data["speed"])


def test_biomes_set_the_drive_side_and_it_does_not_change_the_world_id() -> None:
    sides = {n: biomes.resolve(n).roads.drive_side for n in biomes.names()}
    assert sides["khartoum"] == "right" and sides["dense_urban"] == "left"
    cfg = biomes.resolve("dense_urban")
    flipped = dataclasses.replace(cfg, roads=dataclasses.replace(cfg.roads, drive_side="right"))
    assert procedural._jsonable(cfg) == procedural._jsonable(flipped)
    assert biomes.resolve("khartoum", {"roads": {"drive_side": "left"}}).roads.drive_side == "left"


def test_a_generated_world_gives_cars_on_its_roads_in_the_tile_frame(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("dense_urban", {"size_m": 3000.0}))
    roads, bounds, side = agents.tile_roads(world, CENTER, 2000.0)
    assert side == "left" and bounds[2] == pytest.approx(2000.0, abs=1.0)
    g = agents.build_graph(roads, bounds)
    tr = agents.simulate_cars(
        g, np.arange(0.0, 10.0, 0.5), 3, agents.TrafficConfig(drive_side=side)
    )
    assert len(tr.ids) > 50
    inside = (tr.x > 0) & (tr.x < bounds[2]) & (tr.y > 0) & (tr.y < bounds[3])
    assert inside.mean() > 0.8  # most of the time they are in the tile; at an edge they drive off


def _pair_distances(tr: agents.Tracks) -> np.ndarray:
    pts = np.stack([tr.x, tr.y], axis=-1)
    d = np.hypot(*(pts[:, None] - pts[None]).transpose(3, 0, 1, 2))
    d[np.arange(len(pts)), np.arange(len(pts))] = np.inf
    return d  # type: ignore[no-any-return]


def test_cars_keep_a_gap_and_a_busy_clip_overlaps_far_less_than_without_interaction() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    busy = np.arange(0.0, 60.0, 0.5)
    with_it = agents.simulate_cars(g, busy, 2, agents.TrafficConfig(cars_per_km=40.0))
    without = agents.simulate_cars(
        g, busy, 2, agents.TrafficConfig(cars_per_km=40.0, interact=False)
    )
    near_with = int((_pair_distances(with_it) < 2.0).sum())
    near_without = int((_pair_distances(without) < 2.0).sum())
    assert near_without > 50 and near_with < 0.25 * near_without
    # on the open arterial, cars heading the same way in the same lane keep a car length apart
    a, b = with_it, with_it
    same_lane = np.zeros(_pair_distances(a).shape, dtype=bool)
    on_artery = np.abs(a.y - 250.0) < 4.5
    for t in range(len(busy)):
        dy = np.abs(a.y[:, t, None] - a.y[None, :, t])
        dh = np.abs(np.cos(a.heading[:, t, None] - a.heading[None, :, t]) - 1.0)
        same_lane[:, :, t] = (
            (dy < 1.0) & (dh < 0.05) & on_artery[:, t, None] & on_artery[None, :, t]
        )
    gaps = _pair_distances(b)[same_lane]
    assert len(gaps) == 0 or gaps.min() > 4.5


def test_a_busy_junction_does_not_lock_up() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    tr = agents.simulate_cars(
        g, np.arange(0.0, 90.0, 0.5), 2, agents.TrafficConfig(cars_per_km=40.0)
    )
    assert (tr.speed[:, -1] < 0.3).sum() <= 2  # a few may be waiting, but traffic still flows
    assert tr.speed[:, -1].mean() > 3.0


def test_people_walk_on_the_verges_at_walking_pace_with_stops() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    times = np.arange(0.0, 60.0, 0.5)
    cfg = agents.WalkerConfig(people_per_km=30.0, stops_per_min=3.0)
    people = agents.simulate_people(g, times, 3, cfg)
    assert people.kind == "person" and len(people.ids) == 30 and people.ids[0] == "person-0000"
    assert people.speed.max() <= cfg.walk_speed_mps * (1 + cfg.speed_spread) + 1e-9
    assert (people.speed == 0).any() and (people.speed > 0.8).any()  # they stop and they walk
    from shapely.geometry import Point

    centre = CROSS.geometry.union_all()
    inside = (people.x > 5) & (people.x < 495) & (people.y > 5) & (people.y < 495)
    dist = np.array(
        [
            centre.distance(Point(x, y))
            for x, y in zip(people.x[inside], people.y[inside], strict=True)
        ]
    )
    assert dist.min() > 0.0 and np.median(dist) > 4.0  # beside the road, not on it
    assert np.percentile(dist, 90) < 7.2 + 0.5  # and no farther than the verge of the wide road
    moved = np.hypot(people.x[:, -1] - people.x[:, 0], people.y[:, -1] - people.y[:, 0])
    assert (moved > 5.0).mean() > 0.8
    step = np.hypot(np.diff(people.x, axis=1), np.diff(people.y, axis=1))
    assert step.max() < 1.7 * 0.5 + 0.5  # no jumps


def test_people_are_deterministic_and_independent_of_the_drive_side() -> None:
    g = agents.build_graph(CROSS, BOUNDS)
    a = agents.simulate_people(g, TIMES, 5)
    b = agents.simulate_people(g, TIMES, 5)
    c = agents.simulate_people(g, TIMES, 6)
    assert np.array_equal(a.x, b.x) and not np.array_equal(a.x, c.x)
    cars = agents.simulate_cars(g, TIMES, 5, agents.TrafficConfig(drive_side="left"))
    assert len(cars.ids) > 0 and np.array_equal(a.x, agents.simulate_people(g, TIMES, 5).x)


def test_no_roads_gives_no_people_and_bad_input_is_refused() -> None:
    empty = agents.build_graph(_roads([("local", 6.0, [(10, 10), (12, 10)])]), BOUNDS)
    assert agents.simulate_people(empty, TIMES, 0).ids == []
    g = agents.build_graph(CROSS, BOUNDS)
    with pytest.raises(agents.AgentError, match="ascending"):
        agents.simulate_people(g, [2.0, 1.0], 0)
    with pytest.raises(agents.AgentError, match="positive"):
        agents.simulate_people(g, TIMES, 0, agents.WalkerConfig(walk_speed_mps=0.0))
