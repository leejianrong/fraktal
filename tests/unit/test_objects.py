from pathlib import Path

import numpy as np
import pytest
from test_render import CENTER, SMALL, one_building, terrain  # noqa: F401

from fraktal.kernels.objects import paint_objects
from fraktal.objects import CAR_CLASSES, place, roof_pattern, scatter
from fraktal.render import RenderConfig, render_region


def test_golden_rectangle_and_dome() -> None:
    shapes = np.array([[3.0, 3.0, 1.0, 1.0, 0.0, 1.0], [8.0, 8.0, 2.0, 2.0, 0.0, 0.0]])
    height, colour, covered = paint_objects(
        (12, 12),
        shapes,
        np.array([2.0, 6.0]),
        np.array([[200, 0, 0], [0, 200, 0]]),
        np.zeros((12, 12)),
    )
    square = ["".join("#" if v else "." for v in row[2:5]) for row in covered[2:5]]
    assert square == ["###", "###", "###"]
    assert height[3, 3] == 2.0 and (colour[3, 3] == [200, 0, 0]).all()
    assert height[8, 8] == pytest.approx(6.0)  # dome peak
    assert 0 < height[8, 9] < 6.0 and covered[8, 10] == 1 and covered[8, 11] == 0
    assert covered.sum() == 9 + int(covered[6:11, 6:11].sum())


def test_blocked_pixels_stay_clear_and_rotation_changes_the_footprint() -> None:
    blocked = np.zeros((12, 12))
    blocked[3, 3] = 1
    shapes = np.array([[6.0, 6.0, 3.0, 1.0, 0.0, 1.0]])
    _, _, flat = paint_objects((12, 12), shapes, np.ones(1), np.ones((1, 3)), np.zeros((12, 12)))
    _, _, turned = paint_objects(
        (12, 12),
        shapes * [1, 1, 1, 1, 0, 1] + [0, 0, 0, 0, np.pi / 2, 0],
        np.ones(1),
        np.ones((1, 3)),
        np.zeros((12, 12)),
    )
    assert flat[6, 3:10].all() and not turned[6, 3:10].all() and turned[3:10, 6].all()
    _, _, clear = paint_objects(
        (12, 12), np.array([[3.0, 3.0, 2.0, 2.0, 0.0, 1.0]]), np.ones(1), np.ones((1, 3)), blocked
    )
    assert clear[3, 3] == 0 and clear[3, 4] == 1


def test_kernel_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="N, 6"):
        paint_objects((4, 4), np.zeros((2, 5)), np.ones(2), np.ones((2, 3)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="same length"):
        paint_objects((4, 4), np.zeros((2, 6)), np.ones(1), np.ones((2, 3)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="grid shape"):
        paint_objects((4, 4), np.zeros((1, 6)), np.ones(1), np.ones((1, 3)), np.zeros((3, 3)))


def land(value: int = 50, size: int = 400) -> np.ndarray:
    out: np.ndarray = np.full((size, size), value, dtype="uint8")
    return out


def test_placement_is_deterministic_scales_with_density_and_respects_classes() -> None:
    lc, none = land(), np.zeros((400, 400), dtype="uint8")
    a = place(lc, none, 0.5, 1.0, 3)
    assert all((x == y).all() for x, y in zip(a, place(lc, none, 0.5, 1.0, 3), strict=True))
    assert len(place(lc, none, 0.5, 2.0, 3)[0]) > 1.7 * len(a[0])
    assert not (
        place(lc, none, 0.5, 1.0, 4)[0].shape == a[0].shape
        and (place(lc, none, 0.5, 1.0, 4)[0] == a[0]).all()
    )
    water = place(land(80), none, 0.5, 1.0, 3)
    assert len(water[0]) == 0
    only_trees = place(land(10), none, 0.5, 1.0, 3)  # tree cover: no vehicles on it
    assert (only_trees[0][:, 5] == 0).all() and 10 not in CAR_CLASSES


def test_objects_never_land_on_buildings_and_cast_height() -> None:
    lc = land()
    mask = np.zeros((400, 400), dtype="uint8")
    mask[100:300, 100:300] = 1
    height, colour, covered = scatter(lc, mask, 0.5, 2.0, 1)
    assert covered.sum() > 0 and not covered[mask > 0].any()
    assert height.max() > 3.0 and (height[covered == 0] == 0).all()


def test_roof_pattern_stripes_roofs_only_and_follows_the_id() -> None:
    index = np.zeros((40, 40), dtype="uint32")
    index[5:20, 5:20], index[22:38, 22:38] = 1, 2
    ids = ["0123456789abcdef", "fedcba9876543210"]
    gain = roof_pattern(index, ids, 0.5, 0.2)
    assert (gain[index == 0] == 1).all()
    assert gain[index > 0].min() < 0.9 and gain[index > 0].max() > 1.1
    again = roof_pattern(index, ids, 0.5, 0.2)
    assert (gain == again).all()
    assert not np.allclose(gain[index == 1].mean(), 0)  # both roofs patterned
    assert (roof_pattern(index, ids, 0.5, 0.0) == 1).all()


def test_render_with_objects_keeps_labels_and_is_deterministic(
    tmp_path: Path,
    terrain: Path,  # noqa: F811
) -> None:
    layer = one_building(CENTER, 30, 6.0)
    cfg = RenderConfig(**{**SMALL.__dict__, "variation": 0.15, "objects": 8.0, "roof_detail": 0.2})
    a = render_region(terrain, layer, CENTER, tmp_path / "a", cfg)
    b = render_region(terrain, layer, CENTER, tmp_path / "b", cfg)
    plain = render_region(
        terrain,
        layer,
        CENTER,
        tmp_path / "p",
        RenderConfig(**{**SMALL.__dict__, "variation": 0.15}),
    )
    assert a["dropped"] == [] and a["tiles"] == b["tiles"]
    assert [t["mask_sha256"] for t in a["tiles"]] == [t["mask_sha256"] for t in plain["tiles"]]
    import hashlib

    def digest(d: Path) -> str:
        return hashlib.sha256((d / "tiles" / "r00_c00.tif").read_bytes()).hexdigest()

    assert digest(tmp_path / "a") == digest(tmp_path / "b") != digest(tmp_path / "p")


def road_strip() -> np.ndarray:
    road = np.zeros((400, 400), dtype="uint8")
    road[190:210, :] = 202  # a 10 m east-west street at 0.5 m pixels
    return road


def test_cars_stand_on_the_road_and_point_along_it() -> None:
    road, none = road_strip(), np.zeros((400, 400), dtype="uint8")
    lc = land(30)
    shapes, _, _ = place(lc, none, 0.5, 8.0, 3, road)
    cars = shapes[shapes[:, 5] == 1]
    assert len(cars) > 5
    assert (road[cars[:, 0].astype(int), cars[:, 1].astype(int)] > 0).all()
    along = np.abs(np.sin(cars[:, 4]))  # a horizontal street: heading near 0 or pi
    assert np.median(along) < 0.15
    again, _, _ = place(lc, none, 0.5, 8.0, 3, road)
    assert (again == shapes).all()


def test_roads_leave_trees_alone_and_cars_avoid_buildings() -> None:
    road, mask = road_strip(), np.zeros((400, 400), dtype="uint8")
    mask[185:215, 100:200] = 1
    lc = land(10)  # tree cover: a car can only be on a road
    shapes, _, _ = place(lc, mask, 0.5, 8.0, 3, road)
    cars = shapes[shapes[:, 5] == 1]
    assert len(cars) > 0
    assert not mask[cars[:, 0].astype(int), cars[:, 1].astype(int)].any()
