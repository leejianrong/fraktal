"""Overhead cables between roadside poles (FRK-147)."""

from __future__ import annotations

import numpy as np

from fraktal import cables

# two rows of poles 30 m apart along a road, zigzag as the exporter makes them
POLES = [[10.0 + 28.0 * k, 50.0 + (6.0 if k % 2 else -6.0), 9.0] for k in range(8)]


def test_biomes_without_a_style_have_no_wires_and_keep_their_poles() -> None:
    assert cables.build(POLES, "dense_urban", 1) == []
    assert cables.pole_plan("dense_urban", 40.0, 8.0) == (40.0, 8.0)
    assert cables.pole_plan("manila_quezon", 40.0, 8.0) == (28.0, 9.0)


def test_every_span_carries_the_styles_wires_and_hangs_below_the_pole_tops() -> None:
    wires = cables.build(POLES, "manila_quezon", 1)
    assert len(wires) > 0 and len(wires) % 5 == 0  # five wires per span in Manila-Quezon
    for line, normal in wires:
        assert line.shape == (cables.SEGMENTS + 1, 3)
        assert line[:, 2].max() <= 9.0 and line[:, 2].min() > 3.0  # below the tops, off the ground
        assert abs(float(np.hypot(normal[0], normal[1])) - 1.0) < 1e-9 and normal[2] == 0.0
        assert line[len(line) // 2, 2] < line[0, 2]  # it sags in the middle


def test_same_world_same_wires_and_a_new_seed_changes_the_sag() -> None:
    a, b = cables.build(POLES, "nairobi_centre", 4), cables.build(POLES, "nairobi_centre", 4)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b, strict=True))
    c = cables.build(POLES, "nairobi_centre", 5)
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a, c, strict=True))


def test_cities_differ_in_how_many_wires_they_string() -> None:
    manila = len(cables.build(POLES, "manila_quezon", 1))
    khartoum = len(cables.build(POLES, "khartoum_city", 1))
    assert manila > khartoum > 0


def test_pack_gives_two_triangles_per_piece_and_valid_indices() -> None:
    wires = cables.build(POLES, "khartoum_city", 1)
    packed = cables.pack(wires)
    assert len(packed["faces"]) == 2 * cables.SEGMENTS * len(wires)
    assert packed["faces"].max() < len(packed["vertices"])
    assert len(packed["face_material"]) == len(packed["faces"]) == len(packed["face_owner"])
    empty = cables.pack([])
    assert empty["faces"].shape == (0, 3) and len(empty["materials"]) == len(cables.MATERIALS)


def test_poles_too_close_or_too_far_apart_are_not_wired() -> None:
    assert cables.build([[0.0, 0.0, 9.0], [2.0, 0.0, 9.0]], "manila_quezon", 1) == []
    assert cables.build([[0.0, 0.0, 9.0], [300.0, 0.0, 9.0]], "manila_quezon", 1) == []
