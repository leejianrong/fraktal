from collections import Counter
from typing import Any

import numpy as np

from fraktal import trees3d


def test_small_trees_are_shrubs_and_the_choice_is_stable() -> None:
    assert trees3d.species_for("dense_urban", 5.0, 6.0, 2.0, 1) == "shrub"
    a = trees3d.species_for("dense_urban", 5.0, 6.0, 8.0, 1)
    assert a == trees3d.species_for("dense_urban", 5.0, 6.0, 8.0, 1)
    assert a in {"broadleaf", "palm"}


def test_the_mix_follows_the_biome() -> None:
    def share(biome: str) -> Counter[str]:
        return Counter(
            trees3d.species_for(biome, float(i), float(i * 7 % 100), 8.0, 0) for i in range(3000)
        )

    dry, wet = share("khartoum"), share("oil_palm_plantation")
    assert dry["broadleaf"] / 3000 > 0.75 and dry["palm"] / 3000 < 0.12
    assert wet["palm"] / 3000 > 0.45
    for biome, mix in trees3d.MIX.items():
        assert abs(sum(mix) - 1.0) < 1e-9, biome
    assert abs(sum(trees3d.DEFAULT_MIX) - 1.0) < 1e-9


def _scene(
    biome: str = "khartoum", walls: list[list[float]] | None = None, density: float = 1.0
) -> tuple[Any, Any, list[list[Any]]]:
    surface = np.full((400, 400), 60, dtype="uint8")  # bare ground, 0.5 m cells: a 200 m tile
    mask = np.zeros((400, 400), dtype="uint8")
    mask[100:140, 100:140] = 1  # a building
    return surface, mask, trees3d.scatter(surface, mask, 0.5, density, 3, biome, walls)


def test_scatter_is_deterministic_and_scales_with_density() -> None:
    _, _, a = _scene()
    assert a == _scene()[2]
    assert len(_scene(density=0.0)[2]) == 0
    assert len(_scene(density=2.0)[2]) > 1.6 * len(a)


def test_scatter_makes_thickets_small_shrubs_and_a_few_big_trees() -> None:
    _, _, trees = _scene()
    radius = np.array([t[2] for t in trees])
    assert 0.9 < np.median(radius) < 1.8 and 0.01 < (radius > 3.0).mean() < 0.1
    xy = np.array([[t[0], t[1]] for t in trees])

    # clustering: counts per 20 m cell vary far more than for an even random scatter
    counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=10, range=[[0, 200], [0, 200]])
    assert counts.var() / counts.mean() > 2.5  # 1.0 for a uniform random scatter
    assert sum(t[4] == "shrub" for t in trees) / len(trees) > 0.2


def test_scatter_keeps_trees_off_buildings_and_walls() -> None:
    wall = [[20.0, 20.0, 80.0, 20.0]]
    _, mask, trees = _scene(walls=wall)
    for x, y, radius, *_ in trees:
        assert mask[min(399, int((200 - y) / 0.5)), min(399, int(x / 0.5))] == 0
        near = abs(y - 20.0) < radius * 0.6 + 0.25 and 20.0 <= x <= 80.0
        assert not near or radius < 0.3


def test_yard_trees_stand_at_the_yard_point_for_a_share_of_compounds() -> None:
    compounds = [{"id": f"p{i}", "yard": [10.0 + i, 20.0]} for i in range(200)] + [
        {"id": "none", "yard": None}
    ]
    trees = trees3d.yard_trees(compounds, 1, "khartoum", 0.3)
    assert trees == trees3d.yard_trees(compounds, 1, "khartoum", 0.3)
    assert 40 < len(trees) < 80 and all(2.0 <= t[2] <= 3.4 and t[1] == 20.0 for t in trees)
    assert trees3d.yard_trees(compounds, 1, "khartoum", 0.0) == []


def test_varieties_refine_broadleaf_and_palm_by_biome_without_moving_the_species() -> None:
    def tally(biome: str, family: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in range(3000):
            v = trees3d.variety_for(biome, family, float(i), float(i * 7 % 100), 0)
            out[v] = out.get(v, 0) + 1
        return out

    dry, wet = tally("khartoum", "broadleaf"), tally("mangrove_coast", "broadleaf")
    assert dry["acacia"] / 3000 > 0.3 and dry["neem"] / 3000 > 0.3 and dry["broadleaf"] / 3000 < 0.3
    assert set(wet) == {"broadleaf"}  # the mangrove coast keeps the plain crown
    palms = tally("khartoum", "palm")
    assert palms["date_palm"] / 3000 > 0.6
    assert trees3d.variety_for("khartoum", "shrub", 1.0, 2.0, 0) == "shrub"
    assert trees3d.variety_for("khartoum", "broadleaf", 1.0, 2.0, 5) == trees3d.variety_for(
        "khartoum", "broadleaf", 1.0, 2.0, 5
    )
    assert trees3d.species_for("khartoum", 5.0, 6.0, 8.0, 1) in {"broadleaf", "palm", "shrub"}
