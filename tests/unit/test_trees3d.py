from collections import Counter

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


def _scene(biome: str = "khartoum", walls: list[list[float]] | None = None, density: float = 1.0):
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

    # clustering: nearest-neighbour distances are shorter than for the same count spread evenly
    def mean_nn(points):
        d = np.hypot(*(points[:, None] - points[None]).transpose(2, 0, 1))
        return np.sort(d, axis=1)[:, 1].mean()

    even = np.random.default_rng(0).uniform(0, 200, xy.shape)
    assert mean_nn(xy) < 0.8 * mean_nn(even)
    assert sum(t[4] == "shrub" for t in trees) / len(trees) > 0.2


def test_scatter_keeps_trees_off_buildings_and_walls() -> None:
    wall = [[20.0, 20.0, 80.0, 20.0]]
    _, mask, trees = _scene(walls=wall)
    for x, y, radius, *_ in trees:
        assert mask[min(399, int((200 - y) / 0.5)), min(399, int(x / 0.5))] == 0
        near = abs(y - 20.0) < radius * 0.6 + 0.25 and 20.0 <= x <= 80.0
        assert not near or radius < 0.3
