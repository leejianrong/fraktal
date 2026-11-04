from collections import Counter

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
