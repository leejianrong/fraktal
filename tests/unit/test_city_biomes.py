"""The pilot city biomes (FRK-146): Khartoum, Manila-Quezon and Nairobi centre at street level."""

from pathlib import Path

import pytest

from fraktal import biomes, compounds, ground_soil, heights, trees3d
from fraktal.procedural import generate_world, read_buildings, read_roads

CITIES = ("khartoum_city", "manila_quezon", "nairobi_centre")
CENTER = (32.55, 15.64)


@pytest.mark.parametrize("city", CITIES)
def test_each_city_is_a_registered_biome_with_its_own_table_entries(city: str) -> None:
    assert city in biomes.names()
    assert biomes.resolve(city).biome == city
    assert city in trees3d.MIX and city in trees3d.PER_HA and city in compounds.WALL_SHARE
    assert city in ground_soil.BIOME_SOIL
    assert heights.profile_for(city) in heights.PROFILES


def test_every_height_profile_sums_to_one() -> None:
    for name, profile in heights.PROFILES.items():
        assert sum(share for _, share in profile) == pytest.approx(1.0), name


def test_the_cities_differ_the_way_the_photos_do() -> None:
    manila, nairobi = biomes.resolve("manila_quezon"), biomes.resolve("nairobi_centre")
    khartoum = biomes.resolve("khartoum_city")
    assert (khartoum.roads.drive_side, manila.roads.drive_side) == ("right", "right")
    assert nairobi.roads.drive_side == "left"  # Kenya drives on the left
    assert nairobi.buildings.height_core_m[1] > 2 * manila.buildings.height_core_m[1]
    assert (
        trees3d.PER_HA["manila_quezon"] > 4 * trees3d.PER_HA["khartoum_city"]
    )  # canopy against bare
    assert manila.roads.arterial_width_m > 2 * biomes.resolve("dense_urban").roads.local_width_m


def test_a_city_world_is_byte_identical_per_seed_and_the_old_biomes_are_unchanged(
    tmp_path: Path,
) -> None:
    config = biomes.resolve("nairobi_centre", {"size_m": 1000.0})
    one = generate_world(tmp_path / "a", CENTER, 4, config)
    two = generate_world(tmp_path / "b", CENTER, 4, config)
    assert one["world_id"] == two["world_id"] and one["biome"] == "nairobi_centre"
    assert len(read_buildings(tmp_path / "a")) > 50 and len(read_roads(tmp_path / "a")) > 5
    assert heights.profile_for("dense_urban") is None  # still keeps the world's own heights
    assert heights.profile_for("khartoum") == "lowrise"
