from fraktal import randomise
from fraktal.render import KHARTOUM_TONE


def test_khartoum_3d_tone_is_neutral_and_the_benchmark_tone_is_unchanged() -> None:
    arid = randomise.BIOME_RANGES["khartoum"]
    assert max(arid.base_tone) - min(arid.base_tone) < 0.1
    assert randomise.Ranges().base_tone == KHARTOUM_TONE  # the 2.5D benchmark path
    assert randomise.TROPICAL.base_tone == randomise.TROPICAL_TONE
