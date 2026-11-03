import dataclasses
from typing import Any

import pytest

from fraktal.randomise import COMPONENTS, Ranges, sample_config
from fraktal.render import RenderConfig

BASE = RenderConfig()


def test_same_seed_and_scene_give_identical_parameters() -> None:
    assert sample_config(BASE, Ranges(), 1, 0) == sample_config(BASE, Ranges(), 1, 0)


def test_scenes_and_seeds_differ() -> None:
    a = sample_config(BASE, Ranges(), 1, 0)
    assert a != sample_config(BASE, Ranges(), 1, 1)
    assert a != sample_config(BASE, Ranges(), 2, 0)
    assert a.seed != sample_config(BASE, Ranges(), 1, 1).seed


def test_values_lie_inside_their_ranges() -> None:
    r = Ranges()
    for scene in range(20):
        c = sample_config(BASE, r, 3, scene)
        for name in COMPONENTS:
            lo, hi = getattr(r, name)
            assert lo <= getattr(c, name) <= hi


@pytest.mark.parametrize("changed", COMPONENTS)
def test_changing_one_range_leaves_every_other_draw_alone(changed: str) -> None:
    before = sample_config(BASE, Ranges(), 5, 2)
    edited: dict[str, Any] = {changed: (0.0, 1.0)}
    after = sample_config(BASE, dataclasses.replace(Ranges(), **edited), 5, 2)
    for name in (*COMPONENTS, "tone", "seed"):
        if name != changed:
            assert getattr(before, name) == getattr(after, name), name


def test_a_fixed_value_is_a_range_with_equal_ends() -> None:
    c = sample_config(BASE, Ranges(noise_sigma=(7.0, 7.0)), 1, 4)
    assert c.noise_sigma == 7.0


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="reversed"):
        sample_config(BASE, Ranges(haze=(0.2, 0.0)), 1, 0)


def test_tone_jitters_around_the_base_tone() -> None:
    c = sample_config(BASE, Ranges(tone_jitter=(1.0, 1.0)), 1, 0)
    assert c.tone == Ranges().base_tone
