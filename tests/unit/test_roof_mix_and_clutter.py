import numpy as np
import pytest

from fraktal import building_grammar as bg

RING = [[0, 0], [14, 0], [14, 12], [0, 12], [0, 0]]


def test_default_mix_is_unchanged_and_new_colours_have_weight_in_biomes() -> None:
    assert bg.TIN_WEIGHTS[4:] == (0.0, 0.0, 0.0, 0.0)  # a caller with no biome sees the old mix
    for mix in (*bg.TIN_MIXES.values(), bg.TIN_MIX_TROPICAL):
        assert len(mix) == len(bg.TIN_ROOFS) and sum(mix) == pytest.approx(1.0)
    assert bg.tin_mix("informal_settlement") == bg.TIN_MIX_TROPICAL
    assert bg.tin_mix("khartoum") != bg.TIN_MIX_TROPICAL


def test_tropical_mix_gives_painted_colours_and_default_does_not() -> None:
    def roofs(mix: tuple[float, ...] | None) -> set[str]:
        out: set[str] = set()
        for i in range(120):
            mesh = bg.build(f"b{i}", RING, 3.0, "tin", 1, roof_mix=mix)
            out |= {bg.pack([mesh])["materials"][m] for m in set(bg.pack([mesh])["face_material"])}
        return out

    painted = {"roof-tin-blue", "roof-tin-green", "roof-tin-white", "roof-tin-pink"}
    assert painted <= roofs(bg.TIN_MIX_TROPICAL)
    assert not (painted & roofs(None))


def test_clutter_appears_on_a_share_of_flat_roofs_and_is_deterministic() -> None:
    big = [[0, 0], [20, 0], [20, 16], [0, 16], [0, 0]]
    counts = []
    for i in range(80):
        packed = bg.pack([bg.build(f"f{i}", big, 6.0, None, 2)])
        used = {packed["materials"][m] for m in set(packed["face_material"])}
        counts.append(bool(used & set(bg.ROOF_CLUTTER)))
    share = float(np.mean(counts))
    assert 0.25 < share < 0.85
    a = bg.pack([bg.build("f3", big, 6.0, None, 2)])
    b = bg.pack([bg.build("f3", big, 6.0, None, 2)])
    assert np.array_equal(a["vertices"], b["vertices"])
