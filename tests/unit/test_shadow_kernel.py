import numpy as np
import pytest

from fraktal.kernels.shadow import shadow_mask


def box_dsm(px: float, height: float, size_px: int = 120) -> np.ndarray:
    dsm = np.zeros((size_px, size_px))
    c = size_px // 2
    half = max(1, int(round(5 / px)))  # a 10 m box
    dsm[c - half : c + half, c - half : c + half] = height
    return dsm


def test_golden_output_small_box() -> None:
    # Sun due east at 45 degrees and 1 m pixels: a 3 m tall block casts 3 px of shadow west.
    dsm = np.zeros((7, 12))
    dsm[2:5, 8:10] = 3.0
    expected = [
        "............",
        "............",
        ".....###....",
        ".....###....",
        ".....###....",
        "............",
        "............",
    ]
    got = ["".join("#" if v else "." for v in row) for row in shadow_mask(dsm, 1.0, 90.0, 45.0)]
    assert got == expected


def test_ten_metre_box_at_45_degrees_casts_a_ten_metre_shadow() -> None:
    px = 0.5
    dsm = box_dsm(px, 10.0)
    mask = shadow_mask(dsm, px, 270.0, 45.0)  # sun in the west, shadow falls east
    east = mask[:, dsm.shape[1] // 2 :]
    ground_east = dsm[:, dsm.shape[1] // 2 :] == 0
    length_px = int((east & ground_east)[dsm.shape[0] // 2].sum())
    assert abs(length_px - 10.0 / px) <= 1


def test_shadow_falls_away_from_the_sun() -> None:
    px = 1.0
    dsm = box_dsm(px, 6.0, 60)
    c = 30
    for azimuth, probe in [(180, (c - 8, c)), (0, (c + 8, c)), (90, (c, c - 8)), (270, (c, c + 8))]:
        mask = shadow_mask(dsm, px, azimuth, 45.0)  # sun south: shadow falls north, and so on
        assert mask[probe] == 1, azimuth


def test_flat_ground_has_no_shadow() -> None:
    assert shadow_mask(np.zeros((20, 20)), 1.0, 135.0, 30.0).sum() == 0


def test_steeper_sun_shortens_shadows() -> None:
    dsm = box_dsm(1.0, 10.0, 80)
    low = shadow_mask(dsm, 1.0, 270.0, 30.0).sum()
    high = shadow_mask(dsm, 1.0, 270.0, 60.0).sum()
    assert high < low


def test_deterministic_and_input_untouched() -> None:
    dsm = box_dsm(0.5, 8.0)
    before = dsm.copy()
    a = shadow_mask(dsm, 0.5, 135.0, 40.0)
    b = shadow_mask(dsm, 0.5, 135.0, 40.0)
    assert a.tobytes() == b.tobytes()
    assert (dsm == before).all()
    assert a.dtype == np.uint8


@pytest.mark.parametrize(("px", "elevation"), [(0.0, 45.0), (1.0, 0.0), (1.0, 90.0)])
def test_bad_parameters_rejected(px: float, elevation: float) -> None:
    with pytest.raises(ValueError):
        shadow_mask(np.zeros((4, 4)), px, 0.0, elevation)
