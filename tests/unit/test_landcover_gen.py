import numpy as np
import pytest

from fraktal.kernels.hydrology import flow_accumulation
from fraktal.landcover_gen import BARE, GRASS, SHRUB, TREE, WATER, generate_landcover, slope_degrees
from fraktal.terrain_gen import fractal_terrain


def valley(size: int = 120) -> np.ndarray:
    """A broad valley draining south, with a pit-free floor."""
    cols = (np.abs(np.arange(size) - size / 2) / (size / 2)) ** 1.5 * 30
    rows = np.linspace(20, 0, size)[:, None]
    out: np.ndarray = (rows + cols[None, :]).astype("float64")
    return out


def test_water_sits_in_low_ground_that_collects_a_large_catchment() -> None:
    dem = valley()
    land = generate_landcover(
        dem, 5.0, 1, water_min_catchment_m2=20_000.0, water_elevation_quantile=0.1
    )
    acc, _ = flow_accumulation(dem)
    water = land == WATER
    assert water.any()
    assert (acc[water] * 25 >= 20_000).all()  # 25 m2 per cell
    assert dem[water].max() <= np.quantile(dem, 0.1) + 1e-9
    assert not water[:10].any()  # no standing water on the high ground


def test_real_classes_are_kept_where_given() -> None:
    dem = valley()
    real = np.zeros(dem.shape, dtype="uint8")
    real[40:60, 40:60] = 50
    land = generate_landcover(dem, 5.0, 1, real=real)
    assert (land[40:60, 40:60] == 50).all()
    assert (land[real == 0] != 50).all()


def test_same_seed_same_classes_different_seed_differs() -> None:
    dem = fractal_terrain((150, 150), 5.0, 2)
    a = generate_landcover(dem, 5.0, 7)
    assert (a == generate_landcover(dem, 5.0, 7)).all()
    assert (a != generate_landcover(dem, 5.0, 8)).any()


def test_aridity_trades_vegetation_for_bare_ground() -> None:
    dem = fractal_terrain((200, 200), 5.0, 3)
    lush = generate_landcover(dem, 5.0, 1, aridity=0.0)
    dry = generate_landcover(dem, 5.0, 1, aridity=1.0)
    assert (dry == BARE).mean() > (lush == BARE).mean() + 0.2
    assert ((lush == TREE) | (lush == GRASS) | (lush == SHRUB)).mean() > (
        (dry == TREE) | (dry == GRASS) | (dry == SHRUB)
    ).mean()


def test_steep_ground_is_bare_and_slope_is_measured_in_degrees() -> None:
    ramp = np.tile(np.arange(40, dtype="float64") * 10.0, (40, 1))  # 10 m rise per 10 m: 45 degrees
    assert slope_degrees(ramp, 10.0)[20, 20] == pytest.approx(45.0, abs=0.5)
    assert (generate_landcover(ramp, 10.0, 1)[5:35, 10:35] == BARE).all()


def test_only_worldcover_codes_are_produced_and_bad_input_is_rejected() -> None:
    land = generate_landcover(fractal_terrain((100, 100), 5.0, 4), 5.0, 1)
    assert set(np.unique(land)) <= {10, 20, 30, 40, 60, 80} and land.dtype == np.uint8
    with pytest.raises(ValueError, match="aridity"):
        generate_landcover(np.zeros((5, 5)), 5.0, 1, aridity=2.0)
    with pytest.raises(ValueError, match="shape"):
        generate_landcover(np.zeros((5, 5)), 5.0, 1, real=np.zeros((4, 4), dtype="uint8"))


def test_defaults_give_a_little_standing_water_on_rolling_terrain() -> None:
    land = generate_landcover(fractal_terrain((600, 600), 5.0, 0, relief_m=60.0), 5.0, 0)
    assert 0.001 < (land == WATER).mean() < 0.1
