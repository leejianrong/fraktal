from pathlib import Path

import numpy as np
import pytest
import rasterio
from test_render import CENTER, SMALL, one_building, terrain  # noqa: F401

from fraktal.render import RenderConfig, render_region
from fraktal.sensor import add_noise, apply_psf


def flat(value: int = 128, size: int = 200) -> np.ndarray:
    out: np.ndarray = np.full((3, size, size), value, dtype="uint8")
    return out


def test_noise_variance_matches_its_parameter() -> None:
    noisy = add_noise(flat(), 6.0, 1, "t").astype(float)
    assert noisy.std() == pytest.approx(6.0, rel=0.03)
    assert noisy.mean() == pytest.approx(128, abs=0.2)


def test_noise_is_deterministic_per_seed_and_tile_name() -> None:
    a = add_noise(flat(), 5.0, 1, "r00_c00")
    assert (a == add_noise(flat(), 5.0, 1, "r00_c00")).all()
    assert not (a == add_noise(flat(), 5.0, 1, "r00_c01")).all()
    assert not (a == add_noise(flat(), 5.0, 2, "r00_c00")).all()


def test_zero_noise_and_zero_psf_change_nothing() -> None:
    img = flat()
    assert add_noise(img, 0.0, 1, "t") is img
    assert apply_psf(img, 0.0) is img


def test_psf_softens_an_edge_but_keeps_flat_areas() -> None:
    img = flat(50)
    img[:, :, 100:] = 200
    out = apply_psf(img, 1.5)
    assert (out[:, :, :80] == 50).all() and (out[:, :, 120:] == 200).all()
    assert 50 < out[0, 0, 100] < 200
    assert np.abs(np.diff(out[0, 0].astype(int))).max() < 150


@pytest.mark.parametrize(("gsd", "tile_px", "size"), [(0.5, 100, 100.0), (3.0, 30, 180.0)])
def test_sensor_output_stays_aligned_with_masks_at_each_gsd(
    tmp_path: Path,
    terrain: Path,  # noqa: F811
    gsd: float,
    tile_px: int,
    size: float,  # noqa: F811
) -> None:
    cfg = RenderConfig(
        gsd_m=gsd, tile_px=tile_px, size_m=size, seed=3, variation=0.15,
        psf_sigma_px=1.0, noise_sigma=5.0,
    )  # fmt: skip
    layer = one_building(CENTER, 40, 6.0)
    manifest = render_region(terrain, layer, CENTER, tmp_path / "o", cfg)
    assert manifest["dropped"] == [] and manifest["tiles"]
    assert manifest["gsd_m"] == gsd
    name = manifest["tiles"][0]["tile"]
    with (
        rasterio.open(tmp_path / "o" / "tiles" / f"{name}.tif") as img,
        rasterio.open(tmp_path / "o" / "tiles" / f"{name}_buildings.tif") as msk,
    ):
        assert img.transform == msk.transform
        assert img.crs == msk.crs
        assert (img.height, img.width) == (msk.height, msk.width)


def test_sensor_leaves_labels_untouched(tmp_path: Path, terrain: Path) -> None:  # noqa: F811
    layer = one_building(CENTER, 10, 5.0)
    clean = render_region(terrain, layer, CENTER, tmp_path / "c", SMALL)
    cfg = RenderConfig(**{**SMALL.__dict__, "psf_sigma_px": 1.0, "noise_sigma": 8.0})
    noisy = render_region(terrain, layer, CENTER, tmp_path / "n", cfg)
    assert [t["mask_sha256"] for t in clean["tiles"]] == [t["mask_sha256"] for t in noisy["tiles"]]


def test_blank_wedge_is_deterministic_and_respects_its_probability_and_size() -> None:
    from fraktal.sensor import blank_wedge

    assert not blank_wedge((64, 64), 0.0, 0.3, 1, "t").any()
    a = blank_wedge((64, 64), 1.0, 0.3, 1, "t")
    assert (a == blank_wedge((64, 64), 1.0, 0.3, 1, "t")).all()
    assert 0.04 <= a.mean() <= 0.31
    hits = sum(blank_wedge((16, 16), 0.5, 0.3, 1, f"t{i}").any() for i in range(200))
    assert 70 <= hits <= 130
