import math
from pathlib import Path

import numpy as np
import pytest

from fraktal import biomes
from fraktal.ablation import TROPICAL_RANDOMISED, variant_config, variant_ranges
from fraktal.procedural import generate_world, read_buildings, read_roads
from fraktal.randomise import TROPICAL, sample_config
from fraktal.render import (
    CLOUD_GSD_M,
    CLOUD_HEIGHT_M,
    RenderConfig,
    apply_clouds,
    cloud_layers,
    render_region,
)


@pytest.mark.parametrize("cover", [0.0, 0.2, 0.5, 0.8, 1.0])
def test_cloud_cover_sets_the_share_of_covered_pixels(cover: float) -> None:
    clouds, _ = cloud_layers(200, CLOUD_GSD_M, 3, cover, 135.0, 60.0)
    assert abs(float((clouds > 0.5).mean()) - cover) < 0.03


def test_cloud_shadows_fall_away_from_the_sun() -> None:
    az, el = 90.0, 45.0  # sun due east, so shadows land due west of their clouds
    clouds, shadow = cloud_layers(240, CLOUD_GSD_M, 5, 0.4, az, el)
    shift = round(CLOUD_HEIGHT_M / math.tan(math.radians(el)) / CLOUD_GSD_M)
    assert shift > 5
    inner = (slice(40, 200), slice(60, 200))
    best = max(
        range(-2 * shift, 2 * shift + 1),
        key=lambda d: float(
            np.corrcoef(np.roll(clouds, d, axis=1)[inner].ravel(), shadow[inner].ravel())[0, 1]
        ),
    )
    assert abs(best + shift) <= 1  # the shadow is the cloud moved west by the expected distance
    south = cloud_layers(240, CLOUD_GSD_M, 5, 0.4, 180.0, el)[1]  # sun south: shadow moves north
    assert not np.allclose(south, shadow)


def test_clouds_are_deterministic_and_clear_sky_changes_nothing() -> None:
    a = cloud_layers(100, CLOUD_GSD_M, 9, 0.5, 135.0, 60.0)
    b = cloud_layers(100, CLOUD_GSD_M, 9, 0.5, 135.0, 60.0)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    rgb = np.full((3, 50, 50), 100, dtype="uint8")
    clear = np.zeros((50, 50), dtype="float32")
    assert (apply_clouds(rgb, clear, clear) == rgb).all()


def test_cloud_covers_ground_and_shadow_darkens_it() -> None:
    rgb = np.full((3, 4, 4), 100, dtype="uint8")
    cloud = np.zeros((4, 4), dtype="float32")
    cloud[0, 0] = 1.0
    shadow = np.zeros((4, 4), dtype="float32")
    shadow[3, 3] = 1.0
    out = apply_clouds(rgb, cloud, shadow)
    assert out[:, 0, 0].min() > 200 and out[:, 3, 3].max() < 100 and out[0, 1, 1] == 100


def test_clouds_leave_every_label_untouched(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, (104.0, 1.5), 1, biomes.resolve("khartoum", {"size_m": 600.0}))
    base = RenderConfig(gsd_m=1.0, tile_px=200, size_m=200.0, seed=1)
    clear = render_region(
        world, read_buildings(world), (104.0, 1.5), tmp_path / "a", base, roads=read_roads(world)
    )
    cloudy = render_region(
        world,
        read_buildings(world),
        (104.0, 1.5),
        tmp_path / "b",
        RenderConfig(gsd_m=1.0, tile_px=200, size_m=200.0, seed=1, cloud_cover=0.6),
        roads=read_roads(world),
    )
    assert [t["mask_sha256"] for t in clear["tiles"]] == [t["mask_sha256"] for t in cloudy["tiles"]]
    for t in clear["tiles"]:
        name = t["tile"]
        for suffix in ("_buildings", "_landcover"):
            assert (tmp_path / "a/tiles" / f"{name}{suffix}.tif").read_bytes() == (
                tmp_path / "b/tiles" / f"{name}{suffix}.tif"
            ).read_bytes()
    assert (tmp_path / "a/tiles" / f"{clear['tiles'][0]['tile']}.tif").read_bytes() != (
        tmp_path / "b/tiles" / f"{clear['tiles'][0]['tile']}.tif"
    ).read_bytes()
    assert cloudy["cloud_cover"] == 0.6


def test_tropical_ranges_draw_haze_cloud_and_a_greener_tone() -> None:
    config, randomised = variant_config(TROPICAL_RANDOMISED, RenderConfig())
    assert randomised and variant_ranges(TROPICAL_RANDOMISED) == TROPICAL
    draws = [sample_config(config, TROPICAL, 4, scene) for scene in range(40)]
    clouds = [d.cloud_cover for d in draws]
    assert min(clouds) >= 0.0 and max(clouds) <= 0.7 and max(clouds) - min(clouds) > 0.4
    assert all(0.12 <= d.haze <= 0.35 for d in draws)
    mean = np.mean([d.tone for d in draws], axis=0)
    assert mean[1] > mean[0] > mean[2]  # greener than red and blue, unlike the Khartoum tone
