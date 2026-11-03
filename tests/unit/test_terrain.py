import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fraktal import cli
from fraktal.terrain import (
    Bounds,
    TerrainError,
    dem_tile_urls,
    ingest_terrain,
    worldcover_tile_urls,
)

AOI: Bounds = (32.51, 15.51, 32.53, 15.53)


def write_raster(path: Path, bounds: Bounds, res: float, fill: float, dtype: str) -> None:
    left, bottom, right, top = bounds
    width, height = round((right - left) / res), round((top - bottom) / res)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(left, top, res, res),
    ) as dst:
        data = np.full((height, width), fill, dtype=dtype)
        if dtype == "float32":
            data += np.linspace(0, 5, width, dtype="float32")[None, :]  # a slope, never flat
        dst.write(data, 1)


@pytest.fixture
def sources(tmp_path: Path) -> tuple[Path, Path]:
    dem, wc = tmp_path / "src_dem.tif", tmp_path / "src_wc.tif"
    write_raster(dem, (32.0, 15.0, 33.0, 16.0), 1 / 100, 380.0, "float32")
    write_raster(wc, (30.0, 15.0, 33.0, 18.0), 1 / 1000, 50, "uint8")
    return dem, wc


def test_tile_urls_cover_the_aoi() -> None:
    assert dem_tile_urls(AOI) == [
        "https://copernicus-dem-30m.s3.amazonaws.com/"
        "Copernicus_DSM_COG_10_N15_00_E032_00_DEM/Copernicus_DSM_COG_10_N15_00_E032_00_DEM.tif"
    ]
    assert worldcover_tile_urls(AOI)[0].endswith("_N15E030_Map.tif")
    assert len(dem_tile_urls((31.9, 15.5, 32.1, 15.6))) == 2  # straddles a tile edge
    assert "S02_00_W001_00" in dem_tile_urls((-0.5, -1.5, -0.4, -1.4))[0]


def test_layers_share_one_grid_and_carry_attribution(
    tmp_path: Path, sources: tuple[Path, Path]
) -> None:
    dem_src, wc_src = sources
    result = ingest_terrain(
        tmp_path / "data",
        AOI,
        dem_urls=lambda _b: [str(dem_src)],
        worldcover_urls=lambda _b: [str(wc_src)],
    )
    assert result.status == "fetched"
    with (
        rasterio.open(result.path / "dem.tif") as dem,
        rasterio.open(result.path / "landcover.tif") as lc,
    ):
        assert dem.transform == lc.transform
        assert (dem.width, dem.height) == (lc.width, lc.height)
        assert dem.crs == lc.crs
        assert dem.bounds.left <= AOI[0] and dem.bounds.right >= AOI[2]
        assert "Copernicus" in dem.tags()["attribution"]
        assert lc.tags()["licence"] == "CC-BY-4.0"
        assert dem.read(1).std() > 0  # not flat
        assert (lc.read(1) == 50).all()
        assert dem.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"


def test_second_run_makes_no_network_calls(tmp_path: Path, sources: tuple[Path, Path]) -> None:
    dem_src, wc_src = sources
    data = tmp_path / "data"
    ingest_terrain(
        data, AOI, dem_urls=lambda _b: [str(dem_src)], worldcover_urls=lambda _b: [str(wc_src)]
    )

    def boom(_b: Bounds) -> list[str]:
        raise AssertionError("network used")

    assert ingest_terrain(data, AOI, dem_urls=boom, worldcover_urls=boom).status == "cached"


def test_missing_dem_tile_is_a_hard_error_not_a_flat_dem(
    tmp_path: Path, sources: tuple[Path, Path]
) -> None:
    _dem, wc_src = sources
    with pytest.raises(TerrainError, match="missing or unreadable"):
        ingest_terrain(
            tmp_path / "data",
            AOI,
            dem_urls=lambda _b: [str(tmp_path / "nope.tif")],
            worldcover_urls=lambda _b: [str(wc_src)],
        )
    assert not (tmp_path / "data" / "terrain" / "khartoum").exists()


def test_dem_that_does_not_cover_the_aoi_is_an_error(
    tmp_path: Path, sources: tuple[Path, Path]
) -> None:
    _dem, wc_src = sources
    small = tmp_path / "small.tif"
    write_raster(small, (32.0, 15.0, 32.515, 16.0), 1 / 100, 380.0, "float32")
    with pytest.raises(TerrainError):
        ingest_terrain(
            tmp_path / "data",
            AOI,
            dem_urls=lambda _b: [str(small)],
            worldcover_urls=lambda _b: [str(wc_src)],
        )


def test_cli_terrain_missing_data_exits_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["ingest", "--what", "terrain", "--data-dir", str(tmp_path), "--json"]
    assert cli.main(args) == 4
    assert json.loads(capsys.readouterr().out)["error"]["kind"] == "ingest_failed"
