from pathlib import Path

import pytest
import rasterio

from fraktal.terrain import ingest_terrain

pytestmark = pytest.mark.integration


def test_real_khartoum_window_reads_from_public_cogs(tmp_path: Path) -> None:
    result = ingest_terrain(tmp_path, (32.52, 15.55, 32.53, 15.56))
    assert result.status == "fetched"
    with rasterio.open(result.path / "dem.tif") as dem:
        elevation = dem.read(1)
        assert 300 < elevation.mean() < 500  # Khartoum sits near 380 m
        assert "Copernicus" in dem.tags()["attribution"]
    with rasterio.open(result.path / "landcover.tif") as lc:
        assert lc.shape == dem.shape
        assert lc.read(1).max() <= 100
    assert ingest_terrain(tmp_path, (32.52, 15.55, 32.53, 15.56)).status == "cached"
