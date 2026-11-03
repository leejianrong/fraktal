import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from test_chips import make_dataset
from test_render import CENTER, SMALL, one_building, terrain  # noqa: F401

from fraktal import cli
from fraktal.chips import ChipTiles, build_chips, load_splits, tile_path
from fraktal.palette import Palette, PaletteError, fit, kmeans, read, write
from fraktal.render import ROOF_MATERIALS, RenderConfig, render_region, roof_table, shade

TONE = (0.5, 1.0, 2.0)


def landcover_for(data_dir: Path) -> Path:
    """Class 40 everywhere over the fixture tiles (lon 32.0-32.2, lat 14.99-15.01)."""
    path = data_dir / "terrain" / "khartoum" / "landcover.tif"
    path.parent.mkdir(parents=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=2000, height=200, count=1, dtype="uint8",
        crs="EPSG:4326", transform=from_origin(32.0, 15.01, 0.0001, 0.0001),
    ) as dst:  # fmt: skip
        dst.write(np.full((200, 2000), 40, dtype="uint8"), 1)
    return path


Built = tuple[Path, ChipTiles, dict[str, Path]]


@pytest.fixture
def built(tmp_path: Path) -> Built:
    make_dataset(tmp_path, 12)
    assert cli.main(["split", "--data-dir", str(tmp_path), "--seed", "1", "--json"]) == 0
    chips = build_chips(tmp_path, tmp_path / "spacenet2" / "khartoum-split.json")
    for path in chips.glob("*_img.npy"):  # the fixture's stretch makes ground black (no data)
        image = np.load(path)
        image[:, image.max(axis=0) == 0] = np.array([60, 90, 30], dtype=np.uint8)[:, None]
        np.save(path, image)
    train = load_splits(chips)[0]
    root = tmp_path / "spacenet2" / "khartoum"
    return tmp_path, train, {t: tile_path(root, t) for t in train.ids}


def test_kmeans_recovers_separated_clusters_deterministically() -> None:
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(20, 2, (200, 3)), rng.normal(200, 2, (100, 3))])
    centres, counts = kmeans(pts, 2, 3)
    assert sorted(counts.tolist()) == [100, 200]
    assert sorted(centres.mean(axis=1).round().tolist()) == pytest.approx([20, 200], abs=2)
    again, _ = kmeans(pts, 2, 3)
    assert (centres == again).all()
    with pytest.raises(PaletteError):
        kmeans(pts[:1], 2, 3)


def test_fit_reads_class_and_roof_colours_in_albedo_space(built: Built) -> None:
    data_dir, train, paths = built
    palette = fit(train, paths, landcover_for(data_dir), TONE, seed=0, max_tiles=50)
    image, _ = train.get(0)
    ground, roof = image[:, 0, 0].astype(float), image[:, 6, 6].astype(float)  # fixture layout
    # a few roof-edge pixels fall outside the fixture's label mask and nudge the mean up
    assert palette.classes[40] == pytest.approx(tuple(ground / np.array(TONE)), rel=0.08)
    assert len(palette.roofs) == 1 and palette.roofs[0][1] == pytest.approx(1.0)
    assert palette.roofs[0][0] == pytest.approx(tuple(roof / np.array(TONE)), rel=0.02)


def test_fit_and_write_use_train_tiles_only(built: Built) -> None:
    data_dir, train, paths = built
    palette = fit(train, paths, landcover_for(data_dir), TONE, seed=0, max_tiles=50)
    out = data_dir / "p.json"
    write(palette, out, train.ids)
    body = json.loads(out.read_text())
    meta = json.loads((data_dir / "spacenet2" / "chips" / "chips.json").read_text())
    assert set(body["train_tiles"]) == set(meta["train"])
    assert not set(meta["test"]) & set(body["train_tiles"])
    back = read(out)
    assert back.classes == palette.classes and back.tone == palette.tone
    assert [r[1] for r in back.roofs] == [r[1] for r in palette.roofs]


def test_palette_command_writes_a_file_from_train_tiles(
    built: Built, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir, _train, _paths = built
    landcover_for(data_dir)
    code = cli.main(["palette", "--data-dir", str(data_dir), "--max-tiles", "20", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and Path(out["data"]["palette"]).is_file()
    assert out["data"]["roof_colours"] >= 1


def test_palette_command_needs_land_cover(tmp_path: Path) -> None:
    make_dataset(tmp_path, 10)
    cli.main(["split", "--data-dir", str(tmp_path), "--json"])
    build_chips(tmp_path, tmp_path / "spacenet2" / "khartoum-split.json")
    assert cli.main(["palette", "--data-dir", str(tmp_path)]) == 4


def test_renderer_uses_palette_colours_and_roof_shares() -> None:
    palette = Palette(
        classes={30: (10.0, 20.0, 30.0)},
        roofs=(((200.0, 0.0, 0.0), 0.5), ((0.0, 200.0, 0.0), 0.5)),
        tone=(1, 1, 1),
    )
    land = np.full((2, 2), 30, dtype="uint8")
    none = np.zeros((2, 2), dtype="uint8")
    assert (shade(land, none, none, palette=palette)[:, 0, 0] == [10, 20, 30]).all()
    ids = [f"{int(i * 0xFFFFFFFF / 99):08x}00000000" for i in range(100)]
    table = roof_table(ids, 0.0, palette=palette)[1:]
    red = (table[:, 0] > 100).mean()
    assert 0.4 <= red <= 0.6 and set(map(tuple, table.tolist())) == {
        (200.0, 0.0, 0.0),
        (0.0, 200.0, 0.0),
    }
    assert roof_table(ids, 0.0)[1].tolist() == list(ROOF_MATERIALS[int(ids[0][:8], 16) % 4])


def test_render_with_a_palette_changes_the_image_but_not_the_labels(
    tmp_path: Path,
    terrain: Path,  # noqa: F811
) -> None:
    palette = Palette(
        classes={30: (200.0, 10.0, 10.0)}, roofs=(((10.0, 10.0, 200.0), 1.0),), tone=(1, 1, 1)
    )
    layer = one_building(CENTER, 30, 6.0)
    base = RenderConfig(**{**SMALL.__dict__, "variation": 0.0})
    plain = render_region(terrain, layer, CENTER, tmp_path / "a", base)
    with_palette = RenderConfig(**{**base.__dict__, "palette": palette})
    coloured = render_region(terrain, layer, CENTER, tmp_path / "b", with_palette)
    assert [t["mask_sha256"] for t in plain["tiles"]] == [
        t["mask_sha256"] for t in coloured["tiles"]
    ]
    with rasterio.open(tmp_path / "b" / "tiles" / "r00_c00.tif") as src:
        rgb = src.read()
    assert rgb[0].max() > 100 and coloured["palette"] is True and plain["palette"] is False
