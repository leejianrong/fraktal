"""Looking at real reference chips (ADR-0012; FRK-100, FRK-101). Evaluation and looking only.

The benchmark chips for SpaceNet 2 Khartoum are stretched to uint8 with one range for all three
bands (``chips.stretch_range``). The raw blue band is far lower than green, so that keeps a strong
green cast, and the Khartoum tone and palette were fitted to it. They stay as they are until the
next training milestone (ADR-0012 addendum). To *look* at a SpaceNet tile, ``spacenet_view``
stretches each band by its own range instead, which gives a natural colour image.
"""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from fraktal.chips import tile_path

SAMPLE_TILES = 64
LOW_PCT, HIGH_PCT = 2.0, 98.0


def band_ranges(root: Path, tile_ids: Sequence[str]) -> np.ndarray[Any, Any]:
    """A (3, 2) array: the 2nd and 98th percentile of each band over a sample of tiles.

    No-data (zero in any band) is ignored. ``root`` is ``<data>/spacenet2/khartoum``.
    """
    step = max(1, len(tile_ids) // SAMPLE_TILES)
    parts = []
    for tile_id in sorted(tile_ids)[::step]:
        with rasterio.open(tile_path(root, tile_id)) as src:
            data = src.read().astype("float32")
        valid = (data > 0).all(axis=0)
        parts.append(data[:, valid][:, ::37])
    flat = np.concatenate(parts, axis=1)
    return np.stack([np.percentile(band, [LOW_PCT, HIGH_PCT]) for band in flat]).astype("float32")


def spacenet_view(root: Path, tile_id: str, ranges: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The tile as a (3, H, W) uint8 image, each band stretched by its own range."""
    with rasterio.open(tile_path(root, tile_id)) as src:
        data = src.read().astype("float32")
    low = ranges[:, 0][:, None, None]
    high = ranges[:, 1][:, None, None]
    scaled = (data - low) / np.maximum(high - low, 1.0)
    out: np.ndarray[Any, Any] = np.clip(np.rint(scaled * 255), 0, 255).astype(np.uint8)
    return out


# Which real chips to show beside each biome's renders. The match is a judgement (what the place
# looks like), not a label. Satellite views use Maxar events and SpaceNet; drone views Open Cities.
SATELLITE_EVENTS = {
    "dense_urban": ["Typhoon-Kalmaegi", "Earthquake-Myanmar", "Kenya-Flooding"],
    "informal_settlement": ["Kenya-Flooding", "Indonesia-Earthquake22"],
    "khartoum": ["Sudan-flooding"],
    "mangrove_coast": ["BayofBengal", "Typhoon-Kalmaegi"],
    "oil_palm_plantation": ["Typhoon-Kalmaegi", "Indonesia-Earthquake22"],
    "paddy_terraces": ["Indonesia-Earthquake22", "BayofBengal"],
}
SPACENET_GSD_M, OPEN_CITIES_GSD_M = 0.3, 0.5
BANNER = (
    "Private. Real imagery for looking and evaluation only: never train on, redistribute, "
    "commit or ship it. Local file, not published."
)


@dataclass(frozen=True)
class Chip:
    label: str
    image: np.ndarray[Any, Any]  # (H, W, 3) uint8
    gsd_m: float


def usable(image: np.ndarray[Any, Any]) -> bool:
    """False for a chip that is mostly no-data, cloud or featureless (open sea, bare haze)."""
    grey = image.astype("float32").mean(axis=2)
    bright_grey = (image.min(axis=2) > 165) & (np.ptp(image, axis=2) < 45)  # cloud, thin or thick
    if (image.max(axis=2) == 0).mean() > 0.02 or bright_grey.mean() > 0.04:
        return False
    side = grey.shape[0] // 8 * 8, grey.shape[1] // 8 * 8
    blocks = grey[: side[0], : side[1]].reshape(8, side[0] // 8, 8, side[1] // 8).mean(axis=(1, 3))
    return bool(blocks.std() >= 6.0)


def _spread(items: Sequence[Any], count: int) -> list[Any]:
    """``count`` items spread evenly over a sorted list, so the pick is deterministic."""
    if len(items) <= count:
        return list(items)
    return [items[round(k * (len(items) - 1) / max(count - 1, 1))] for k in range(count)]


def _hwc(chip: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    out: np.ndarray[Any, Any] = np.moveaxis(chip, 0, -1)
    return out


def maxar_chips(data_dir: Path, biome: str, count: int) -> list[Chip]:
    """Usable Maxar chips for a biome, taking its events in priority order, round robin."""
    root = data_dir / "eval" / "maxar-reference"
    per_event: list[list[Chip]] = []
    for event in SATELLITE_EVENTS.get(biome, []):
        chips = []
        for path in sorted(root.glob(f"{event}*.npy")):
            image = _hwc(np.load(path))
            if usable(image):
                meta = json.loads(path.with_suffix(".json").read_text())
                label = f"{meta['event']} {meta['scene'].split('/')[2]} {meta['kind']}"
                chips.append(Chip(label, image, float(meta["pixel_size"][0])))
        per_event.append(chips)
    out: list[Chip] = []
    for k in range(count):
        for chips in per_event:
            if k < len(chips) and len(out) < count:
                out.append(chips[k])
    return out


def spacenet_chips(data_dir: Path, count: int) -> list[Chip]:
    root = data_dir / "spacenet2" / "khartoum"
    chips_dir = data_dir / "spacenet2" / "chips"
    ids = sorted(p.name.removesuffix("_img.npy") for p in chips_dir.glob("img*_img.npy"))
    if not ids:
        return []
    ranges = band_ranges(root, ids)
    out = []
    for tile_id in ids[::7]:  # a sample of tiles, the first usable ones
        image = _hwc(spacenet_view(root, tile_id, ranges))
        if usable(image):
            out.append(Chip(f"SpaceNet {tile_id}", image, SPACENET_GSD_M))
    return _spread(out, count)


def open_cities_chips(data_dir: Path, count: int, row: int = 0, rows: int = 1) -> list[Chip]:
    names = sorted((data_dir / "eval" / "open-cities-10").glob("*_img.npy"))
    chips = [
        Chip(f"Open Cities {p.name.removesuffix('_img.npy')}", _hwc(np.load(p)), OPEN_CITIES_GSD_M)
        for p in names
    ]
    spread = _spread([c for c in chips if usable(c.image)], count * rows)
    return spread[row * count : (row + 1) * count]


def reference_chips(
    data_dir: Path, view: str, biome: str, count: int, row: int = 0, rows: int = 1
) -> list[Chip]:
    """Real chips for one row of a review sheet. Drone chips are not tied to a biome, so each
    row of ``rows`` shows different ones."""
    if view == "drone":
        return open_cities_chips(data_dir, count, row, rows)
    if biome == "khartoum":  # one SpaceNet tile, then Maxar Sudan
        return (spacenet_chips(data_dir, 1) + maxar_chips(data_dir, biome, count))[:count]
    return maxar_chips(data_dir, biome, count)


def synthetic_chips(root: Path, biome: str, count: int) -> list[Chip]:
    """Renders of one biome from a ``render3d-batch`` layout: ``<biome>/<tile>/sensor.png``."""
    from PIL import Image

    tiles = sorted((root / biome).glob("tile_*"))
    out = []
    for tile in _spread(tiles, count):
        image = next((tile / n for n in ("sensor.png", "image.png") if (tile / n).is_file()), None)
        if image is None:
            continue
        info = tile / "render.json"
        gsd = json.loads(info.read_text()).get("gsd_m", 0.5) if info.is_file() else 0.5
        pixels = np.asarray(Image.open(image).convert("RGB"))
        out.append(Chip(f"synthetic {tile.name}", pixels, float(gsd)))
    return out


def crop_to_ground(chip: Chip, ground_m: float, cell: int) -> Any:
    """The middle ``ground_m`` metres of the chip, resized to ``cell`` pixels, as a PIL image."""
    from PIL import Image

    h, w, _ = chip.image.shape
    side = min(h, w, max(1, round(ground_m / chip.gsd_m)))
    top, left = (h - side) // 2, (w - side) // 2
    window = chip.image[top : top + side, left : left + side]
    return Image.fromarray(window).resize((cell, cell), Image.Resampling.LANCZOS)


def make_sheet(
    rows: Sequence[tuple[str, Sequence[Chip], Sequence[Chip]]],
    ground_m: float = 250.0,
    cell: int = 300,
) -> Any:
    """One row per biome: renders on the left, real chips on the right, each ``ground_m`` wide."""
    from PIL import Image, ImageDraw

    columns = max((len(a) + len(b) for _, a, b in rows), default=1)
    margin = 18
    sheet = Image.new("RGB", (columns * cell, len(rows) * (cell + margin)), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    for r, (biome, synthetic, real) in enumerate(rows):
        y = r * (cell + margin)
        for c, chip in enumerate([*synthetic, *real]):
            sheet.paste(crop_to_ground(chip, ground_m, cell), (c * cell, y + margin))
            colour = (240, 166, 58) if c >= len(synthetic) else (232, 230, 225)
            text = f"{biome} | {chip.label}" if c == 0 else chip.label
            draw.text((c * cell + 4, y + 3), text[:52], fill=colour)
    return sheet


def write_board(data_dir: Path, out: Path) -> Path:
    """A local HTML page with every reference chip (Maxar, Open Cities, SpaceNet per-band)."""
    from PIL import Image

    def figure(chip: Chip) -> str:
        image = Image.fromarray(chip.image).resize((512, 512), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=84)
        data = base64.b64encode(buf.getvalue()).decode()
        src = f"data:image/jpeg;base64,{data}"
        return f"<figure><img src={src!r}><figcaption>{chip.label}</figcaption></figure>"

    sections = {
        "Maxar Open Data: satellite, 0.31 m (CC BY-NC 4.0)": [
            c for biome in SATELLITE_EVENTS for c in maxar_chips(data_dir, biome, 99)
        ],
        "Open Cities AI: drone (ODbL-1.0)": open_cities_chips(data_dir, 24),
        "SpaceNet 2 Khartoum: satellite, per-band stretch": spacenet_chips(data_dir, 8),
    }
    seen: set[str] = set()
    body = ""
    for title, chips in sections.items():
        figures = []
        for chip in chips:
            if chip.label not in seen:  # a Maxar chip can match two biomes
                seen.add(chip.label)
                figures.append(figure(chip))
        body += f"<h2>{title}</h2><div class=g>{''.join(figures)}</div>"
    html = (
        "<!doctype html><meta charset=utf-8><title>Fraktal reference board</title>"
        "<style>body{background:#14161a;color:#e8e6e1;font:15px system-ui;padding:16px}"
        "h2{color:#f0a63a;font-size:17px}.g{display:grid;gap:10px;"
        "grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}"
        "img{width:100%;border-radius:4px}figcaption{font-size:12px;color:#9a9890}</style>"
        f"<h1>Fraktal reference board</h1><p>{BANNER}</p>{body}"
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reference-board.html"
    path.write_text(html)
    return path


__all__ = [
    "Chip",
    "band_ranges",
    "make_sheet",
    "reference_chips",
    "spacenet_view",
    "synthetic_chips",
    "write_board",
]
