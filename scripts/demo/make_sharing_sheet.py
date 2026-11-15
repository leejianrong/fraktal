"""One landscape page for a short talk: drone shots, the asset library, a GeoTIFF dataset sample.

    uv run python scripts/demo/make_sharing_sheet.py --nadir DIR --flight DIR --out DIR

``--nadir`` is a ``fraktal render3d`` run with no trajectory (``tile.tif`` and ``labels.tif``),
``--flight`` one with ``--trajectory orbit|flight|path`` (``frames/``). Writes into ``--out``:
``sheet.pdf``, ``sheet.png``, and ``sample/`` with the RGB tile, a one-band class GeoTIFF with a
colour table, the instance GeoTIFF and a ``sample.json`` that says how it was made. Everything on
the page comes from the run or from ``assets/``; no real photograph is used.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

from fraktal.labels3d import CLASSES, decode

ROOT = Path(__file__).resolve().parents[2]
PALETTE = {
    "none": (0, 0, 0),
    "ground": (120, 120, 120),
    "building": (200, 60, 50),
    "tree": (50, 160, 60),
    "vehicle": (240, 200, 40),
    "pole": (240, 240, 240),
    "person": (230, 60, 200),
    "drone": (60, 220, 230),
    "wall": (240, 140, 40),
    "yard_object": (150, 100, 60),
}
TEXTURES = (
    "concrete031-color", "plaster001-color", "roofingtiles013a-color", "corrugatedsteel007a-color",
    "gen-clay-tile", "gen-concrete-cracked", "gen-paint-peeling", "gen-plaster-cracked",
    "gen-soil-red", "concrete034-color",
)  # fmt: skip


def write_sample(nadir: Path, out: Path) -> dict[str, object]:
    """The dataset sample: RGB, a class band with a colour table, instances, and what they are."""
    sample = out / "sample"
    sample.mkdir(parents=True, exist_ok=True)
    with rasterio.open(nadir / "tile.tif") as src:
        rgb = src.read()
        profile = src.profile
    with rasterio.open(nadir / "labels.tif") as src:
        semantic, instance = decode(src.read())
    base = {**profile, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    with rasterio.open(sample / "rgb.tif", "w", **base) as dst:
        dst.write(rgb)
    with rasterio.open(sample / "class.tif", "w", **{**base, "count": 1, "dtype": "uint8"}) as dst:
        dst.write(semantic[None])
        dst.write_colormap(1, {v: (*PALETTE[k], 255) for k, v in CLASSES.items()})
        dst.update_tags(1, classes=json.dumps(CLASSES))
    with rasterio.open(
        sample / "instance.tif", "w", **{**base, "count": 1, "dtype": "uint16"}
    ) as dst:
        dst.write(instance.astype("uint16")[None])
    share = {k: round(float((semantic == v).mean()), 4) for k, v in CLASSES.items() if v}
    info = {
        "crs": str(profile["crs"]),
        "pixel_m": abs(profile["transform"].a),
        "size_px": [profile["width"], profile["height"]],
        "bands": {
            "rgb.tif": "3 x uint8",
            "class.tif": "1 x uint8 + colour table",
            "instance.tif": "1 x uint16",
        },
        "class_share": share,
        "render": json.loads((nadir / "render.json").read_text()),
    }  # fmt: skip
    (sample / "sample.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def class_picture(semantic: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), dtype="uint8")
    for name, v in CLASSES.items():
        lut[v] = PALETTE[name]
    return lut[semantic]


def asset_stats() -> tuple[Counter[str], Counter[str]]:
    rows = json.loads((ROOT / "assets" / "manifest.json").read_text())["assets"]
    return Counter(r["kind"] for r in rows), Counter(r["licence"] for r in rows)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, TEXT, DIM = (17, 20, 24), (255, 255, 255), (154, 165, 177)
W, H = 3200, 1800  # a 16:9 page


def fit(path_or_array: object, w: int, h: int) -> Image.Image:
    """The picture cropped to the w x h box, centred."""
    img = (
        Image.fromarray(path_or_array)  # type: ignore[arg-type]
        if isinstance(path_or_array, np.ndarray)
        else Image.open(path_or_array).convert("RGB")  # type: ignore[arg-type]
    )
    scale = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def sheet(
    nadir: Path, flight: Path, out: Path, info: dict[str, object], viewer: Path | None = None
) -> None:
    page = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(page)

    def text(
        xy: tuple[int, int],
        s: str,
        size: int,
        colour: tuple[int, int, int] = TEXT,
        font: str = FONT,
    ) -> None:
        draw.text(xy, s, fill=colour, font=ImageFont.truetype(font, size))

    text((60, 40), "Fraktal", 84, font=BOLD)
    text((420, 68), "procedural worlds, rendered from any camera, written as GeoTIFF", 46, DIM)
    margin, gap = 60, 50
    colw = (W - 2 * margin - 2 * gap) // 3
    top = 210
    xs = [margin + i * (colw + gap) for i in range(3)]
    heads = [
        (
            "1  Drone shots",
            "one world, any camera; traffic, people, drones move",
        ),
        ("2  Asset library", "every asset has a licence row, checked in CI"),
        ("3  GeoTIFF datasets", "imagery and exact labels, georeferenced"),
    ]
    for x, (title, sub) in zip(xs, heads, strict=True):
        text((x, top), title, 58, font=BOLD)
        text((x, top + 74), sub, 30, DIM)
    y0 = top + 140

    # 1. drone shots
    frames = sorted((flight / "frames").glob("*.png"))
    pick = [frames[0], frames[len(frames) // 2], frames[-1]] if frames else []
    ph = (H - y0 - 60 - 2 * 24) // 3
    for k, f in enumerate(pick):
        page.paste(fit(f, colw, ph), (xs[0], y0 + k * (ph + 24)))

    # 2. asset library
    kinds, licences = asset_stats()
    x = xs[1]
    total = sum(kinds.values()) - kinds["licence"]
    text((x, y0), f"{total} assets: {kinds['material']} materials,", 40, font=BOLD)
    text(
        (x, y0 + 50), f"{kinds['texture']} textures, {kinds['mesh']} CC0 car meshes", 40, font=BOLD
    )
    text(
        (x, y0 + 112),
        "licences: " + ", ".join(f"{n} {k}" for k, n in sorted(licences.items())),
        30,
        DIM,
    )
    cell = (colw - 4 * 14) // 5
    for k, name in enumerate(TEXTURES):
        page.paste(
            fit(ROOT / "assets" / "textures" / f"{name}.jpg", cell, cell),
            (x + (k % 5) * (cell + 14), y0 + 170 + (k // 5) * (cell + 14)),
        )
    y = y0 + 170 + 2 * (cell + 14) + 30
    text((x, y), "procedural props: a seed changes size, colour and wear", 30, DIM)
    props = Image.open(ROOT / "docs" / "img" / "realism" / "street-props.jpg").convert("RGB")
    props = props.crop((0, 40, props.width, props.height - 60))  # the white margins
    scale = colw / props.width
    props = props.resize((colw, round(props.height * scale)), Image.LANCZOS)
    page.paste(props, (x, y + 46))
    y += 46 + props.height

    if viewer is not None:
        y += 50
        room = H - y - 60 - 46
        text((x, y), "Godot viewer: drive, walk or fly the same world", 30, DIM)
        page.paste(fit(viewer, colw, room), (x, y + 46))

    # 3. dataset
    x = xs[2]
    with rasterio.open(nadir / "tile.tif") as src:
        rgb = np.moveaxis(src.read(), 0, -1)
    with rasterio.open(nadir / "labels.tif") as src:
        semantic, _ = decode(src.read())
    half = (colw - 20) // 2
    side = half
    for k, (img, cap) in enumerate([(rgb, "rgb.tif"), (class_picture(semantic), "class.tif")]):
        page.paste(fit(img, side, side), (x + k * (half + 20), y0 + 50))
        text((x + k * (half + 20), y0), cap, 38, font=BOLD)
    share = info["class_share"]
    assert isinstance(share, dict)
    size = info["size_px"]
    assert isinstance(size, list)
    lines = [
        f"{info['crs']}  {info['pixel_m']} m pixels  {size[0]} x {size[1]} px",
        "rgb.tif       3 x uint8",
        "class.tif     1 x uint8 + colour table",
        "instance.tif  1 x uint16",
        "labels come from the same camera: exact,",
        "no model in the loop",
        "",
        "  ".join(f"{k} {v * 100:.1f}%" for k, v in list(share.items())[:3]),
        "  ".join(f"{k} {v * 100:.1f}%" for k, v in list(share.items())[3:5]),
        "",
        "next: SAR (backscatter, layover, shadow,",
        "speckle) from the same world",
    ]
    ytxt = y0 + 50 + side + 50
    for i, line in enumerate(lines):
        text((x, ytxt + i * 44), line, 30, (201, 209, 217), MONO)

    out.mkdir(parents=True, exist_ok=True)
    page.save(out / "sheet.png")
    page.save(out / "sheet.pdf", resolution=200.0)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--nadir", type=Path, required=True)
    p.add_argument("--flight", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--viewer", type=Path, help="a screenshot of the Godot viewer")
    a = p.parse_args()
    info = write_sample(a.nadir, a.out)
    sheet(a.nadir, a.flight, a.out, info, a.viewer)
    print(f"wrote {a.out}/sheet.pdf, sheet.png and sample/")


if __name__ == "__main__":
    main()
