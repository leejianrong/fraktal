"""Pull a few real satellite tiles per region from SpaceNet for the reference board.

EVALUATION AND LOOKING ONLY. SpaceNet is CC BY-SA 4.0 (ADR-0008), so the tiles stay under
data/eval/ (git-ignored), are never trained on, committed, redistributed or shown on a public page.

Close-ups (0.3 to 0.5 m): a few tiles each from Rio (SN1), Las Vegas, Paris, Shanghai (SN2) and
Mumbai, Moscow (SN5). Global (4 m): one mid-year scene from every SN7 area, labelled with its
latitude and longitude. Also writes two contact sheets under --sheets (inside data/).

Usage: uv run python scripts/refboard/pull_spacenet_regions.py --out data/eval/spacenet-regions \
           --sheets data/review/regions
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import rasterio
from PIL import Image, ImageDraw

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
CLOSEUPS = {  # region -> folder of loose tiles (SN2 RGB-PanSharpen is empty; PS-RGB has the files)
    "rio": "spacenet/SN1_buildings/train/3band/",
    "vegas": "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/",
    "paris": "spacenet/SN2_buildings/train/AOI_3_Paris/PS-RGB/",
    "shanghai": "spacenet/SN2_buildings/train/AOI_4_Shanghai/PS-RGB/",
    "mumbai": "spacenet/SN5_roads/train/AOI_8_Mumbai/PS-RGB/",
    "moscow": "spacenet/SN5_roads/train/AOI_7_Moscow/PS-RGB/",
}
SN7 = "spacenet/SN7_buildings/train/"
CITATION = (
    "Private evaluation only: looked at beside our renders, never trained on, committed, "
    "redistributed or shown publicly.\nSource: SpaceNet on AWS Open Data (s3://spacenet-dataset), "
    "CC BY-SA 4.0.\nCite: SpaceNet LLC. SpaceNet on AWS, https://registry.opendata.aws/spacenet/\n"
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fetch(url: str, tries: int = 4) -> bytes:
    for n in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()  # type: ignore[no-any-return]
        except OSError:
            if n == tries - 1:
                raise
            time.sleep(2 * (n + 1))
    raise AssertionError("unreachable")


def listing(prefix: str, delimiter: str = "") -> tuple[list[str], list[str]]:
    url = f"{BUCKET}?list-type=2&max-keys=1000&prefix={urllib.parse.quote(prefix)}"
    if delimiter:
        url += f"&delimiter={delimiter}"
    root = ElementTree.fromstring(fetch(url))
    keys = [e.findtext(f"{NS}Key") or "" for e in root.iter(f"{NS}Contents")]
    dirs = [e.findtext(f"{NS}Prefix") or "" for e in root.iter(f"{NS}CommonPrefixes")]
    return keys, dirs


def spread(keys: list[str], n: int) -> list[str]:
    tifs = sorted(k for k in keys if k.endswith(".tif"))
    return tifs if len(tifs) <= n else [tifs[int(i * len(tifs) / n)] for i in range(n)]


def download(key: str, out: Path) -> None:
    dest = out / Path(key).name
    if not dest.exists():
        dest.write_bytes(fetch(BUCKET + urllib.parse.quote(key)))


def to_image(path: Path, window_m: float | None = None) -> tuple[Image.Image, float]:
    """Read three bands with a per-band 2 to 98 percent stretch (raw data is not display-ready)."""
    with rasterio.open(path) as d:
        a = d.read([1, 2, 3]).astype(np.float32)
        gsd = d.res[0] * 111320 if d.crs.is_geographic else d.res[0]
    if window_m:
        n = min(int(window_m / gsd), a.shape[1], a.shape[2])
        y, x = (a.shape[1] - n) // 2, (a.shape[2] - n) // 2
        a = a[:, y : y + n, x : x + n]
    lo, hi = np.percentile(a, 2, axis=(1, 2)), np.percentile(a, 98, axis=(1, 2))
    a = np.clip((a - lo[:, None, None]) / (hi - lo + 1e-6)[:, None, None], 0, 1)
    return Image.fromarray((a.transpose(1, 2, 0) * 255).astype(np.uint8)), gsd


def tile_latlon(name: str) -> tuple[float, float]:
    """SN7 area names end in the slippy-map tile x, y and zoom (13)."""
    m = re.search(r"_(\d+)_(\d+)_13", name)
    assert m, name
    x, y = int(m[1]), int(m[2])
    return (
        math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / 8192)))),
        x / 8192 * 360 - 180,
    )


def closeup_sheet(out: Path, sheets: Path, per_row: int = 6, tile: int = 256) -> None:
    sheet = Image.new("RGB", (90 + tile * per_row, tile * len(CLOSEUPS)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, region in enumerate(CLOSEUPS):
        for c, f in enumerate(sorted((out / region).glob("*.tif"))[:per_row]):
            img, gsd = to_image(f, 200)  # the middle 200 m, so a house is the same size everywhere
            sheet.paste(img.resize((tile, tile)), (90 + c * tile, r * tile))
        draw.text((6, r * tile + 6), f"{region} {gsd:.2f} m", fill="black")
    sheet.save(sheets / "spacenet_regions.jpg", quality=88)


def global_sheet(out: Path, sheets: Path, cols: int = 10, tile: int = 170) -> None:
    files = sorted((out / "sn7").glob("*.tif"))
    rows = math.ceil(len(files) / cols)
    sheet = Image.new("RGB", (tile * cols, (tile + 14) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, f in enumerate(files):
        img, _ = to_image(f)
        x, y = (i % cols) * tile, (i // cols) * (tile + 14)
        sheet.paste(img.resize((tile, tile)), (x, y))
        lat, lon = tile_latlon(f.name)
        draw.text((x + 2, y + tile + 1), f"{i:02d} {lat:.1f},{lon:.1f}", fill="black")
    sheet.save(sheets / "sn7_grid.jpg", quality=85)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="inside data/eval (git-ignored)")
    ap.add_argument("--sheets", type=Path, required=True, help="inside data/review (git-ignored)")
    ap.add_argument("--per-region", type=int, default=8)
    a = ap.parse_args()
    for p in (a.out, a.sheets):
        if "data" not in p.resolve().parts:
            sys.exit(f"{p} must be inside data/ (git-ignored): these are not our images")
        p.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str]] = []
    for region, prefix in CLOSEUPS.items():
        jobs += [(region, k) for k in spread(listing(prefix)[0], a.per_region)]
    for area in listing(SN7, "/")[1]:
        keys = listing(area + "images/")[0]
        mid = [k for k in keys if "2019_06" in k] or keys[:1]
        jobs += [("sn7", k) for k in mid[:1]]
    log(f"{len(jobs)} tiles")
    for i, (region, key) in enumerate(jobs, 1):
        (a.out / region).mkdir(exist_ok=True)
        download(key, a.out / region)
        if i % 10 == 0:
            log(f"{i}/{len(jobs)}")
    (a.out / "EVAL_ONLY.txt").write_text(CITATION)
    closeup_sheet(a.out, a.sheets)
    global_sheet(a.out, a.sheets)
    log(f"wrote {a.sheets}/spacenet_regions.jpg and sn7_grid.jpg")


if __name__ == "__main__":
    main()
