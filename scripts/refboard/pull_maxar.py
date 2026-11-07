"""Pull a few real satellite windows from Maxar Open Data for the reference board (FRK-99).

EVALUATION AND LOOKING ONLY. Maxar Open Data is CC BY-NC 4.0 (registry page checked 2026-11-07),
so the chips stay under data/eval/ (git-ignored), are never trained on, redistributed or used in
sales material (ADR-0012). Each scene is a cloud-optimised GeoTIFF, so only the windows we keep are
downloaded.

Usage: uv run python scripts/refboard/pull_maxar.py --out data/eval/maxar-reference
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import rasterio
from rasterio.windows import Window

os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

BUCKET = "https://maxar-opendata.s3.amazonaws.com/"
EVENTS = [  # tropical events; the biome each is closest to is a guess, for matching on the board
    "Indonesia-Earthquake22",
    "Typhoon-Kalmaegi-Nov-2025",
    "Earthquake-Myanmar-March-2025",
    "Cyclone-Senyar-Indonesia-Nov-2025",
    "BayofBengal-Cyclone-Mocha-May-23",
    "Kenya-Flooding-May24",
    "Sudan-flooding-8-22-2022",
]
WINDOW = 1024
CELL = 64  # thumbnail cell used to rank windows by texture, in thumbnail pixels


def _get(url: str) -> bytes:
    for attempt in range(6):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def visual_tiles(event: str) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    token = None
    while len(keys) < 4000:
        url = f"{BUCKET}?list-type=2&prefix=events/{event}/ard/&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        text = _get(url).decode()
        root = ElementTree.fromstring(text)
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for item in root.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns) or ""
            if key.endswith("-visual.tif"):
                keys.append((key, int(item.findtext("s:Size", namespaces=ns) or 0)))
        found = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", text)
        if not found:
            break
        token = found.group(1)
    return keys


def pick_tiles(keys: list[tuple[str, int]], n: int = 3) -> list[str]:
    """The biggest scenes, one per area (quadkey prefix), best first."""
    out: list[str] = []
    seen: set[str] = set()
    for key, _ in sorted(keys, key=lambda t: -t[1]):
        quad = key.split("/ard/")[1].split("/")[1][:8]
        if quad in seen:
            continue
        seen.add(quad)
        out.append(key)
        if len(out) == n * 3:  # spare candidates: some scenes are single-band
            break
    return out


def windows(src: rasterio.DatasetReader) -> list[tuple[str, Window]]:
    """Two windows from a thumbnail: one with the most large-scale structure, one from the upper
    middle. Cloud, sea (no large-scale structure) and scene edges are rejected."""
    scale = max(1, max(src.width, src.height) // 1024)
    shape = (src.height // scale, src.width // scale)
    thumb = src.read([1, 2, 3], out_shape=shape).astype("float32")
    grey = thumb.mean(axis=0)
    cloud = thumb.min(axis=0) > 215
    step = max(1, WINDOW // scale)
    cells = []
    for r in range(0, shape[0] - step + 1, step):
        for c in range(0, shape[1] - step + 1, step):
            g = grey[r : r + step, c : c + step]
            blocks = (
                g[: step // 8 * 8, : step // 8 * 8].reshape(step // 8, 8, step // 8, 8).mean((1, 3))
            )
            cells.append(
                (
                    float((g == 0).mean()),
                    float(cloud[r : r + step, c : c + step].mean()),
                    float(blocks.std()),
                    r * scale,
                    c * scale,
                )
            )
    for gap_limit in (0.02, 0.1, 0.25):  # prefer full windows; accept some no-data at a scene edge
        ok = sorted(
            (std, r, c)
            for gap, cl, std, r, c in cells
            if gap <= gap_limit and cl <= 0.05 and std >= 8
        )
        if len(ok) >= 2:
            break
    else:
        return []
    picks = [("busy", ok[-1]), ("median", ok[len(ok) * 2 // 3])]
    return [
        (name, Window(c, r, min(WINDOW, src.width - c), min(WINDOW, src.height - r)))
        for name, (_, r, c) in picks
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    (args.out / "EVAL_ONLY.txt").write_text(
        "private evaluation and looking only: never train on, redistribute, commit or ship "
        "these chips\n"
        "Licence: CC BY-NC 4.0 (https://registry.opendata.aws/maxar-open-data/)\n"
        f"Maxar Open Data Program was accessed on {today} from https://registry.opendata.aws/maxar-open-data.\n"
    )
    for event in EVENTS:
        kept = 0
        for key in pick_tiles(visual_tiles(event)):
            if kept == 3:
                break
            scene = key.split("/ard/")[1]
            with rasterio.open("/vsicurl/" + BUCKET + key) as src:
                if src.count < 3:  # panchromatic only: no colour to look at
                    continue
                found = windows(src)
                kept += bool(found)
                for kind, window in found:
                    name = f"{event}__{scene.replace('/', '_').removesuffix('-visual.tif')}__{kind}"
                    data = src.read([1, 2, 3], window=window)
                    np.save(args.out / f"{name}.npy", data)
                    meta = {
                        "event": event,
                        "scene": scene,
                        "kind": kind,
                        "window": [window.col_off, window.row_off, window.width, window.height],
                        "pixel_size": [abs(src.res[0]), abs(src.res[1])],
                        "crs": str(src.crs),
                        "bounds": list(rasterio.windows.bounds(window, src.transform)),
                    }
                    (args.out / f"{name}.json").write_text(json.dumps(meta) + "\n")
                    print(f"{name} {data.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
