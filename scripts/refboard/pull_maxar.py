"""Pull a few real satellite windows from Maxar Open Data for the reference board (FRK-99).

EVALUATION AND LOOKING ONLY. Maxar Open Data is CC BY-NC 4.0 (registry page checked 2026-11-07),
so the chips stay under data/eval/ (git-ignored), are never trained on, redistributed or used in
sales material (ADR-0012). Each scene is a cloud-optimised GeoTIFF, so only the windows we keep are
downloaded.

Usage: uv run python scripts/refboard/pull_maxar.py --out data/eval/maxar-reference
"""

from __future__ import annotations

import argparse
import contextlib
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
# Approximate dates of each event, to call a scene before or after it (a scene on the day counts as
# after). Taken from the event descriptions and public reporting, not from the scene metadata, so a
# phase label near the date can be wrong by a few days.
EVENT_DATES = {
    "Indonesia-Earthquake22": "2022-11-21",  # Cianjur earthquake; the set ends a week later
    "Typhoon-Kalmaegi-Nov-2025": "2025-11-04",
    "Earthquake-Myanmar-March-2025": "2025-03-28",
    "Cyclone-Senyar-Indonesia-Nov-2025": "2025-11-27",
    "BayofBengal-Cyclone-Mocha-May-23": "2023-05-14",
    "Kenya-Flooding-May24": "2024-04-29",  # Mai Mahiu dam burst; heavy rain from March
    "Sudan-flooding-8-22-2022": "2022-07-01",  # start of the flood season
}
PAIRS_PER_EVENT = 3
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


def plan(
    event: str, keys: list[tuple[str, int]], limit: int | None = None
) -> list[tuple[str | None, str | None]]:
    """(pre, post) scene keys to try, best first, at most ``limit``.

    The same ARD tile before and after the event comes first, since it shows the same ground twice;
    tiles with only one side fill the rest.
    """
    event_date = EVENT_DATES[event]
    tiles: dict[str, dict[str, tuple[str, int]]] = {}
    for key, size in keys:
        _, quad, date = key.split("/ard/")[1].split("/")[:3]
        tiles.setdefault(quad, {})[date] = (key, size)

    def best(scenes: dict[str, tuple[str, int]], after: bool) -> tuple[str, int] | None:
        found = [v for d, v in scenes.items() if (d >= event_date) == after]
        return max(found, key=lambda v: v[1]) if found else None

    both, single = [], []
    for scenes in tiles.values():
        pre, post = best(scenes, False), best(scenes, True)
        size = max(v[1] for v in (pre, post) if v)
        (both if pre and post else single).append((size, pre and pre[0], post and post[0]))
    chosen = sorted(both, reverse=True) + sorted(single, reverse=True)
    return [(pre, post) for _, pre, post in chosen[:limit]]


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


def save_pair(out: Path, event: str, pre: str | None, post: str | None) -> int:
    """Windows chosen on the first available scene, then the same ground read from the other."""
    from fraktal.refboard import usable

    scenes = {"pre": pre, "post": post}
    with contextlib.ExitStack() as stack:
        opened = {
            phase: stack.enter_context(rasterio.open("/vsicurl/" + BUCKET + key))
            for phase, key in scenes.items()
            if key
        }
        opened = {p: src for p, src in opened.items() if src.count >= 3}  # skip panchromatic scenes
        if not opened:
            return 0
        saved = 0
        first = next(iter(opened.values()))
        for kind, window in windows(first):
            bounds = rasterio.windows.bounds(window, first.transform)
            chips = {}
            for phase, src in opened.items():
                win = rasterio.windows.from_bounds(*bounds, transform=src.transform).round_offsets()
                win = win.round_lengths()
                data = src.read([1, 2, 3], window=win, boundless=False)
                if usable(np.moveaxis(data, 0, -1)):
                    chips[phase] = (src, win, data)
            # A side that is cloudy or empty is skipped; the clear side is still kept, as unpaired.
            for phase, (src, win, data) in chips.items():
                key = scenes[phase] or ""
                scene = key.split("/ard/")[1]
                name = f"{event}__{scene.replace('/', '_').removesuffix('-visual.tif')}__{kind}"
                np.save(out / f"{name}.npy", data)
                meta = {
                    "event": event,
                    "scene": scene,
                    "kind": kind,
                    "phase": phase,
                    "event_date": EVENT_DATES[event],
                    "paired": len(chips) == 2,
                    "window": [win.col_off, win.row_off, win.width, win.height],
                    "pixel_size": [abs(src.res[0]), abs(src.res[1])],
                    "crs": str(src.crs),
                    "bounds": list(rasterio.windows.bounds(win, src.transform)),
                }
                (out / f"{name}.json").write_text(json.dumps(meta) + "\n")
                print(f"{name} {data.shape} {phase}", flush=True)
                saved += 1
        return saved


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
        done = 0
        for pre, post in plan(event, visual_tiles(event), limit=40):  # many are sea, cloud or grey
            done += save_pair(args.out, event, pre, post) > 0
            if done == PAIRS_PER_EVENT:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
