"""Pull street-level reference photos for the reference board (FRK-119). Looking only.

Sources: Mapillary (bounding-box search, needs MAPILLARY_TOKEN, a read-only client token) and
Wikimedia Commons (no token). Photos go under data/eval/ground-reference (git-ignored) and are
never trained on, used to condition or fine-tune a model, redistributed or put in a dataset
(ADR-0012). Mapillary images are reported as CC BY-SA 4.0 and Commons files carry their own licence,
which is recorded beside each photo.

The token is read from the environment, else by name from ./.env, and is sent in an Authorization
header, so it is in no URL and nothing here prints it.

Usage: uv run python scripts/refboard/pull_ground.py --out data/eval/ground-reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = "FraktalRefBoard/0.1 (reference imagery for a research tool)"
HALF = 0.0045  # Mapillary bounding boxes must be smaller than 0.01 degrees square
PLACES = {  # name: (lat, lon) of a busy street area, and a Commons search phrase
    "nairobi-centre": (-1.2864, 36.8219, "Nairobi street"),
    "nairobi-mathare": (-1.2590, 36.8590, "Mathare Nairobi street"),
    "kampala": (0.3136, 32.5811, "Kampala street"),
    "dar-es-salaam": (-6.8160, 39.2803, "Dar es Salaam street"),
    "accra": (5.6037, -0.1870, "Accra street"),
    "jakarta": (-6.1754, 106.8272, "Jakarta kampung street"),
    "manila": (14.5995, 120.9842, "Manila street"),
    "cebu": (10.3157, 123.8854, "Cebu City street"),
    "mandalay": (21.9750, 96.0836, "Mandalay street"),
    "khartoum": (15.5007, 32.5599, "Khartoum street"),
    "manila-quezon": (14.6507, 121.0495, "Quezon City street"),
    "khartoum-bahri": (15.6350, 32.5400, "Khartoum Bahri street"),
}
STREET_WORDS = re.compile(
    r"street|road|avenue|market|traffic|slum|settlement|neighbou?rhood|junction|boulevard|alley|bus",
    re.I,
)  # Commons search returns portraits and interiors too; keep titles that name a street scene


def token() -> str | None:
    value = os.environ.get("MAPILLARY_TOKEN")
    if value:
        return value
    env = Path(".env")
    if env.is_file():
        for line in env.read_text().splitlines():
            key, _, rest = line.partition("=")
            if key.strip() == "MAPILLARY_TOKEN":
                return rest.strip().strip("\"'")
    return None


def fetch(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=60).read()
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(
        f"could not fetch {url.split('?')[0]}"
    )  # no query string: it may carry a token


def pick(items: list[dict], count: int, key: str) -> list[dict]:
    """A deterministic spread: sort by a hash of the id, so no place is favoured by file order."""
    ranked = sorted(items, key=lambda i: hashlib.sha256(str(i[key]).encode()).hexdigest())
    return ranked[:count]


def mapillary(out: Path, per_place: int, tok: str) -> int:
    saved = 0
    for name, (lat, lon, _) in PLACES.items():
        bbox = f"{lon - HALF},{lat - HALF},{lon + HALF},{lat + HALF}"
        fields = "id,captured_at,compass_angle,is_pano,sequence,geometry,thumb_1024_url"
        url = f"https://graph.mapillary.com/images?bbox={bbox}&fields={fields}&limit=100"
        try:
            data = json.loads(fetch(url, {"Authorization": f"OAuth {tok}"}))["data"]
        except (RuntimeError, ValueError, KeyError):
            data = []
        usable = [d for d in data if not d.get("is_pano") and d.get("thumb_1024_url")]
        # one image per sequence, so a drive does not fill the quota
        by_sequence = {d["sequence"]: d for d in sorted(usable, key=lambda d: d["id"])}
        chosen = pick(list(by_sequence.values()), per_place, "id")
        found = f"{len(data)} images, {len(by_sequence)} sequences"
        print(f"mapillary {name}: {found}, keeping {len(chosen)}")
        for d in chosen:
            stem = f"mapillary__{name}__{d['id']}"
            if (out / f"{stem}.jpg").exists():
                continue
            (out / f"{stem}.jpg").write_bytes(fetch(d["thumb_1024_url"]))
            meta = {
                "source": "mapillary",
                "place": name,
                "id": d["id"],
                "captured_at": d.get("captured_at"),
                "compass_angle": d.get("compass_angle"),
                "lonlat": d.get("geometry", {}).get("coordinates"),
                "licence": "CC BY-SA 4.0 (reported by Mapillary, not re-read)",
            }
            (out / f"{stem}.json").write_text(json.dumps(meta) + "\n")
            saved += 1
    return saved


def commons(out: Path, per_place: int) -> int:
    saved = 0
    for name, (_, _, phrase) in PLACES.items():
        params = {
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": "6",
            "gsrsearch": f"{phrase} filetype:bitmap", "gsrlimit": "40", "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata", "iiurlwidth": "1024",
        }  # fmt: skip
        try:
            pages = json.loads(
                fetch("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params))
            )
        except (RuntimeError, ValueError):
            pages = {}
        good = []
        for page in pages.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            ext = info.get("extmetadata", {})
            if (
                info.get("mime") == "image/jpeg"
                and info.get("width", 0) >= 1200
                and info.get("thumburl")
                and STREET_WORDS.search(page["title"])
            ):
                good.append({"title": page["title"], "info": info, "ext": ext})
        chosen = pick(good, per_place, "title")
        print(f"commons {name}: {len(good)} candidates, keeping {len(chosen)}")
        for c in chosen:
            stem = (
                "commons__"
                + name
                + "__"
                + re.sub(r"[^A-Za-z0-9]+", "-", c["title"][5:60]).strip("-")
            )
            if (out / f"{stem}.jpg").exists():
                continue
            (out / f"{stem}.jpg").write_bytes(fetch(c["info"]["thumburl"]))
            licence = c["ext"].get("LicenseShortName", {}).get("value", "unknown")
            artist = c["ext"].get("Artist", {}).get("value", "unknown")
            author = re.sub(r"<[^>]+>", "", artist)[:80]
            meta = {
                "source": "commons",
                "place": name,
                "title": c["title"],
                "licence": licence,
                "author": author,
                "url": c["info"].get("descriptionurl"),
            }
            (out / f"{stem}.json").write_text(json.dumps(meta) + "\n")
            saved += 1
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mapillary-per-place", type=int, default=5)
    parser.add_argument("--commons-per-place", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "EVAL_ONLY.txt").write_text(
        "private evaluation and looking only: never train, fine-tune or condition a model\n"
        "on these photos, and never redistribute, commit or ship them.\n"
        "Licences are in each metadata file.\n"
    )
    total = commons(args.out, args.commons_per_place)
    tok = token()
    if tok:
        total += mapillary(args.out, args.mapillary_per_place, tok)
    else:
        print("no MAPILLARY_TOKEN: skipped Mapillary")
    print(f"saved {total} new photos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
