"""Plan views of each biome's road network, for looking at (FRK-104).

Usage: python scripts/roads_preview.py OUT.png [--size-m 1200] [--seed 3] [--straight]

Draws arterials, local streets, footpaths and plantation tracks as lines on one canvas per biome.
``--straight`` zeroes the organic settings, so the old and the new networks can be put side by
side. Writes nothing but the PNG; worlds go to a temporary directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from fraktal import biomes, procedural

COLOURS = {
    "arterial": (60, 60, 70), "local": (200, 90, 40), "footpath": (30, 130, 200),
    "track": (120, 150, 60),
}  # fmt: skip
WIDTHS = {"arterial": 4, "local": 2, "footpath": 1, "track": 2}
CELL = 400
STRAIGHT = {
    "local_wander_deg": 0.0, "local_curl_deg": 0.0, "branch_rate": 0.0, "footpath_rate": 0.0,
}  # fmt: skip


def draw(roads, bounds, label: str) -> Image.Image:
    west, south, east, north = bounds
    scale = CELL / (east - west)
    img = Image.new("RGB", (CELL, CELL), (245, 243, 236))
    d = ImageDraw.Draw(img)
    for _, row in roads.iterrows():
        pts = [((x - west) * scale, (north - y) * scale) for x, y in row.geometry.coords]
        d.line(
            pts, fill=COLOURS.get(row.road_class, (0, 0, 0)), width=WIDTHS.get(row.road_class, 1)
        )
    d.text((6, 4), label, fill=(0, 0, 0))
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--size-m", type=float, default=1200.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--straight", action="store_true")
    a = p.parse_args()
    names = [n for n in biomes.BIOMES if n != "khartoum"] + ["khartoum"]
    sheet = Image.new("RGB", (CELL * 3, CELL * ((len(names) + 2) // 3)), (255, 255, 255))
    for i, name in enumerate(names):
        config = biomes.resolve(name, {"size_m": a.size_m})
        if a.straight:
            config = dataclasses.replace(
                config, roads=dataclasses.replace(config.roads, **STRAIGHT)
            )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "w"
            manifest = procedural.generate_world(out, (32.55, 15.64), a.seed, config)
            roads = procedural.read_roads(out)
            counts = roads["road_class"].value_counts().to_dict()
            bounds = manifest["bounds"]
        sheet.paste(draw(roads, bounds, f"{name} {counts}"), ((i % 3) * CELL, (i // 3) * CELL))
        print(name, counts, flush=True)
    sheet.save(a.out)


if __name__ == "__main__":
    main()
