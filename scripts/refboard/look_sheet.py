"""One row per target look: our Blender overhead render at 4 m beside real SpaceNet SN7 cells.

EVALUATION AND LOOKING ONLY. The real tiles come from data/eval/spacenet-regions/sn7 (CC BY-SA 4.0,
ADR-0008), so the sheet is written under data/ and never committed or shown publicly. The cell
numbers are the ones on sn7_grid.jpg (see docs/research/overhead-region-survey.md).

Needs `FRAKTAL_BPY_PYTHON` (a Python with bpy, or scripts/wsl/winblender.sh), as `fraktal render3d`.

Usage: uv run python scripts/refboard/look_sheet.py \
           --sn7 data/eval/spacenet-regions/sn7 --work data/scratch/looks \
           --sheet data/review/regions/looks.jpg
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from fraktal import biomes
from fraktal.procedural import generate_world

sys.path.insert(0, str(Path(__file__).parent))
from pull_spacenet_regions import to_image  # noqa: E402

# (label, biome, centre lon/lat, SN7 cells). The centre only places the world; it is not the place.
LOOKS = [
    ("arid sprawl and relief", "arid_badlands", (32.5, 15.6), (34, 37, 30)),
    ("planned suburb", "planned_suburb", (-115.1, 36.1), (9, 6, 8)),
    ("farmed river plain", "river_plain", (-1.5, 52.0), (53, 54, 57)),
]
SIZE_M, GSD_M, TILE = 1800.0, 4.0, 300  # a little under the 2 km world, so it covers the tile


def render(biome: str, centre: tuple[float, float], work: Path, seed: int) -> Image.Image:
    world, out = work / f"{biome}-{seed}", work / f"{biome}-{seed}-img"
    generate_world(world, centre, seed, biomes.resolve(biome))
    cmd = [
        "uv", "run", "fraktal", "render3d", str(world), "--out", str(out),
        "--center", str(centre[0]), str(centre[1]), "--size-m", str(SIZE_M),
        "--gsd-m", str(GSD_M), "--samples", "16", "--device", "cpu",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)
    return Image.open(out / "image.png").convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sn7", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True, help="inside data/ (git-ignored)")
    ap.add_argument("--sheet", type=Path, required=True, help="inside data/ (git-ignored)")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    for p in (a.work, a.sheet):
        if "data" not in p.resolve().parts:
            sys.exit(f"{p} must be inside data/ (git-ignored): the sheet holds real tiles")
    a.work.mkdir(parents=True, exist_ok=True)
    a.sheet.parent.mkdir(parents=True, exist_ok=True)
    cells = sorted(a.sn7.glob("*.tif"))
    cols = 1 + max(len(look[3]) for look in LOOKS)
    sheet = Image.new("RGB", (TILE * cols, (TILE + 16) * len(LOOKS)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, biome, centre, picks) in enumerate(LOOKS):
        y = r * (TILE + 16)
        ours = render(biome, centre, a.work, a.seed).resize((TILE, TILE))
        sheet.paste(ours, (0, y))
        draw.text((3, y + TILE + 2), f"OURS {biome} ({label})", fill="black")
        for c, cell in enumerate(picks, 1):
            img, _ = to_image(cells[cell])
            sheet.paste(img.resize((TILE, TILE)), (c * TILE, y))
            draw.text((c * TILE + 3, y + TILE + 2), f"SN7 cell {cell:02d}", fill="black")
    sheet.save(a.sheet, quality=88)
    print(f"wrote {a.sheet}", file=sys.stderr)


if __name__ == "__main__":
    main()
