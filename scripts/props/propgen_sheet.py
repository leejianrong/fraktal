"""Contact sheet of the propgen renders, one row per kind: ground view then aerial view of each
variant (FRK-130). Usage: python propgen_sheet.py PNG_DIR OUT.jpg [--cell 200]"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from fraktal.propgen import GENERATORS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("png", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cell", type=int, default=200)
    parser.add_argument("--variants", type=int, default=10)
    args = parser.parse_args()
    c, n = args.cell, args.variants
    sheet = Image.new("RGB", (c * n, (c * 2 + 18) * len(GENERATORS)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, kind in enumerate(GENERATORS):
        y0 = row * (c * 2 + 18)
        draw.text(
            (4, y0 + 2), f"{kind}: ground (top) and aerial (bottom), seeds 0-{n - 1}", fill="black"
        )
        for seed in range(n):
            for k, view in enumerate(("ground", "aerial")):
                f = args.png / f"{kind}-s{seed:02d}-{view}.png"
                if f.exists():
                    sheet.paste(
                        Image.open(f).convert("RGB").resize((c, c)), (seed * c, y0 + 18 + k * c)
                    )
    sheet.save(args.out, quality=88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
