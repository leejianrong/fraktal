#!/usr/bin/env python
"""Where does seed-to-seed brightness come from? (FRK-171)

  measure DIR...                     brightness, contrast, sky share of the frames in each strip dir
  relight SRC SKY_FROM OUT [--no-variety]
                                     render SRC's scene (layout, trees, props, camera) again in
                                     Blender with SKY_FROM's sky; OUT gets frames/. Beauty only.

A strip dir is data/review/street<tag>_<biome>. relight copies SRC's scene, swaps ``sky`` and runs
the same Blender script ``fraktal render3d`` does, so one factor moves at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from fraktal import streetview

ROOT = Path(__file__).resolve().parents[2]


def strip_stats(strip: Path) -> dict[str, float]:
    frames = sorted((strip / "out" / "review" / "fly" / "frames").glob("*.png"))
    rows = [streetview.statistics(streetview._read(f)) for f in frames]
    return {
        k: float(np.mean([r[k] for r in rows])) for k in ("brightness", "contrast", "sky_share")
    }


def relight(src: Path, sky_from: Path, out: Path, variety: bool) -> None:
    fly = src / "out" / "review" / "fly"
    scene = out / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    shutil.copytree(fly / "scene", scene)
    sky = json.loads((sky_from / "out" / "review" / "fly" / "scene" / "scene.json").read_text())[
        "sky"
    ]
    if not variety:
        sky.pop("variety", None)
    data = json.loads((scene / "scene.json").read_text())
    data["sky"] = sky
    (scene / "scene.json").write_text(json.dumps(data))
    frames = out / "out" / "review" / "fly" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    py = os.environ.get("FRAKTAL_BPY_PYTHON", sys.executable)
    script = ROOT / "src" / "fraktal" / "blender_scene.py"
    subprocess.run(
        [py, str(script), str(scene), str(frames), "--samples", "12", "--assets",
         str(ROOT / "assets"), "--px", "512"],
        check=True, stdout=subprocess.DEVNULL,
    )  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("dirs", nargs="+", type=Path)
    r = sub.add_parser("relight")
    r.add_argument("src", type=Path)
    r.add_argument("sky_from", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--no-variety", action="store_true")
    args = ap.parse_args()
    if args.cmd == "measure":
        for d in args.dirs:
            s = strip_stats(d)
            print(d.name, *(f"{k}={v:.3f}" for k, v in s.items()))
    else:
        relight(args.src, args.sky_from, args.out, not args.no_variety)
    return 0


if __name__ == "__main__":
    sys.exit(main())
