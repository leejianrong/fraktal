"""A sheet with one clear-sky overview render per biome, to judge what each generator makes.

Usage: python biome_overview.py OUT.png [SEED]
Each panel is a 1 km window at 2 m per pixel of one procedural world (seed SEED, default 3). It
is cheap enough to run on a laptop at low priority. Needs the `report` extra.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

from fraktal import biomes
from fraktal.procedural import generate_world, read_buildings, read_fields, read_roads
from fraktal.render import RenderConfig, render_region

CENTER = (104.0, 1.5)


def panel(name: str, seed: int, work: Path) -> np.ndarray:
    config = biomes.resolve(name)
    world = work / name
    generate_world(world, CENTER, seed, config)
    # A 1 km window at a coarse pixel size. A tile is dropped when its label check finds a
    # boundary pixel the two rasterisers disagree on (seen with dense_urban at 2 m), so try a
    # few pixel sizes.
    for gsd in (2.0, 2.5, 1.6, 4.0):
        scene = RenderConfig(
            gsd_m=gsd, tile_px=int(1000 / gsd), size_m=1000.0, objects=1.0, variation=0.12,
            texture=0.7, seed=seed, tone=(0.92, 1.0, 0.86), sun_elevation_deg=62.0, ambient=0.35,
        )  # fmt: skip
        out = work / f"{name}_{gsd}"
        render_region(
            world, read_buildings(world), CENTER, out, scene,
            roads=read_roads(world), fields=read_fields(world),
        )  # fmt: skip
        tiles = sorted((out / "tiles").glob("r00_c00.tif"))
        if tiles:
            with rasterio.open(tiles[0]) as src:
                return np.moveaxis(src.read(), 0, -1)
    raise RuntimeError(f"no tile survived label validation for {name}")


def main(argv: list[str]) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, seed = Path(argv[0]), int(argv[1]) if len(argv) > 1 else 3
    names = biomes.names()
    fig, axs = plt.subplots(2, (len(names) + 1) // 2, figsize=(4.2 * ((len(names) + 1) // 2), 8.6))
    with tempfile.TemporaryDirectory() as tmp:
        for ax, name in zip(axs.ravel(), names, strict=False):
            ax.imshow(panel(name, seed, Path(tmp)))
            ax.set_title(name, fontsize=11)
            ax.axis("off")
        for ax in axs.ravel()[len(names) :]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
