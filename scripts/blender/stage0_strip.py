"""Side-by-side strip for Stage 0: one real chip, the 2.5D render and the Blender render.

Usage: python stage0_strip.py WORK_DIR REAL_CHIPS_DIR OUT.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fraktal import chips

work, real_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads((real_dir / "chips.json").read_text())
real = chips.load_all(real_dir) if "tiles" in meta else chips.load_splits(real_dir)[0]
flat, blender = np.load(work / "flat.npy"), np.load(work / "blender.npy")
panels = [("real", real.get(0)[0])]
panels += [("2.5D", flat[:, :500, :500]), ("Blender", blender[:, :500, :500])]
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
for ax, (name, img) in zip(axes, panels, strict=True):
    ax.imshow(np.moveaxis(img, 0, 2))
    ax.set_title(name)
    ax.axis("off")
fig.tight_layout()
fig.savefig(out, dpi=100)
