#!/usr/bin/env python
"""Write assets/textures/leaf-cluster.png from fraktal.leafcards (FRK-169), seed 0, 256 px.

The texture is ours (licence owned). Run it, then add or update the manifest row with the printed
sha256 and `fraktal assets`.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import rasterio

from fraktal.leafcards import leaf_cluster

OUT = Path(__file__).resolve().parents[2] / "assets" / "textures" / "leaf-cluster.png"


def main() -> int:
    img = leaf_cluster(0, 256)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUT, "w", driver="PNG", width=256, height=256, count=4, dtype="uint8"
    ) as dst:
        dst.write(img.transpose(2, 0, 1))
    print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
