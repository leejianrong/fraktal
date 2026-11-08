"""Write every propgen variant to one JSON file for scripts/props/bpy_build.py (FRK-130).

Usage: python propgen_dump.py OUT.json [--variants 10]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fraktal.log import emit
from fraktal.propgen import GENERATORS, MATERIALS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--variants", type=int, default=10)
    args = parser.parse_args()
    props, timing = [], {}
    for kind, make in GENERATORS.items():
        t = time.perf_counter()
        for seed in range(args.variants):
            prop = make(seed)
            props.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "size_m": [round(s, 3) for s in prop.size()],
                    "parts": [
                        {
                            "material": p.material,
                            "verts": p.verts.round(4).tolist(),
                            "faces": p.faces.tolist(),
                        }
                        for p in prop.parts
                    ],
                }
            )
        timing[kind] = round((time.perf_counter() - t) / args.variants * 1000, 2)
        emit("propgen_kind", kind=kind, ms_per_variant=timing[kind])
    args.out.write_text(json.dumps({"materials": MATERIALS, "props": props, "gen_ms": timing}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
