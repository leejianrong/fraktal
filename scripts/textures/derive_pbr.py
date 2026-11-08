"""Write normal and roughness maps for the colour textures in assets/ and wire them into the
materials and the manifest (FRK-126). Usage: python derive_pbr.py [--assets assets]

Each ``textures/<name>-color.jpg`` (or ``gen-*.jpg``) gets ``<name>-normal.jpg`` and
``<name>-rough.jpg`` from ``fraktal.pbr``; every material that uses the colour image gains
``normal`` and ``roughness_map`` keys; and each new file gets a manifest row that copies the
source row's licence (and generator block). JPEG keeps the repository small (a PNG normal map
was half a megabyte). Safe to run again: it rewrites the same files and rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fraktal import assets, pbr

# strength, detail blur sigma (px), roughness base: tin has fine ribs and a lower roughness
PARAMS = {"corrugatedsteel007a": (6.0, 10.0, 0.5)}
DEFAULT = (4.0, 24.0, 0.8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("assets"))
    root = parser.parse_args().assets
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = {r["id"]: r for r in manifest["assets"]}
    by_file = {r["file"]: r for r in manifest["assets"]}
    sources = sorted(
        p
        for p in (root / "textures").glob("*.jpg")
        if "-normal" not in p.name and "-rough" not in p.name
    )
    for src in sources:
        stem = src.stem.removesuffix("-color")
        strength, sigma, base = PARAMS.get(stem, DEFAULT)
        normal, rough = pbr.derive(
            np.asarray(Image.open(src).convert("RGB")), strength, sigma, base
        )
        outputs = {"normal": (root / "textures" / f"{stem}-normal.jpg", normal),
                   "rough": (root / "textures" / f"{stem}-rough.jpg", rough)}  # fmt: skip
        source_row = by_file[f"textures/{src.name}"]
        for kind, (path, data) in outputs.items():
            Image.fromarray(data).save(path, quality=92)
            row_id = f"tex-{stem}-{kind}"
            row = {
                **{k: v for k, v in source_row.items() if k != "generator"},
                "id": row_id,
                "file": f"textures/{path.name}",
                "source": f"derived with fraktal.pbr from {source_row['id']} ({kind} map)",
                "sha256": assets.sha256(path),
            }
            if "generator" in source_row:
                row["generator"] = source_row["generator"]
            rows[row_id] = row
        for mat_file in sorted((root / "materials").glob("*.json")):
            spec = json.loads(mat_file.read_text())
            if spec.get("image") == f"textures/{src.name}":
                spec["normal"] = f"textures/{stem}-normal.jpg"
                spec["roughness_map"] = f"textures/{stem}-rough.jpg"
                mat_file.write_text(json.dumps(spec, indent=2) + "\n")
                rows[f"mat-{mat_file.stem}"]["sha256"] = assets.sha256(mat_file)
    ordered = [r for r in manifest["assets"] if r["id"] in rows]
    ordered_ids = {r["id"] for r in ordered}
    manifest["assets"] = [rows[r["id"]] for r in ordered] + [
        r for i, r in rows.items() if i not in ordered_ids
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(sources)} textures, {len(sources) * 2} maps written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
