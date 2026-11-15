#!/usr/bin/env python
"""Pack the Godot viewer, one exported tile and Godot itself into a Windows demo zip (FRK-116).

    uv run python scripts/godot/make_demo_zip.py --scene data/scene_demo \\
        --godot-zip ~/.local/opt/godot-dl/godot-win64.zip --out data/dist/fraktal-demo.zip

The zip unpacks to one folder; the user double-clicks `Run Fraktal.bat`. It holds the official Godot
4.7.2 Windows build (MIT, checked here against SHA512-SUMS.txt next to the zip), `project/` (the
viewer, with the script cache Godot builds on import), `tile/` (what `fraktal scene --agents` wrote,
without `ground_rough.npy`, which the viewer does not read) and `assets/` (materials and textures).
Nothing from `data/eval` or any real photograph can end up in it: it only copies those four things.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP = "fraktal-demo"
SKIP_TILE = {"ground_rough.npy"}  # the viewer never reads it
# The script cache is what lets `class_name` scripts be found; Godot builds it on import.
PROJECT_FILES = (
    "project.godot", "main.tscn", ".godot/global_script_class_cache.cfg", ".godot/uid_cache.bin",
)  # fmt: skip
README = """Fraktal viewer (a synthetic city you can drive, walk and fly in)

Double-click "Run Fraktal.bat". The first start takes a few seconds to load.

  1  car       W/S throttle and brake, A/D steer, Space handbrake
  2  person    W/S/A/D walk, Shift runs, mouse look (click to capture, Esc to release) or arrow keys
  3  drone     W/S/A/D fly where you look, E up, Q down, Shift fast
  R  reset     H is not used here; close the window to quit

Cars and people you see are a recorded simulation that loops; they do not react to you.
Everything is synthetic. Godot is MIT licensed (godotengine.org). If nothing opens, run
"Fraktal.console.exe" the same way to see the messages.
"""
BAT = (
    '@echo off\r\ncd /d "%~dp0"\r\n'
    'start "" "%~dp0Fraktal.exe" --path "%~dp0project" -- --scene "%~dp0tile"\r\n'
)


def sha512(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def check_godot(zip_path: Path) -> None:
    sums = zip_path.with_name("SHA512-SUMS.txt")
    if not sums.is_file():
        raise SystemExit(f"{sums} is missing: download it from the same release page")
    want = {
        name: digest
        for digest, name in (line.split(None, 1) for line in sums.read_text().splitlines() if line)
    }
    expected = want.get("Godot_v4.7.2-stable_win64.exe.zip", "")
    if sha512(zip_path) != expected:
        raise SystemExit("the Godot zip does not match SHA512-SUMS.txt; not using it")


def build(scene: Path, godot_zip: Path, out: Path) -> list[str]:
    check_godot(godot_zip)
    project = ROOT / "godot"
    if not (project / ".godot").is_dir():
        raise SystemExit(
            "run `godot --headless --path godot --import` first (it builds the script cache)"
        )
    if not (scene / "scene.json").is_file():
        raise SystemExit(f"{scene} is not a tile written by `fraktal scene`")
    out.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:

        def put(src: Path, arc: str) -> None:
            z.write(src, f"{TOP}/{arc}")
            names.append(arc)

        with zipfile.ZipFile(godot_zip) as g:
            for member, arc in (
                ("Godot_v4.7.2-stable_win64.exe", "Fraktal.exe"),
                ("Godot_v4.7.2-stable_win64_console.exe", "Fraktal.console.exe"),
            ):
                z.writestr(f"{TOP}/{arc}", g.read(member))
                names.append(arc)
        z.writestr(f"{TOP}/Run Fraktal.bat", BAT)
        z.writestr(f"{TOP}/README.txt", README.replace("\n", "\r\n"))
        files = [project / f for f in PROJECT_FILES if (project / f).is_file()]
        for sub in ("scripts", "shapes"):
            files += sorted(p for p in (project / sub).rglob("*") if p.is_file())
        for path in files:
            put(path, f"project/{path.relative_to(project).as_posix()}")
        for path in sorted(scene.iterdir()):
            if path.is_file() and path.name not in SKIP_TILE:
                put(path, f"tile/{path.name}")
        for kind in ("materials", "textures"):
            for path in sorted((ROOT / "assets" / kind).iterdir()):
                if path.is_file():
                    put(path, f"assets/{kind}/{path.name}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--godot-zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    names = build(args.scene, args.godot_zip, args.out)
    print(f"{args.out}: {len(names)} files, {args.out.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
