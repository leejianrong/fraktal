"""Render one flat sheet of a material under a low sun, for several spec overrides (FRK-126).

Runs inside a Python with the ``bpy`` wheel. It loads the material through
``fraktal.blender_scene.load_material`` so it tests the real node setup, not a copy.

Usage: python material_probe.py ASSETS_DIR OUT_DIR [--material roof-tin-red] [--px 300] [--gsd 0.02]
"""

# ruff: noqa: I001  (bpy before mathutils)
import argparse
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

VARIANTS = {
    "off": {"rib_strength": 0.0, "rust": 0.0},
    "d01": {"rib_strength": 1.0, "rib_depth_m": 0.01, "rust": 0.0},
    "d02": {"rib_strength": 1.0, "rib_depth_m": 0.02, "rust": 0.0},
    "d04": {"rib_strength": 1.0, "rib_depth_m": 0.04, "rust": 0.0},
}
AZIMUTHS = (0.0, 90.0)


def load_scene_module(src: Path):
    path = src / "fraktal" / "blender_scene.py"
    spec = importlib.util.spec_from_file_location("blender_scene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("assets", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--material", default="roof-tin-red")
    ap.add_argument("--px", type=int, default=300)
    ap.add_argument("--gsd", type=float, default=0.02)
    ap.add_argument("--elevation", type=float, default=15.0)
    ap.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[2] / "src")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mod = load_scene_module(args.src)
    base = json.loads((args.assets / "materials" / f"{args.material}.json").read_text())
    tmp = Path(tempfile.mkdtemp())
    (tmp / "materials").mkdir()
    for sub in ("textures",):
        (tmp / sub).symlink_to((args.assets / sub).resolve())
    stats = {}
    for name, override in VARIANTS.items():
        spec = {**base, **override}
        (tmp / "materials" / f"probe-{name}.json").write_text(json.dumps(spec))
        for az in AZIMUTHS:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            sc = bpy.context.scene
            sc.render.engine, sc.cycles.device, sc.cycles.samples = "CYCLES", "CPU", 16
            sc.view_settings.view_transform = "Standard"
            sc.view_settings.exposure = -1.3
            size = args.px * args.gsd
            bpy.ops.mesh.primitive_plane_add(size=size * 1.5, location=(0, 0, 0))
            plane = bpy.context.object
            plane.data.materials.append(mod.load_material(f"probe-{name}", tmp))
            cam = bpy.data.cameras.new("c")
            cam.type, cam.ortho_scale = "ORTHO", size
            co = bpy.data.objects.new("c", cam)
            co.location = (0, 0, 50)
            bpy.context.collection.objects.link(co)
            sc.camera = co
            sun = bpy.data.lights.new("s", "SUN")
            sun.energy, sun.angle = 3.0, math.radians(0.53)
            so = bpy.data.objects.new("s", sun)
            bpy.context.collection.objects.link(so)
            e, a = math.radians(args.elevation), math.radians(az)
            so.rotation_euler = (math.pi / 2 - e, 0, a)
            sc.render.resolution_x = sc.render.resolution_y = args.px
            sc.render.filepath = str(args.out / f"{name}_az{int(az)}.png")
            sc.render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(sc.render.filepath)
            px = np.array(img.pixels[:]).reshape(args.px, args.px, 4)[..., :3].mean(-1)
            periods = {}
            for axis, label in ((0, "along_y"), (1, "along_x")):
                profile = px.mean(axis=axis)
                spec_ = np.abs(np.fft.rfft(profile - profile.mean()))
                k = int(np.argmax(spec_[2:]) + 2)  # skip the slow lighting trend
                periods[label] = round(args.px * args.gsd / k, 4)
            stats[f"{name}_az{int(az)}"] = {
                "mean": float(px.mean()),
                "std": float(px.std()),
                "stripe_period_m": periods,
            }
            print(f"PROBE periods {name} az={az} {periods}", flush=True)
            print(f"PROBE {name} az={az} mean={px.mean():.4f} std={px.std():.4f}", flush=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
