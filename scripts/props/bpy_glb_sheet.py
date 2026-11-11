"""Render a folder of generated GLB props in Blender (FRK-123). Runs inside a Python that has the
``bpy`` wheel; nothing is imported from fraktal.

Usage: python bpy_glb_sheet.py MESHES_DIR OUT_DIR [--px 320] [--samples 16]

Each ``<id>.glb`` (metres, z up, standing on z = 0, as ``meshqa`` leaves it) gets a grey clay
material, a ground plane and the same two views as the code-written props (``bpy_build.py``): a
ground-level three-quarter view and a nadir view at aerial scale, rendered with Cycles on the CPU.
Seconds per prop go to timing.json. A GLB that fails to import is recorded, not hidden.
"""

# ruff: noqa: I001  (bpy before mathutils)
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


def say(msg: str) -> None:
    print(f"STAGE {msg}", flush=True)


def reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -1.3
    world = bpy.data.worlds.new("sky")
    scene.world = world
    world.use_nodes = True
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    world.node_tree.links.new(
        sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"]
    )
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    scene.collection.objects.link(sun)


def clay(name: str, rgb: tuple, rough: float = 0.8):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def camera(location, look_at, ortho: float | None = None) -> None:
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = location
    cam.rotation_euler = (Vector(look_at) - Vector(location)).to_track_quat("-Z", "Y").to_euler()
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
    else:
        cam.data.lens = 50


def render(path: Path, px: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = px
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("meshes", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--px", type=int, default=320)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    (args.out / "png").mkdir(parents=True, exist_ok=True)
    timing: dict = {}
    for glb in sorted(args.meshes.glob("*.glb")):
        name = glb.stem
        t = time.perf_counter()
        try:
            reset()
            bpy.ops.import_scene.gltf(filepath=str(glb))
            objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
            if not objs:
                raise RuntimeError("no mesh in the GLB")
            body = clay("clay", (0.55, 0.52, 0.48))
            for o in objs:
                o.data.materials.clear()
                o.data.materials.append(body)
            corners = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
            lo = Vector([min(c[i] for c in corners) for i in range(3)])
            hi = Vector([max(c[i] for c in corners) for i in range(3)])
            lx, ly, lz = (hi - lo)[0], (hi - lo)[1], (hi - lo)[2]
            ground = bpy.data.meshes.new("ground")
            s = 30.0
            ground.from_pydata(
                [(-s, -s, -0.001), (s, -s, -0.001), (s, s, -0.001), (-s, s, -0.001)],
                [], [(0, 1, 2, 3)],
            )  # fmt: skip
            gobj = bpy.data.objects.new("ground", ground)
            gobj.data.materials.append(clay("dirt", (0.22, 0.18, 0.12), 0.95))
            bpy.context.scene.collection.objects.link(gobj)
            reach = max(lx, ly, lz)
            camera((reach * 1.7, -reach * 1.7, lz * 0.55 + 0.6), (0, 0, lz / 2))
            render(args.out / "png" / f"{name}-ground.png", args.px, args.samples)
            for cam in [o for o in bpy.data.objects if o.type == "CAMERA"]:
                bpy.data.objects.remove(cam)
            bpy.context.scene.view_settings.exposure = -2.3
            camera((0, 0, 40.0), (0, 0, 0), ortho=max(lx, ly) * 1.5 + 1.0)
            render(args.out / "png" / f"{name}-aerial.png", args.px, args.samples)
            timing[name] = round(time.perf_counter() - t, 2)
            say(f"rendered {name} {timing[name]}s")
        except Exception as err:  # noqa: BLE001 - a failed prop is a result
            timing[name] = f"error: {type(err).__name__}: {err}"[:200]
            say(f"FAILED {name} {timing[name]}")
    (args.out / "timing.json").write_text(json.dumps(timing, indent=1))
    return 0 if all(isinstance(v, float) for v in timing.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
