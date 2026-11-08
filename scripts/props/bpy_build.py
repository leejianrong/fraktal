"""Build, render and export the propgen variants in Blender (FRK-130). Runs inside a Python that
has the ``bpy`` wheel; nothing else is imported from fraktal.

Usage: python bpy_build.py PROPS.json OUT_DIR [--px 320] [--samples 16]

For each prop: a ground-level three-quarter view and a nadir view at aerial scale, both rendered
with Cycles on the CPU, plus a GLB with the colours as materials. Per-variant seconds are printed
as ``STAGE`` lines and written to timing.json.
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
    scene.view_settings.exposure = -1.3  # the Nishita sky plus sun clips white plastic otherwise
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


def material(name: str, rgb: tuple[float, float, float], rough: float, metal: float):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return mat


def build(prop: dict, mats: dict) -> list:
    objs = []
    for i, part in enumerate(prop["parts"]):
        mesh = bpy.data.meshes.new(f"m{i}")
        mesh.from_pydata([tuple(v) for v in part["verts"]], [], [tuple(f) for f in part["faces"]])
        mesh.update()
        obj = bpy.data.objects.new(f"{prop['kind']}-{i}", mesh)
        obj.data.materials.append(mats[part["material"]])
        bpy.context.scene.collection.objects.link(obj)
        objs.append(obj)
    ground = bpy.data.meshes.new("ground")
    s = 30.0
    ground.from_pydata(
        [(-s, -s, -0.001), (s, -s, -0.001), (s, s, -0.001), (-s, s, -0.001)], [], [(0, 1, 2, 3)]
    )
    gobj = bpy.data.objects.new("ground", ground)
    gobj.data.materials.append(material("dirt", (0.22, 0.18, 0.12), 0.95, 0.0))
    bpy.context.scene.collection.objects.link(gobj)
    return objs


def camera(location, look_at, ortho: float | None = None):
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.location = location
    direction = Vector(look_at) - Vector(location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if ortho:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = ortho
    else:
        cam.data.lens = 50
    return cam


def render(path: Path, px: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = scene.render.resolution_y = px
    scene.cycles.samples = samples
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("props", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--px", type=int, default=320)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None)
    data = json.loads(args.props.read_text())
    for sub in ("png", "glb"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    timing = {}
    for prop in data["props"]:
        name = f"{prop['kind']}-s{prop['seed']:02d}"
        t = time.perf_counter()
        reset()
        mats = {k: material(k, *v) for k, v in data["materials"].items()}
        objs = build(prop, mats)
        lx, ly, lz = prop["size_m"]
        reach = max(lx, ly, lz)
        mid = (0, 0, lz / 2)
        camera((reach * 1.7, -reach * 1.7, lz * 0.55 + 0.6), mid)
        render(args.out / "png" / f"{name}-ground.png", args.px, args.samples)
        for cam in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(cam)
        camera((0, 0, 40.0), (0, 0, 0), ortho=max(lx, ly) * 1.5 + 1.0)  # aerial: straight down
        render(args.out / "png" / f"{name}-aerial.png", args.px, args.samples)
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.ops.export_scene.gltf(
            filepath=str(args.out / "glb" / f"{name}.glb"), use_selection=True
        )
        timing[name] = round(time.perf_counter() - t, 2)
        say(f"built {name} {timing[name]}s")
    (args.out / "timing.json").write_text(json.dumps(timing, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
