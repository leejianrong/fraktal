"""Render one exported tile with Blender Cycles, nadir, orthographic (ADR-0010).

Not imported by fraktal: ``fraktal.render3d`` runs it as a script with a Python that has ``bpy``.

Usage: python blender_scene.py SCENE_DIR OUT.png [--px N] [--samples N] [--threads N]

Needs the ``bpy`` wheel (``pip install bpy``, Python 3.11). Reads ``scene.json`` and ``ground.npy``
written by ``fraktal.blender_export``; builds one ground plane, one extruded mesh for all the
buildings and a sun plus sky; writes an 8-bit PNG and prints timings as ``STAGE`` lines.
Materials are procedural (noise-modulated colour), so no third-party asset enters the render.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path

# ruff: noqa: I001  (bpy must be imported before bmesh and mathutils, so the order is not sorted)
import bpy
import bmesh
import numpy as np
from mathutils import Vector

# Wall and roof colours (0-1 linear-ish), echoing the 2.5D palette: concrete, clay, metal, bitumen.
ROOFS = [(0.55, 0.50, 0.45), (0.61, 0.36, 0.28), (0.73, 0.73, 0.72), (0.31, 0.31, 0.33)]
WALLS = [(0.80, 0.76, 0.68), (0.72, 0.62, 0.52), (0.86, 0.84, 0.80)]
MATERIAL_ROOF = {"tin": 2, "tile": 1}


def say(msg: str) -> None:
    print(f"STAGE {msg}", flush=True)


def material(name: str, rgb: tuple[float, float, float], rough: float, seed: int):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0 + seed % 5
    noise.inputs["Detail"].default_value = 8.0
    ramp = nodes.new("ShaderNodeMix")
    ramp.data_type = "RGBA"
    ramp.inputs[6].default_value = (*rgb, 1.0)
    ramp.inputs[7].default_value = (*(c * 0.72 for c in rgb), 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    links.new(ramp.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def write_png(path: str, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB PNG (Blender's own save of a generated image came out black)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def build_ground(scene: dict, ground: np.ndarray):
    size = scene["size_m"]
    h, w, _ = ground.shape
    path = str(Path(tempfile.gettempdir()) / "fraktal_ground.png")
    write_png(path, (np.clip(ground, 0.0, 1.0) * 255 + 0.5).astype("uint8"))
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"  # the colours are already display reflectance
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(size / 2, size / 2, 0.0))
    plane = bpy.context.object
    plane.scale = (size, size, 1.0)
    mat = bpy.data.materials.new("ground")
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"  # smooth class edges; the texture is already in the image
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    plane.data.materials.append(mat)


def build_buildings(scene: dict) -> int:
    mats = [material(f"roof{i}", c, 0.7, i) for i, c in enumerate(ROOFS)]
    mats += [material(f"wall{i}", c, 0.9, 10 + i) for i, c in enumerate(WALLS)]
    bm = bmesh.new()
    rng = random.Random(0)
    n = 0
    for b in scene["buildings"]:
        ring = b["ring"][:-1]
        if len(ring) < 3:
            continue
        height = max(2.0, b["height_m"])
        roof = MATERIAL_ROOF.get(b["roof_material"] or "", rng.randrange(len(ROOFS)))
        wall = len(ROOFS) + rng.randrange(len(WALLS))
        low = [bm.verts.new((x, y, 0.0)) for x, y in ring]
        high = [bm.verts.new((x, y, height)) for x, y in ring]
        try:
            top = bm.faces.new(high)
            top.material_index = roof
            for i in range(len(ring)):
                j = (i + 1) % len(ring)
                face = bm.faces.new((low[i], low[j], high[j], high[i]))
                face.material_index = wall
            n += 1
        except ValueError:  # a degenerate footprint
            continue
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("buildings")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("buildings", mesh)
    bpy.context.collection.objects.link(obj)
    for m in mats:
        mesh.materials.append(m)
    return n


FOLIAGE = [(0.15, 0.31, 0.13), (0.19, 0.36, 0.14), (0.12, 0.26, 0.12)]


def build_trees(scene: dict) -> int:
    """One mesh of crown ellipsoids (a low icosphere each), tall ones on a short hidden trunk."""
    trees = scene.get("trees", [])
    if not trees:
        return 0
    template = bmesh.new()
    bmesh.ops.create_icosphere(template, subdivisions=2, radius=1.0)
    verts = [v.co.copy() for v in template.verts]
    index = {v: i for i, v in enumerate(template.verts)}
    faces = [[index[v] for v in f.verts] for f in template.faces]
    template.free()
    mats = [material(f"leaf{i}", c, 0.8, 20 + i) for i, c in enumerate(FOLIAGE)]
    bm = bmesh.new()
    for k, (x, y, radius, height) in enumerate(trees):
        half = max(radius * 0.9, height * 0.3)  # crown half-height; the crown sits above 0.4 h
        zc = max(half, height - half)
        ring = [bm.verts.new((x + v.x * radius, y + v.y * radius, zc + v.z * half)) for v in verts]
        for f in faces:
            face = bm.faces.new([ring[i] for i in f])
            face.material_index = k % len(mats)
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("trees")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("trees", mesh)
    bpy.context.collection.objects.link(obj)
    for m in mats:
        mesh.materials.append(m)
    return len(trees)


def build_light(scene: dict) -> None:
    az, el = scene["sun"]["azimuth_deg"], scene["sun"]["elevation_deg"]
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 2.6  # about 0.6 of reflectance for a flat surface at 50 degrees elevation
    sun.angle = math.radians(0.53)
    obj = bpy.data.objects.new("sun", sun)
    bpy.context.collection.objects.link(obj)
    # Azimuth is clockwise from north (x east, y north); light travels away from the sun.
    a, e = math.radians(az), math.radians(el)
    toward_sun = Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))
    obj.rotation_euler = (-toward_sun).to_track_quat("-Z", "Y").to_euler()
    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.62, 0.72, 0.9, 1.0)  # clear-sky fill light
    bg.inputs["Strength"].default_value = 0.55
    bpy.context.scene.world = world


def build_camera(scene: dict, px: int) -> None:
    cam = bpy.data.cameras.new("cam")
    cam.type = "ORTHO"
    cam.ortho_scale = scene["size_m"]
    cam.clip_start, cam.clip_end = 1.0, 10000.0
    obj = bpy.data.objects.new("cam", cam)
    obj.location = (scene["size_m"] / 2, scene["size_m"] / 2, 3000.0)
    bpy.context.collection.objects.link(obj)  # looks down -Z, +Y is up in the image: north up
    bpy.context.scene.camera = obj
    render = bpy.context.scene.render
    render.resolution_x = render.resolution_y = px
    render.resolution_percentage = 100


def build_perspective_camera(px: int):
    """A perspective camera that ``pose_camera`` moves from frame to frame."""
    cam = bpy.data.cameras.new("cam")
    cam.type = "PERSP"
    cam.clip_start, cam.clip_end = 1.0, 20000.0
    obj = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    render = bpy.context.scene.render
    render.resolution_x = render.resolution_y = px
    render.resolution_percentage = 100
    return obj


def pose_camera(obj, pose: dict) -> None:
    """Place the camera: nadir at zero off-nadir, tilted toward the heading, heading clockwise."""
    obj.data.angle = math.radians(pose["fov_deg"])
    obj.location = (pose["x"], pose["y"], pose["z"])
    # Rotating about X tilts the view from straight down toward +Y (north); then about Z by
    # minus the heading turns that direction clockwise onto the compass heading.
    obj.rotation_euler = (
        math.radians(pose["off_nadir_deg"]),
        0.0,
        -math.radians(pose["heading_deg"]),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scene_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--px", type=int, default=0)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
    scene = json.loads((a.scene_dir / "scene.json").read_text())
    ground = np.load(a.scene_dir / "ground.npy")
    px = a.px or scene["px"]

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    build_ground(scene, ground)
    n = build_buildings(scene)
    nt = build_trees(scene)
    build_light(scene)
    camera_file = a.scene_dir / "camera.json"
    frames = json.loads(camera_file.read_text())["frames"] if camera_file.exists() else []
    if frames:
        cam_obj = build_perspective_camera(px)
    else:
        build_camera(scene, px)
    cycles = bpy.context.scene
    cycles.render.engine = "CYCLES"
    cycles.cycles.device = "CPU"
    cycles.cycles.samples = a.samples
    cycles.cycles.use_denoising = False
    if a.threads:
        cycles.render.threads_mode = "FIXED"
        cycles.render.threads = a.threads
    cycles.view_settings.view_transform = (
        "Raw"  # no tone curve: 8-bit values compare with the 2.5D render
    )
    cycles.render.image_settings.file_format = "PNG"
    cycles.render.image_settings.color_mode = "RGB"
    t1 = time.time()
    say(f"scene_built buildings={n} trees={nt} seconds={t1 - t0:.1f}")
    if frames:  # OUT is a directory of frame_0000.png, frame_0001.png, ...
        a.out.mkdir(parents=True, exist_ok=True)
        for pose in frames:
            pose_camera(cam_obj, pose)
            cycles.render.filepath = str((a.out / f"frame_{pose['index']:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
            say(f"frame {pose['index'] + 1}/{len(frames)} t={time.time() - t1:.1f}s")
    else:
        cycles.render.filepath = str(a.out.resolve())
        bpy.ops.render.render(write_still=True)
    t2 = time.time()
    say(f"rendered px={px} frames={max(1, len(frames))} samples={a.samples} seconds={t2 - t1:.1f}")
    print(
        json.dumps(
            {
                "build_s": round(t1 - t0, 2),
                "render_s": round(t2 - t1, 2),
                "px": px,
                "frames": len(frames),
                "samples": a.samples,
                "bpy": bpy.app.version_string,
                "buildings": n,
                "trees": nt,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
