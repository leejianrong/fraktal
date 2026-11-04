"""Render one exported tile with Blender Cycles, nadir, orthographic (ADR-0010).

Not imported by fraktal: ``fraktal.render3d`` runs it as a script with a Python that has ``bpy``.

Usage: python blender_scene.py SCENE_DIR OUT.png --assets DIR [--px N] [--samples N] [--threads N]

Needs the ``bpy`` wheel (``pip install bpy``, Python 3.11). Reads ``scene.json`` and ``ground.npy``
written by ``fraktal.blender_export``; builds one ground plane, one extruded mesh for all the
buildings and a sun plus sky; writes an 8-bit PNG and prints timings as ``STAGE`` lines.
Building materials come from the assets directory (owned or CC0, see assets/manifest.json); the
ground and tree materials are procedural, so no other third-party asset enters the render.
"""

from __future__ import annotations

import argparse
import json
import math
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


def build_ground(scene: dict, ground: np.ndarray, rough: np.ndarray | None = None):
    size = scene["size_m"]
    h, w, _ = ground.shape
    path = str(Path(tempfile.gettempdir()) / "fraktal_ground.png")
    write_png(path, (np.clip(ground, 0.0, 1.0) * 255 + 0.5).astype("uint8"))
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"  # the colours are already display reflectance
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(size / 2, size / 2, 0.0))
    plane = bpy.context.object
    plane.name = "ground"
    plane.scale = (size, size, 1.0)
    mat = bpy.data.materials.new("ground")
    mat.use_nodes = True
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"  # smooth class edges; the texture is already in the image
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.95
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if rough is not None:  # water and flooded paddy are glossy, the rest matt
        grey = np.repeat(np.clip(rough, 0.0, 1.0)[..., None], 3, axis=2)
        rpath = str(Path(tempfile.gettempdir()) / "fraktal_ground_rough.png")
        write_png(rpath, (grey * 255 + 0.5).astype("uint8"))
        rimg = bpy.data.images.load(rpath)
        rimg.colorspace_settings.name = "Non-Color"
        rtex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        rtex.image = rimg
        rtex.interpolation = "Closest"
        mat.node_tree.links.new(rtex.outputs["Color"], bsdf.inputs["Roughness"])
    plane.data.materials.append(mat)


def load_material(name: str, assets: Path):
    """A Blender material from ``assets/materials/<name>.json``: a tinted texture or a colour."""
    spec = json.loads((assets / "materials" / f"{name}.json").read_text())
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = spec.get("roughness", 0.8)
    if "image" not in spec:
        bsdf.inputs["Base Color"].default_value = (*spec["color"], 1.0)
        return mat
    img = bpy.data.images.load(str(assets / spec["image"]))
    img.colorspace_settings.name = "Non-Color"  # stored colours are display reflectance
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / spec["scale_m"],) * 3
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.projection = "BOX"  # no UVs: the texture follows world metres on every face
    tex.projection_blend = 0.2
    tint = nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.inputs[0].default_value = 1.0
    tint.inputs[7].default_value = (*spec["tint"], 1.0)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], tint.inputs[6])
    links.new(tint.outputs[2], bsdf.inputs["Base Color"])
    return mat


def build_buildings(scene: dict, scene_dir: Path, assets: Path) -> int:
    """One mesh for every building, read from the exported ``mesh_*.npy`` arrays."""
    load = lambda key: np.load(scene_dir / f"mesh_{key}.npy")  # noqa: E731
    co, sizes, face_verts, face_mat = (
        load("vertices"), load("face_sizes"), load("face_verts"), load("face_material")
    )  # fmt: skip
    mesh = bpy.data.meshes.new("buildings")
    mesh.vertices.add(len(co))
    mesh.vertices.foreach_set("co", co.astype("float32").ravel())
    mesh.loops.add(len(face_verts))
    mesh.loops.foreach_set("vertex_index", face_verts)
    mesh.polygons.add(len(sizes))
    mesh.polygons.foreach_set("loop_start", np.cumsum(sizes) - sizes)
    mesh.polygons.foreach_set("loop_total", sizes)
    mesh.polygons.foreach_set("material_index", face_mat.astype("int32"))
    mesh.update(calc_edges=True)
    for name in scene["materials"]:
        mesh.materials.append(load_material(name, assets))
    obj = bpy.data.objects.new("buildings", mesh)
    bpy.context.collection.objects.link(obj)
    return len(scene["buildings"])


CLASS_GROUND, CLASS_BUILDING, CLASS_TREE = 1, 2, 3


def paint_labels(scene: dict, scene_dir: Path) -> None:
    """Give every face a flat colour: red = class, green and blue = instance + 1 (16-bit)."""

    def paint(obj, classes: np.ndarray, instances: np.ndarray) -> None:
        mesh = obj.data
        sizes = np.array([len(p.vertices) for p in mesh.polygons])
        red, high, low = classes / 255.0, (instances >> 8) / 255.0, (instances & 255) / 255.0
        face = np.stack([red, high, low, np.ones(len(sizes))], axis=1).astype("float32")
        attr = mesh.color_attributes.new("label", "FLOAT_COLOR", "CORNER")
        attr.data.foreach_set("color", np.repeat(face, sizes, axis=0).ravel())
        mesh.color_attributes.active_color = attr

    ground = bpy.data.objects["ground"]
    n_ground = len(ground.data.polygons)
    paint(ground, np.full(n_ground, CLASS_GROUND), np.zeros(n_ground, int))
    if "buildings" in bpy.data.objects:
        obj = bpy.data.objects["buildings"]
        owner = np.load(scene_dir / "mesh_face_building.npy").astype("int64")
        paint(obj, np.full(len(owner), CLASS_BUILDING), owner + 1)
    if "trees" in bpy.data.objects:
        obj = bpy.data.objects["trees"]
        owner = np.asarray(TREE_OWNER, dtype="int64")
        paint(obj, np.full(len(owner), CLASS_TREE), owner + 1)


def use_label_look(sc) -> None:
    """Unlit, unfiltered Cycles rendering that returns the face colours unchanged.

    Workbench would be cheaper but needs a GL context, which a headless pod does not have.
    """
    mat = bpy.data.materials.new("label")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    colour = nodes.new("ShaderNodeVertexColor")
    colour.layer_name = "label"
    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(colour.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    for name in ("ground", "buildings", "trees"):
        if name in bpy.data.objects:
            mesh = bpy.data.objects[name].data
            mesh.materials.clear()
            mesh.materials.append(mat)
            mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), dtype="int32"))
            mesh.update()
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01  # one sample at the pixel centre, no blur
    sc.render.film_transparent = False
    sc.render.dither_intensity = 0.0  # dither would add +-1 noise to the ids
    world = bpy.data.worlds.new("black")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    sc.world = world
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0


FOLIAGE = [(0.15, 0.31, 0.13), (0.19, 0.36, 0.14), (0.12, 0.26, 0.12)]


TREE_OWNER: list[int] = []  # the tree (index into scene["trees"]) each tree-mesh face belongs to


def _icosphere():
    template = bmesh.new()
    bmesh.ops.create_icosphere(template, subdivisions=2, radius=1.0)
    verts = [v.co.copy() for v in template.verts]
    index = {v: i for i, v in enumerate(template.verts)}
    faces = [[index[v] for v in f.verts] for f in template.faces]
    template.free()
    return verts, faces


def _prism(bm, owner: int, x: float, y: float, radius: float, z0: float, z1: float, mat: int):
    ring = [
        (x + radius * math.cos(t), y + radius * math.sin(t))
        for t in np.linspace(0, 2 * math.pi, 6, endpoint=False)
    ]
    for i in range(6):
        a, b = ring[i], ring[(i + 1) % 6]
        face = bm.faces.new([bm.verts.new((*a, z0)), bm.verts.new((*b, z0)),
                             bm.verts.new((*b, z1)), bm.verts.new((*a, z1))])  # fmt: skip
        face.material_index = mat
        TREE_OWNER.append(owner)


def build_trees(scene: dict) -> int:
    """One mesh of trees: round broadleaf crowns on a trunk, palms (trunk and fronds), shrubs."""
    trees = scene.get("trees", [])
    TREE_OWNER.clear()
    if not trees:
        return 0
    verts, faces = _icosphere()
    mats = [material(f"leaf{i}", c, 0.8, 20 + i) for i, c in enumerate(FOLIAGE)]
    mats.append(material("bark", (0.24, 0.17, 0.11), 0.9, 40))
    mats.append(material("frond", (0.2, 0.4, 0.14), 0.7, 41))
    bark, frond = len(FOLIAGE), len(FOLIAGE) + 1
    bm = bmesh.new()

    def blob(k, x, y, zc, rx, rz, mat):
        ring = [bm.verts.new((x + v.x * rx, y + v.y * rx, zc + v.z * rz)) for v in verts]
        for f in faces:
            bm.faces.new([ring[i] for i in f]).material_index = mat
            TREE_OWNER.append(k)

    for k, tree in enumerate(trees):
        x, y, radius, height = tree[:4]
        species = tree[4] if len(tree) > 4 else "broadleaf"
        if species == "palm":
            top = height * 0.9
            _prism(bm, k, x, y, max(0.14, radius * 0.06), 0.0, top, bark)
            length = max(radius, 1.8)
            for j in range(8):
                a = 2 * math.pi * (j + 0.3 * (k % 3)) / 8
                dx, dy = math.cos(a), math.sin(a)
                px, py = -dy * length * 0.22, dx * length * 0.22
                pts = [
                    (x, y, top),
                    (x + dx * length * 0.5 + px, y + dy * length * 0.5 + py, top + 0.3),
                    (x + dx * length, y + dy * length, top - 0.5),
                    (x + dx * length * 0.5 - px, y + dy * length * 0.5 - py, top + 0.3),
                ]  # fmt: skip
                bm.faces.new([bm.verts.new(p) for p in pts]).material_index = frond
                TREE_OWNER.append(k)
        elif species == "shrub":
            rz = max(0.4, min(height, radius * 1.4) * 0.5)
            blob(k, x, y, rz * 0.8, radius * 1.15, rz, k % len(FOLIAGE))
        else:
            half = max(radius * 0.9, height * 0.3)  # crown half-height; the crown sits above 0.4 h
            zc = max(half, height - half)
            if zc - half > 0.4:
                _prism(bm, k, x, y, max(0.12, radius * 0.07), 0.0, zc - half * 0.5, bark)
            blob(k, x, y, zc, radius, half, k % len(FOLIAGE))
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
    p.add_argument("--assets", type=Path, required=True, help="the assets directory")
    p.add_argument("--px", type=int, default=0)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--labels", type=Path, default=None, help="also write label images here")
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
    scene = json.loads((a.scene_dir / "scene.json").read_text())
    ground = np.load(a.scene_dir / "ground.npy")
    px = a.px or scene["px"]

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rough_file = a.scene_dir / "ground_rough.npy"
    build_ground(scene, ground, np.load(rough_file) if rough_file.exists() else None)
    n = build_buildings(scene, a.scene_dir, a.assets)
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
    if a.labels:
        paint_labels(scene, a.scene_dir)
        use_label_look(cycles)
        if frames:
            a.labels.mkdir(parents=True, exist_ok=True)
            for pose in frames:
                pose_camera(cam_obj, pose)
                name = f"frame_{pose['index']:04d}.png"
                cycles.render.filepath = str((a.labels / name).resolve())
                bpy.ops.render.render(write_still=True)
        else:
            cycles.render.filepath = str(a.labels.resolve())
            bpy.ops.render.render(write_still=True)
        say("labels_rendered")
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
