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


def foliage_material(name: str, rgb: tuple[float, float, float], rough: float, seed: int):
    """Leaf clumps: fine, high-contrast noise from sunlit to shadowed green, bumped the same way."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 14.0 + seed % 5
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.65
    stretch = nodes.new("ShaderNodeMapRange")  # clumps rather than a smooth gradient
    stretch.inputs["From Min"].default_value = 0.35
    stretch.inputs["From Max"].default_value = 0.65
    links.new(noise.outputs["Fac"], stretch.inputs["Value"])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*(min(c * 1.25, 1.0) for c in rgb), 1.0)
    mix.inputs[7].default_value = (*(c * 0.45 for c in rgb), 1.0)
    links.new(stretch.outputs["Result"], mix.inputs[0])
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.8
    bump.inputs["Distance"].default_value = 0.15
    links.new(stretch.outputs["Result"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
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


TERRAIN: dict = {}  # "nodes" (n + 1, n + 1) heights, "cell" metres; empty on a flat tile


def terrain_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ground height (bilinear) at tile coordinates; zero when the tile is flat."""
    if not TERRAIN:
        return np.zeros(np.shape(x))
    nodes, cell = TERRAIN["nodes"], TERRAIN["cell"]
    n = nodes.shape[0] - 1
    fx = np.clip(np.asarray(x, dtype="float64") / cell, 0.0, n - 1e-9)
    fy = np.clip(np.asarray(y, dtype="float64") / cell, 0.0, n - 1e-9)
    i, j = fx.astype(int), fy.astype(int)
    wx, wy = fx - i, fy - j
    return (
        nodes[j, i] * (1 - wx) * (1 - wy) + nodes[j, i + 1] * wx * (1 - wy)
        + nodes[j + 1, i] * (1 - wx) * wy + nodes[j + 1, i + 1] * wx * wy
    )  # fmt: skip


def lift_to_terrain(obj) -> None:
    """Raise every vertex of ``obj`` by the ground height under it."""
    if not TERRAIN or obj is None:
        return
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype="float32")
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co[:, 2] += terrain_z(co[:, 0], co[:, 1])
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.update()


def displaced_plane(size: float, nodes: np.ndarray):
    """The ground as a grid of quads, one vertex per height node, with 0..1 UVs for the texture."""
    n = nodes.shape[0] - 1
    xs = np.linspace(0.0, size, n + 1)
    gx, gy = np.meshgrid(xs, xs)  # gy varies down the rows: row j is y = j * cell
    co = np.stack([gx, gy, nodes], axis=-1).reshape(-1, 3).astype("float32")
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    a = (j * (n + 1) + i).ravel()
    quads = np.stack([a, a + 1, a + n + 2, a + n + 1], axis=1).astype("int32")
    mesh = bpy.data.meshes.new("ground")
    mesh.vertices.add(len(co))
    mesh.vertices.foreach_set("co", co.ravel())
    mesh.loops.add(len(quads) * 4)
    mesh.loops.foreach_set("vertex_index", quads.ravel())
    mesh.polygons.add(len(quads))
    mesh.polygons.foreach_set("loop_start", np.arange(len(quads), dtype="int32") * 4)
    mesh.polygons.foreach_set("loop_total", np.full(len(quads), 4, dtype="int32"))
    uv = mesh.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", (co[quads.ravel(), :2] / size).astype("float32").ravel())
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("ground", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


GROUND_DETAIL = "gen-soil-red"  # a generated earth texture (FRK-126), fine grain on every ground
GROUND_DETAIL_SCALE_M = 2.0
GROUND_DETAIL_AMOUNT = 0.35  # how far the colour follows the texture's light and dark
GROUND_DETAIL_BUMP = 0.5


def add_ground_detail(mat, bsdf, base_out, assets: Path) -> None:
    """Fine grain on the ground: the colour is multiplied by the detail texture's lightness around
    its mean, and its derived normal map tilts the surface. Skipped if the assets are missing."""
    colour, normal = (assets / "textures" / f"{GROUND_DETAIL}{s}.jpg" for s in ("", "-normal"))
    if not (colour.exists() and normal.exists()):
        return
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    where = nodes.new("ShaderNodeNewGeometry")  # world position: the ground plane is scaled
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1 / GROUND_DETAIL_SCALE_M,) * 3
    links.new(where.outputs["Position"], mapping.inputs["Vector"])

    def image(path: Path):
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(str(path))
        node.image.colorspace_settings.name = "Non-Color"
        node.extension = "REPEAT"
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    tex = image(colour)
    mean = float(np.asarray(tex.image.pixels[:]).reshape(-1, 4)[:, :3].mean())
    grey = nodes.new("ShaderNodeRGBToBW")
    links.new(tex.outputs["Color"], grey.inputs["Color"])
    relative = nodes.new("ShaderNodeMath")  # lightness over its mean, minus one
    relative.operation = "DIVIDE"
    relative.inputs[1].default_value = max(mean, 1e-3)
    links.new(grey.outputs["Val"], relative.inputs[0])
    gain = nodes.new("ShaderNodeMath")  # 1 + amount * (relative - 1)
    gain.operation = "MULTIPLY_ADD"
    gain.inputs[1].default_value = GROUND_DETAIL_AMOUNT
    gain.inputs[2].default_value = 1.0 - GROUND_DETAIL_AMOUNT
    links.new(relative.outputs["Value"], gain.inputs[0])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    links.new(base_out, mix.inputs[6])
    links.new(gain.outputs["Value"], mix.inputs[7])
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    tilt = nodes.new("ShaderNodeNormalMap")
    tilt.inputs["Strength"].default_value = GROUND_DETAIL_BUMP
    links.new(image(normal).outputs["Color"], tilt.inputs["Color"])
    links.new(tilt.outputs["Normal"], bsdf.inputs["Normal"])


def build_ground(
    scene: dict, ground: np.ndarray, rough: np.ndarray | None = None, assets: Path | None = None
):
    size = scene["size_m"]
    h, w, _ = ground.shape
    path = str(Path(tempfile.gettempdir()) / "fraktal_ground.png")
    write_png(path, (np.clip(ground, 0.0, 1.0) * 255 + 0.5).astype("uint8"))
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"  # the colours are already display reflectance
    if TERRAIN:
        plane = displaced_plane(size, TERRAIN["nodes"])
    else:
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
    if assets is not None:
        add_ground_detail(mat, bsdf, tex.outputs["Color"], assets)
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


def load_material(name: str, assets: Path, tint: list[float] | None = None):
    """A Blender material from ``assets/materials/<name>.json``: a tinted texture or a colour."""
    spec = json.loads((assets / "materials" / f"{name}.json").read_text())
    if tint is not None:
        spec["tint"] = tint
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

    def data_map(key: str):  # a normal or roughness map: raw numbers, same world-metre mapping
        image = bpy.data.images.load(str(assets / spec[key]))
        image.colorspace_settings.name = "Non-Color"
        node = nodes.new("ShaderNodeTexImage")
        node.image = image
        node.projection = "BOX"
        node.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    normal_out = None
    if "normal" in spec:  # derived relief (FRK-126, fraktal.pbr), OpenGL convention
        normal_map = nodes.new("ShaderNodeNormalMap")
        links.new(data_map("normal").outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        normal_out = normal_map.outputs["Normal"]
    if "roughness_map" in spec:
        links.new(data_map("roughness_map").outputs["Color"], bsdf.inputs["Roughness"])
    if "ribs_m" in spec:  # corrugated metal: ribs, rust and stains (FRK-109), all procedural
        add_corrugation(mat, spec, coords, bsdf, tint, normal_out)
    return mat


# A Wave texture in BANDS mode computes x * 20 * Scale and repeats every 2 pi of that,
# so one band is 2 pi / 20 = 0.314 m at Scale 1.
WAVE_BAND_PERIOD_M = 2 * math.pi / 20.0


def add_corrugation(mat, spec: dict, coords, bsdf, tint, normal_in=None) -> None:
    """Ribs as a bump, plus rust patches and a worn tone.

    Ribs run down a sloped face; a flat face gets a random direction per ~14 m cell.

    The rib pitch is real (about 0.1 m), so ribs show at drone scale and wash out at 0.5 m GSD.
    """
    nodes, links = mat.node_tree.nodes, mat.node_tree.links

    def node(kind: str, **inputs):
        n = nodes.new(kind)
        for key, value in inputs.items():
            n.inputs[key].default_value = value
        return n

    def wave(axis: str, vector=None):
        w = nodes.new("ShaderNodeTexWave")
        w.wave_type, w.bands_direction, w.wave_profile = "BANDS", axis, "SIN"
        w.inputs["Scale"].default_value = WAVE_BAND_PERIOD_M / spec["ribs_m"]
        w.inputs["Distortion"].default_value = 0.0
        links.new(vector or coords.outputs["Object"], w.inputs["Vector"])
        return w

    wave_x, wave_y = wave("X"), wave("Y")
    cell = node("ShaderNodeTexVoronoi", Scale=1.0 / 14.0)
    links.new(coords.outputs["Object"], cell.inputs["Vector"])
    sep = nodes.new("ShaderNodeSeparateColor")
    links.new(cell.outputs["Color"], sep.inputs["Color"])
    pick = node("ShaderNodeMath")
    pick.operation = "GREATER_THAN"
    pick.inputs[1].default_value = 0.5
    links.new(sep.outputs["Red"], pick.inputs[0])
    ribs = nodes.new("ShaderNodeMix")
    ribs.data_type = "FLOAT"
    links.new(pick.outputs["Value"], ribs.inputs[0])
    links.new(wave_x.outputs["Fac"], ribs.inputs[2])
    links.new(wave_y.outputs["Fac"], ribs.inputs[3])
    # On a sloped face the ribs run down the slope: the band coordinate is the position across
    # the horizontal part of the face normal. A flat face keeps the random direction above.
    normal = nodes.new("ShaderNodeNewGeometry")
    sep_n = nodes.new("ShaderNodeSeparateXYZ")
    links.new(normal.outputs["Normal"], sep_n.inputs["Vector"])
    flat_n = nodes.new("ShaderNodeCombineXYZ")  # the normal with z dropped
    links.new(sep_n.outputs["X"], flat_n.inputs["X"])
    links.new(sep_n.outputs["Y"], flat_n.inputs["Y"])
    heading = nodes.new("ShaderNodeVectorMath")
    heading.operation = "NORMALIZE"
    links.new(flat_n.outputs["Vector"], heading.inputs[0])
    sep_h = nodes.new("ShaderNodeSeparateXYZ")
    links.new(heading.outputs["Vector"], sep_h.inputs["Vector"])
    minus_y = node("ShaderNodeMath")
    minus_y.operation = "MULTIPLY"
    minus_y.inputs[1].default_value = -1.0
    links.new(sep_h.outputs["Y"], minus_y.inputs[0])
    across = nodes.new("ShaderNodeCombineXYZ")  # perpendicular to the heading
    links.new(minus_y.outputs["Value"], across.inputs["X"])
    links.new(sep_h.outputs["X"], across.inputs["Y"])
    along = nodes.new("ShaderNodeVectorMath")
    along.operation = "DOT_PRODUCT"
    links.new(coords.outputs["Object"], along.inputs[0])
    links.new(across.outputs["Vector"], along.inputs[1])
    band_pos = nodes.new("ShaderNodeCombineXYZ")
    links.new(along.outputs["Value"], band_pos.inputs["X"])
    wave_slope = wave("X", band_pos.outputs["Vector"])
    steep = nodes.new("ShaderNodeVectorMath")
    steep.operation = "LENGTH"
    links.new(flat_n.outputs["Vector"], steep.inputs[0])
    is_sloped = node("ShaderNodeMath")
    is_sloped.operation = "GREATER_THAN"
    is_sloped.inputs[1].default_value = 0.1
    links.new(steep.outputs["Value"], is_sloped.inputs[0])
    ribs_final = nodes.new("ShaderNodeMix")
    ribs_final.data_type = "FLOAT"
    links.new(is_sloped.outputs["Value"], ribs_final.inputs[0])
    links.new(ribs.outputs[0], ribs_final.inputs[2])
    links.new(wave_slope.outputs["Fac"], ribs_final.inputs[3])
    bump = node(
        "ShaderNodeBump",
        Strength=spec.get("rib_strength", 0.6),
        Distance=spec.get("rib_depth_m", 0.02),
    )
    links.new(ribs_final.outputs[0], bump.inputs["Height"])
    if normal_in is not None:  # the ribs ride on top of the derived relief
        links.new(normal_in, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # rust and wear: noise patches blend the tinted texture toward a rust brown
    patches = node("ShaderNodeTexNoise", Scale=1.0 / 2.5, Detail=6.0, Roughness=0.6)
    links.new(coords.outputs["Object"], patches.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    lo = 1.0 - spec.get("rust", 0.25)
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = (
        lo,
        min(lo + 0.12, 1.0),
    )
    links.new(patches.outputs["Fac"], ramp.inputs["Fac"])
    rusted = nodes.new("ShaderNodeMix")
    rusted.data_type = "RGBA"
    rusted.inputs[7].default_value = (0.42, 0.2, 0.1, 1.0)
    links.new(
        ramp.outputs["Color"], rusted.inputs[0]
    )  # the ramp is black-to-white: its value is the factor
    links.new(tint.outputs[2], rusted.inputs[6])
    links.new(rusted.outputs[2], bsdf.inputs["Base Color"])


def build_buildings(scene: dict, scene_dir: Path, assets: Path) -> int:
    """One mesh for every building, read from the exported ``mesh_*.npy`` arrays."""
    if not scene["buildings"]:  # an estate or paddy tile: empty arrays load as floats, no mesh
        return 0
    load = lambda key: np.load(scene_dir / f"mesh_{key}.npy")  # noqa: E731
    co, sizes, face_verts, face_mat = (
        load("vertices"), load("face_sizes"), load("face_verts"), load("face_material")
    )  # fmt: skip
    if TERRAIN:  # each building stands on the lowest ground under its footprint
        owner_face = np.load(scene_dir / "mesh_face_building.npy").astype("int64")
        owner = np.zeros(len(co), dtype="int64")
        owner[face_verts] = np.repeat(owner_face, sizes)
        ground = terrain_z(co[:, 0], co[:, 1])
        base = np.full(owner.max() + 1, np.inf)
        np.minimum.at(base, owner, ground)
        co = co.copy()
        co[:, 2] += base[owner]
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


CLASS_GROUND, CLASS_BUILDING, CLASS_TREE, CLASS_VEHICLE, CLASS_POLE = 1, 2, 3, 4, 5


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
    for name, a in AGENTS.items():  # moving agents: stable instances after any static ones
        per = len(a["boxes"]) * 6
        owner = np.repeat(np.arange(a["n"]) + a["first_instance"], per)
        paint(bpy.data.objects[name], np.full(len(owner), a["class"]), owner + 1)
    if "props" in bpy.data.objects and PROP_FACE_OWNER:
        owner = np.asarray(PROP_FACE_OWNER, dtype="int64")
        paint(
            bpy.data.objects["props"],
            np.asarray(PROP_FACE_CLASS, dtype="int64"),
            owner + 1 + PROP_INSTANCE_BASE,
        )
    if "cables" in bpy.data.objects:  # wires belong to the pole class
        n = len(bpy.data.objects["cables"].data.polygons)
        paint(bpy.data.objects["cables"], np.full(n, CLASS_POLE), np.zeros(n, int))
    if "markings" in bpy.data.objects:  # paint and kerbs are part of the ground
        n = len(bpy.data.objects["markings"].data.polygons)
        paint(bpy.data.objects["markings"], np.full(n, CLASS_GROUND), np.zeros(n, int))
    if "walls" in bpy.data.objects and WALL_FACE_OWNER:
        owner = np.asarray(WALL_FACE_OWNER, dtype="int64")
        paint(bpy.data.objects["walls"], np.full(len(owner), CLASS_WALL), owner + 1)
    for name, klass, faces in (
        ("vehicles", CLASS_VEHICLE, VEHICLE_OWNER),
        ("poles", CLASS_POLE, POLE_OWNER),
    ):
        if name in bpy.data.objects:
            owner = np.asarray(faces, dtype="int64")
            paint(bpy.data.objects[name], np.full(len(owner), klass), owner + 1)


# Every static object the label and depth passes must re-material. One left out renders lit with
# its own material and smears its labels and depth (cables and markings did). Agents: see AGENTS.
LOOK_OBJECTS = (
    "ground", "buildings", "trees", "vehicles", "poles", "props", "walls", "cables", "markings",
)  # fmt: skip


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
    for name in (*LOOK_OBJECTS, *AGENTS):
        if name in bpy.data.objects:
            mesh = bpy.data.objects[name].data
            mesh.materials.clear()
            mesh.materials.append(mat)
            mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), dtype="int32"))
            mesh.update()
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    # Every face is emissive here, so with the light tree Cycles rebuilt a light structure over
    # millions of triangles each time the agents moved: 3.5 s a frame instead of 0.4 s. Nothing
    # is lit in this pass, so the values do not change.
    sc.cycles.use_light_tree = False
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


def use_depth_look(sc, unit_m: float) -> None:
    """Unlit rendering whose colour is the camera z depth in ``unit_m`` steps, split over R and G.

    R is the high byte and G the low byte of ``round(depth / unit_m)``; the background stays black
    (zero, no surface). Like the label look it is one unfiltered sample, so values are exact.
    """
    mat = bpy.data.materials.new("depth")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()

    def math_node(op: str, a, b=None):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, float):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    z = nodes.new("ShaderNodeCameraData").outputs["View Z Depth"]
    units = math_node("ROUND", math_node("DIVIDE", z, unit_m))
    high = math_node("FLOOR", math_node("DIVIDE", units, 256.0))
    low = math_node("SUBTRACT", units, math_node("MULTIPLY", high, 256.0))
    combine = nodes.new("ShaderNodeCombineColor")
    links.new(math_node("DIVIDE", high, 255.0), combine.inputs["Red"])
    links.new(math_node("DIVIDE", low, 255.0), combine.inputs["Green"])
    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(combine.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    for name in (*LOOK_OBJECTS, *AGENTS):
        if name in bpy.data.objects:
            mesh = bpy.data.objects[name].data
            mesh.materials.clear()
            mesh.materials.append(mat)
            mesh.polygons.foreach_set("material_index", np.zeros(len(mesh.polygons), dtype="int32"))
            mesh.update()
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 0.01
    sc.render.film_transparent = False
    sc.render.dither_intensity = 0.0
    world = bpy.data.worlds.new("black-depth")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    sc.world = world
    sc.view_settings.view_transform = "Raw"
    sc.view_settings.look = "None"


FOLIAGE = [
    (0.24, 0.30, 0.17),
    (0.28, 0.33, 0.20),
    (0.21, 0.27, 0.16),
    (0.30, 0.33, 0.21),
    (0.25, 0.29, 0.19),
    (0.33, 0.35, 0.24),
]  # fmt: skip  (olive and grey-green, set by eye against the dry-land reference chips)
CANOPY = [
    (0.13, 0.32, 0.07),
    (0.17, 0.36, 0.09),
    (0.11, 0.29, 0.09),
    (0.20, 0.38, 0.11),
]  # fmt: skip  (lush mid-greens: the wet-tropic street trees in the reference photos)
CANOPY_DRY = [
    (0.27, 0.31, 0.12),
    (0.31, 0.34, 0.15),
    (0.24, 0.29, 0.12),
]  # fmt: skip  (dusty olive: neem and acacia in a dry city)
CROWN_LUMP = 0.30  # radial bumpiness of a crown, so it is not a smooth ball


TREE_OWNER: list[int] = []  # the tree (index into scene["trees"]) each tree-mesh face belongs to


def _icosphere():
    template = bmesh.new()
    bmesh.ops.create_icosphere(template, subdivisions=3, radius=1.0)
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
    mats = [foliage_material(f"leaf{i}", c, 0.8, 20 + i) for i, c in enumerate(FOLIAGE)]
    mats.append(material("bark", (0.24, 0.17, 0.11), 0.9, 40))
    mats.append(foliage_material("frond", (0.2, 0.4, 0.14), 0.7, 41))
    mats.append(foliage_material("neem", (0.09, 0.26, 0.1), 0.8, 42))
    mats.append(foliage_material("acacia", (0.33, 0.38, 0.14), 0.8, 43))
    for i, c in enumerate(CANOPY):  # street trees of the wet-tropic and highland cities
        mats.append(foliage_material(f"canopy{i}", c, 0.8, 44 + i))
    for i, c in enumerate(CANOPY_DRY):
        mats.append(foliage_material(f"canopy-dry{i}", c, 0.8, 50 + i))
    bark, frond = len(FOLIAGE), len(FOLIAGE) + 1
    neem_mat, acacia_mat = len(FOLIAGE) + 2, len(FOLIAGE) + 3
    canopy_mat = len(FOLIAGE) + 4
    bm = bmesh.new()

    def blob(k, x, y, zc, rx, rz, mat):
        rng = random.Random(k * 7919 + 13)  # the same lumps for the same tree, every run
        phase = [rng.uniform(0, 2 * math.pi) for _ in range(6)]
        ring = []
        for v in verts:
            bump = 1.0 + CROWN_LUMP * (
                0.5 * math.sin(3.0 * v.x + phase[0]) * math.cos(3.0 * v.y + phase[1])
                + 0.3 * math.sin(5.0 * v.z + phase[2] + 4.0 * v.x)
                + 0.2 * math.sin(9.0 * v.x + 7.0 * v.y + phase[3])
            )
            ring.append(
                bm.verts.new((x + v.x * rx * bump, y + v.y * rx * bump, zc + v.z * rz * bump))
            )
        for f in faces:
            face = bm.faces.new([ring[i] for i in f])
            face.material_index = mat
            face.smooth = True  # leaf clumps come from the material, not from facets
            TREE_OWNER.append(k)

    for k, tree in enumerate(trees):
        x, y, radius, height = tree[:4]
        species = tree[4] if len(tree) > 4 else "broadleaf"
        if species in ("palm", "date_palm"):
            date = species == "date_palm"  # taller, a thicker trunk and a fuller head
            top = height * (0.94 if date else 0.9)
            _prism(bm, k, x, y, max(0.14, radius * 0.06) * (1.5 if date else 1.0), 0.0, top, bark)
            length = max(radius, 2.2 if date else 1.8)
            fronds = 12 if date else 8
            for j in range(fronds):
                a = 2 * math.pi * (j + 0.3 * (k % 3)) / fronds
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
        elif species in (
            "canopy",
            "canopy_dry",
        ):  # a tall trunk under a wide, flat-topped crown of lobes
            top = height * 0.55
            _prism(bm, k, x, y, max(0.18, radius * 0.06), 0.0, top, bark)
            rz = max(1.6, (height - top) * 0.55)
            rng = random.Random(k * 104729 + 7)
            dry = species == "canopy_dry"
            mat = canopy_mat + (len(CANOPY) + k % len(CANOPY_DRY) if dry else k % len(CANOPY))
            blob(k, x, y, top + rz * 0.9, radius * 0.75, rz, mat)
            for j in range(7):  # lobes spread the crown out sideways
                a, d = rng.uniform(0, 2 * math.pi), radius * rng.uniform(0.35, 0.75)
                blob(k * 8 + j + 1, x + d * math.cos(a), y + d * math.sin(a),
                     top + rz * rng.uniform(0.55, 0.85), radius * rng.uniform(0.3, 0.5),
                     rz * 0.7, mat)  # fmt: skip
        elif species == "acacia":  # a thin trunk and a flat, wide umbrella of leaves
            top = max(height * 0.8, 2.0)
            _prism(bm, k, x, y, max(0.1, radius * 0.05), 0.0, top, bark)
            blob(k, x, y, top, radius * 1.3, max(0.35, radius * 0.28), acacia_mat)
        elif species == "neem":  # a short trunk under a broad dense dome
            half = max(radius * 1.0, height * 0.35)
            zc = max(half, height - half)
            if zc - half > 0.3:
                _prism(bm, k, x, y, max(0.15, radius * 0.08), 0.0, zc - half * 0.5, bark)
            blob(k, x, y, zc, radius * 1.1, half, neem_mat)
        elif species == "shrub":
            rz = max(0.4, min(height, radius * 1.4) * 0.5)
            blob(k, x, y, rz * 0.8, radius * 1.15, rz, (k * 5 + 1) % len(FOLIAGE))
        else:
            half = max(radius * 0.9, height * 0.3)  # crown half-height; the crown sits above 0.4 h
            zc = max(half, height - half)
            if zc - half > 0.4:
                _prism(bm, k, x, y, max(0.12, radius * 0.07), 0.0, zc - half * 0.5, bark)
            blob(k, x, y, zc, radius, half, (k * 5 + 1) % len(FOLIAGE))
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


VEHICLE_OWNER: list[int] = []  # the vehicle each face of the vehicle mesh belongs to
PROP_INSTANCE_BASE = (
    20000  # props number from here, so a tuk-tuk never shares an id with a parked car
)
PROP_FACE_OWNER: list[int] = []  # the prop each face of the props mesh belongs to
PROP_FACE_CLASS: list[int] = []  # and the class of that prop (from the catalogue)
WALL_FACE_OWNER: list[int] = []  # the compound each face of the wall mesh belongs to
CLASS_WALL = 8
POLE_OWNER: list[int] = []
CAR_BODY = [(235, 235, 235), (40, 40, 45), (150, 30, 30), (170, 175, 180), (30, 60, 120)]


def _box(bm, owners: list[int], k: int, cx, cy, ux, uy, half_l, half_w, z0, z1, mat: int) -> None:
    """A box centred at (cx, cy), its long axis along the unit vector (ux, uy)."""
    vx, vy = -uy, ux
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    xy = [
        (cx + ux * a * half_l + vx * b * half_w, cy + uy * a * half_l + vy * b * half_w)
        for a, b in corners
    ]
    bottom = [bm.verts.new((x, y, z0)) for x, y in xy]
    top = [bm.verts.new((x, y, z1)) for x, y in xy]
    sides = [(bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]) for i in range(4)]
    for quad in (*sides, tuple(top), tuple(reversed(bottom))):
        bm.faces.new(quad).material_index = mat
        owners.append(k)


def build_vehicles(scene: dict) -> int:
    """One mesh of boxy cars: a body and a glass cabin, headed along the road."""
    cars = scene.get("vehicles", [])
    VEHICLE_OWNER.clear()
    if not cars:
        return 0
    mats = [
        material(f"car{i}", tuple(c / 255 for c in rgb), 0.35, 60 + i)
        for i, rgb in enumerate(CAR_BODY)
    ]
    mats.append(material("car-glass", (0.05, 0.07, 0.1), 0.1, 70))
    glass = len(mats) - 1
    bm = bmesh.new()
    for k, (x, y, heading, colour) in enumerate(cars):
        ux, uy = math.cos(heading), math.sin(heading)
        _box(bm, VEHICLE_OWNER, k, x, y, ux, uy, 2.25, 0.95, 0.25, 1.0, int(colour))
        _box(bm, VEHICLE_OWNER, k, x - ux * 0.2, y - uy * 0.2, ux, uy, 1.2, 0.85, 1.0, 1.5, glass)
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("vehicles")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("vehicles", mesh)
    bpy.context.collection.objects.link(obj)
    for m in mats:
        mesh.materials.append(m)
    return len(cars)


def build_packed(
    name: str,
    scene_dir: Path,
    prefix: str,
    materials: list,
    textured: dict | None = None,
    assets: Path | None = None,
) -> int:
    """One mesh from packed ``<prefix>_*.npy`` arrays (props, compound walls), with a Principled
    material per colour. Returns the number of faces."""
    verts = np.load(scene_dir / f"{prefix}_vertices.npy")
    faces = np.load(scene_dir / f"{prefix}_faces.npy")
    face_mat = np.load(scene_dir / f"{prefix}_face_material.npy")
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    for mat_name, rgb, rough, metal in materials:
        if textured and assets is not None and mat_name in textured:
            base, tint = textured[mat_name]  # plaster or concrete in this wall's colour
            mesh.materials.append(load_material(base, assets, tint))
            continue
        mat = bpy.data.materials.new(f"{name}-{mat_name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
        mesh.materials.append(mat)
    mesh.polygons.foreach_set("material_index", face_mat.astype("int32"))
    mesh.update()
    bpy.context.collection.objects.link(bpy.data.objects.new(name, mesh))
    return len(faces)


def build_props(scene: dict, scene_dir: Path) -> int:
    """One mesh for every prop (FRK-124), from the packed ``props_*.npy`` arrays.

    The geometry is generated in ``fraktal.props_scene`` and only read here. Each face keeps its
    prop's class and instance for the label pass; the cost of the import is printed per 1000 props.
    """
    props = scene.get("props", [])
    PROP_FACE_OWNER.clear()
    PROP_FACE_CLASS.clear()
    if not props:
        return 0
    if PROP_INSTANCE_BASE + len(props) + 1 > 65535:
        raise SystemExit(f"{len(props)} props do not fit the 16-bit instance label")
    t = time.time()
    n_faces = build_packed("props", scene_dir, "props", scene["prop_materials"])
    owner = np.load(scene_dir / "props_face_owner.npy").astype("int64")
    PROP_FACE_OWNER.extend(owner.tolist())
    classes = np.array([p["class"] for p in props], dtype="int64")
    PROP_FACE_CLASS.extend(classes[owner].tolist())
    seconds = time.time() - t
    say(
        f"props_imported n={len(props)} faces={n_faces} seconds={seconds:.3f} "
        f"per_1000={seconds / len(props) * 1000:.2f}s"
    )
    return len(props)


def build_walls(scene: dict, scene_dir: Path, assets: Path | None = None) -> int:
    """One mesh for every compound wall (FRK-106): class ``wall``, one instance per compound."""
    walls = scene.get("walls", [])
    WALL_FACE_OWNER.clear()
    if not walls:
        return 0
    build_packed(
        "walls", scene_dir, "walls", scene["wall_materials"], scene.get("wall_textures"), assets
    )
    WALL_FACE_OWNER.extend(np.load(scene_dir / "walls_face_owner.npy").astype("int64").tolist())
    return len(walls)


def build_cables(scene: dict, scene_dir: Path) -> int:
    """One mesh of overhead wires, from the packed ``cables_*.npy`` arrays (FRK-147)."""
    wires = scene.get("cables") or {}
    if not wires.get("wires"):
        return 0
    return build_packed("cables", scene_dir, "cables", wires["materials"])


def build_markings(scene: dict, scene_dir: Path) -> int:
    """Lane lines, crossings, kerbs and pavements (FRK-153): one flat-coloured mesh, labelled as
    ground. Returns the number of parts."""
    marks = scene.get("markings") or {}
    if not marks.get("parts"):
        return 0
    build_packed("markings", scene_dir, "markings", marks["materials"])
    return int(marks["parts"])


CLASS_PERSON = 6
CLASS_DRONE = 7
# Moving things are boxes: (forward shift, half length, half width, z0, z1), one per part.
CAR_BOXES = (
    (0.0, 2.2, 0.92, 0.3, 0.95),  # body
    (-0.1, 1.15, 0.84, 0.95, 1.5),  # glazed cabin
    (1.35, 0.33, 0.9, 0.0, 0.66),  # front wheels, one dark block across the axle
    (-1.35, 0.33, 0.9, 0.0, 0.66),  # rear wheels
    (2.25, 0.08, 0.9, 0.35, 0.6),  # front bumper
    (-2.25, 0.08, 0.9, 0.35, 0.6),  # rear bumper
)
PERSON_BOXES = (
    (0.0, 0.13, 0.17, 0.0, 0.85),
    (0.0, 0.12, 0.23, 0.85, 1.45),
    (0.0, 0.11, 0.11, 1.45, 1.7),
)
DRONE_BOXES = (  # a small quadcopter: body and two crossed arms
    (0.0, 0.09, 0.09, -0.04, 0.05),
    (0.0, 0.26, 0.025, 0.0, 0.025),
    (0.0, 0.025, 0.26, 0.0, 0.025),
)
BOX_FACES = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0)]
CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype="float64")
CLOTHES = [
    (0.75, 0.15, 0.15),
    (0.15, 0.3, 0.65),
    (0.85, 0.8, 0.7),
    (0.2, 0.5, 0.25),
    (0.9, 0.6, 0.1),
]
# name -> {"n", "first_instance", "x", "y", "heading", "boxes", "class"}; empty until built
AGENTS: dict[str, dict] = {}


def _box_vertices(x, y, heading, boxes) -> np.ndarray:
    """(n, 8 * len(boxes), 3): the corners of every box of every agent at one frame."""
    ux, uy = np.cos(heading), np.sin(heading)
    out = np.empty((len(x), 8 * len(boxes), 3))
    for b, (shift, half_l, half_w, z0, z1) in enumerate(boxes):
        cx, cy = x + ux * shift, y + uy * shift
        along, across = CORNERS[:, 0] * half_l, CORNERS[:, 1] * half_w
        px = cx[:, None] + ux[:, None] * along - uy[:, None] * across
        py = cy[:, None] + uy[:, None] * along + ux[:, None] * across
        for ring, z in ((0, z0), (4, z1)):
            out[:, 8 * b + ring : 8 * b + ring + 4, 0] = px
            out[:, 8 * b + ring : 8 * b + ring + 4, 1] = py
            out[:, 8 * b + ring : 8 * b + ring + 4, 2] = z
    return out


def build_agents(
    name: str, path: Path, boxes, klass: int, first_instance: int, mats: list, face_mat
) -> int:
    """A mesh of moving agents with fixed topology; ``set_agents`` moves them frame to frame.

    ``face_mat(k, b)`` is the material index of box ``b`` of agent ``k``.
    """
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    n = len(data["ids"])
    if n == 0:
        return 0
    AGENTS[name] = {
        "n": n, "first_instance": first_instance, "boxes": boxes, "class": klass,
        "x": np.array(data["x"]), "y": np.array(data["y"]), "heading": np.array(data["heading"]),
        "z": np.array(data["z"]) if "z" in data else None,
    }  # fmt: skip
    per = 8 * len(boxes)
    faces, material_index = [], []
    for k in range(n):
        for b in range(len(boxes)):
            for f in BOX_FACES:
                faces.append([k * per + b * 8 + v for v in f])
                material_index.append(face_mat(k, b))
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(n * per)
    flat = np.array(faces, dtype="int32")
    mesh.loops.add(flat.size)
    mesh.loops.foreach_set("vertex_index", flat.ravel())
    mesh.polygons.add(len(flat))
    mesh.polygons.foreach_set("loop_start", np.arange(len(flat), dtype="int32") * 4)
    mesh.polygons.foreach_set("loop_total", np.full(len(flat), 4, dtype="int32"))
    mesh.polygons.foreach_set("material_index", np.array(material_index, dtype="int32"))
    mesh.update(calc_edges=True)
    for m in mats:
        mesh.materials.append(m)
    bpy.context.collection.objects.link(bpy.data.objects.new(name, mesh))
    set_agents(0)
    return n


def build_traffic(scene_dir: Path, first_instance: int) -> int:
    mats = [
        material(f"tcar{i}", tuple(c / 255 for c in rgb), 0.35, 90 + i)
        for i, rgb in enumerate(CAR_BODY)
    ]
    mats.append(material("tcar-glass", (0.05, 0.07, 0.1), 0.1, 100))
    mats.append(material("tcar-rubber", (0.03, 0.03, 0.03), 0.9, 101))
    glass, rubber = len(mats) - 2, len(mats) - 1

    def part_material(
        k: int, b: int
    ) -> int:  # box 0 is the body, 1 the cabin glass, the rest tyres
        return (k * 3 + 1) % len(CAR_BODY) if b == 0 else glass if b == 1 else rubber

    return build_agents(
        "traffic",
        scene_dir / "agents.json",
        CAR_BOXES,
        CLASS_VEHICLE,
        first_instance,
        mats,
        part_material,
    )


def build_drones(scene_dir: Path) -> int:
    path = scene_dir / "drones.json"
    scale = float(json.loads(path.read_text()).get("scale", 1.0)) if path.exists() else 1.0
    boxes = tuple((s * scale, hl * scale, hw * scale, z0 * scale, z1 * scale)
                  for s, hl, hw, z0, z1 in DRONE_BOXES)  # fmt: skip
    mats = [material("drone-body", (0.1, 0.1, 0.12), 0.5, 130)]
    mats.append(material("drone-arm", (0.75, 0.75, 0.78), 0.5, 131))
    return build_agents(
        "drones", path, boxes, CLASS_DRONE, 0, mats,
        lambda k, b: 0 if b == 0 else 1,
    )  # fmt: skip


def build_people(scene_dir: Path) -> int:
    mats = [material(f"shirt{i}", c, 0.9, 110 + i) for i, c in enumerate(CLOTHES)]
    mats.append(material("skin", (0.55, 0.38, 0.28), 0.8, 120))
    mats.append(material("trousers", (0.14, 0.15, 0.2), 0.9, 121))
    skin, trousers = len(mats) - 2, len(mats) - 1
    return build_agents(
        "people", scene_dir / "people.json", PERSON_BOXES, CLASS_PERSON, 0, mats,
        lambda k, b: (trousers, k % len(CLOTHES), skin)[b],
    )  # fmt: skip


def set_agents(i: int) -> None:
    """Put every moving agent where its track says at frame ``i`` (and on the ground there)."""
    for name, a in AGENTS.items():
        co = _box_vertices(a["x"][:, i], a["y"][:, i], a["heading"][:, i], a["boxes"])
        co[:, :, 2] += terrain_z(co[:, :, 0], co[:, :, 1])
        if a["z"] is not None:  # flying agents: height above the ground below them
            co[:, :, 2] += a["z"][:, i][:, None]
        if "hide" in a:  # the agent the camera rides on is put out of sight
            co[a["hide"], :, 2] -= 1000.0
        mesh = bpy.data.objects[name].data
        mesh.vertices.foreach_set("co", co.astype("float32").ravel())
        mesh.update()


def build_poles(scene: dict) -> int:
    """One mesh of slim roadside poles with a lamp head."""
    poles = scene.get("poles", [])
    POLE_OWNER.clear()
    if not poles:
        return 0
    mats = [material("pole", (0.33, 0.33, 0.35), 0.6, 80)]
    bm = bmesh.new()
    for k, (x, y, height) in enumerate(poles):
        _box(bm, POLE_OWNER, k, x, y, 1.0, 0.0, 0.1, 0.1, 0.0, height, 0)
        _box(bm, POLE_OWNER, k, x, y, 1.0, 0.0, 0.45, 0.12, height, height + 0.18, 0)
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new("poles")
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("poles", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mats[0])
    return len(poles)


def build_light(scene: dict, eye_level: bool = False) -> None:
    az, el = scene["sun"]["azimuth_deg"], scene["sun"]["elevation_deg"]
    sky = scene.get("sky") or {}  # scenes exported before FRK-113 have none: the old flat light
    eye = (sky.get("eye_level") or None) if eye_level else None  # FRK-152: street-level exposure
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = sky.get("sun_strength", 2.6)  # about 0.6 of reflectance at 50 degrees elevation
    if eye:
        sun.energy *= eye["sun"]
    sun.color = sky.get("sun_colour", (1.0, 1.0, 1.0))
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
    if not sky:
        bg.inputs["Color"].default_value = (0.62, 0.72, 0.9, 1.0)  # clear-sky fill light
        bg.inputs["Strength"].default_value = 0.55
        bpy.context.scene.world = world
        return
    nodes, links = world.node_tree.nodes, world.node_tree.links

    def node(kind: str, **inputs):
        n = nodes.new(kind)
        for key, value in inputs.items():
            n.inputs[key].default_value = value
        return n

    direction = nodes.new("ShaderNodeTexCoord")  # in a world shader: the view direction
    split = nodes.new("ShaderNodeSeparateXYZ")
    links.new(direction.outputs["Generated"], split.inputs["Vector"])
    up = nodes.new("ShaderNodeMath")  # 0 at the horizon, 1 at the zenith, eased
    up.operation, up.use_clamp = "POWER", True
    up.inputs[1].default_value = 0.6
    clamp = nodes.new("ShaderNodeMath")
    clamp.operation, clamp.use_clamp = "MAXIMUM", True
    clamp.inputs[1].default_value = 0.0
    links.new(split.outputs["Z"], clamp.inputs[0])
    links.new(clamp.outputs["Value"], up.inputs[0])
    gradient = nodes.new("ShaderNodeMixRGB")
    links.new(up.outputs["Value"], gradient.inputs["Fac"])
    gradient.inputs["Color1"].default_value = (*sky["horizon"], 1.0)
    gradient.inputs["Color2"].default_value = (*sky["zenith"], 1.0)
    towards = node("ShaderNodeCombineXYZ", X=toward_sun.x, Y=toward_sun.y, Z=toward_sun.z)
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    links.new(direction.outputs["Generated"], dot.inputs[0])
    links.new(towards.outputs["Vector"], dot.inputs[1])
    lobe = nodes.new("ShaderNodeMath")  # the warm bloom around the sun
    lobe.operation, lobe.use_clamp = "POWER", True
    lobe.inputs[1].default_value = 6.0
    positive = nodes.new("ShaderNodeMath")
    positive.operation, positive.use_clamp = "MAXIMUM", True
    positive.inputs[1].default_value = 0.0
    links.new(dot.outputs["Value"], positive.inputs[0])
    links.new(positive.outputs["Value"], lobe.inputs[0])
    scaled = nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = sky["glow_amount"]
    links.new(lobe.outputs["Value"], scaled.inputs[0])
    glow = nodes.new("ShaderNodeMixRGB")
    glow.blend_type = "ADD"
    links.new(scaled.outputs["Value"], glow.inputs["Fac"])
    links.new(gradient.outputs["Color"], glow.inputs["Color1"])
    glow.inputs["Color2"].default_value = (*sky["glow"], 1.0)
    links.new(glow.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = sky["fill"]
    if eye:  # light the scene at ``fill`` and show the sky to the camera at ``sky_seen``
        ray = nodes.new("ShaderNodeLightPath")
        mix = nodes.new("ShaderNodeMath")  # camera ray: sky_seen, any other ray: fill
        mix.operation = "MULTIPLY_ADD"
        links.new(ray.outputs["Is Camera Ray"], mix.inputs[0])
        mix.inputs[1].default_value = eye["sky_seen"] - eye["fill"]
        mix.inputs[2].default_value = eye["fill"]
        strength = nodes.new("ShaderNodeMath")
        strength.operation = "MULTIPLY"
        links.new(mix.outputs["Value"], strength.inputs[0])
        strength.inputs[1].default_value = sky["fill"]
        links.new(strength.outputs["Value"], bg.inputs["Strength"])
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
    cam.clip_start, cam.clip_end = 0.3, 20000.0
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


def use_gpu(device: str) -> bool:
    """Enable the best GPU backend Cycles offers; fail loudly if there is none, never fall back."""
    if device != "gpu":
        return False
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:  # this build does not offer that backend
            continue
        prefs.get_devices()
        found = [d for d in prefs.devices if d.type == kind]
        if found:
            for d in prefs.devices:
                d.use = d.type == kind
            say(f"device {kind}: " + ", ".join(d.name for d in found))
            return True
    raise SystemExit("--device gpu: Blender found no GPU for Cycles")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scene_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--assets", type=Path, required=True, help="the assets directory")
    p.add_argument("--px", type=int, default=0)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--labels", type=Path, default=None, help="also write label images here")
    p.add_argument("--depth", type=Path, default=None, help="also write encoded depth images here")
    p.add_argument("--depth-unit", type=float, default=0.05, help="metres per depth step")
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:])
    scene = json.loads((a.scene_dir / "scene.json").read_text())
    ground = np.load(a.scene_dir / "ground.npy")
    px = a.px or scene["px"]

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rough_file = a.scene_dir / "ground_rough.npy"
    terrain = scene.get("terrain")
    if terrain:
        TERRAIN["nodes"] = np.load(a.scene_dir / terrain["file"]).astype("float64")
        TERRAIN["cell"] = terrain["cell_m"]
    build_ground(scene, ground, np.load(rough_file) if rough_file.exists() else None, a.assets)
    n = build_buildings(scene, a.scene_dir, a.assets)
    nt = build_trees(scene)
    nv, npoles = build_vehicles(scene), build_poles(scene)
    nprops, nwalls = build_props(scene, a.scene_dir), build_walls(scene, a.scene_dir, a.assets)
    nmarks = build_markings(scene, a.scene_dir)
    ncables = build_cables(scene, a.scene_dir)
    for name in ("trees", "vehicles", "poles", "props", "walls", "markings", "cables"):
        lift_to_terrain(bpy.data.objects.get(name))
    moving = build_traffic(a.scene_dir, len(scene.get("vehicles", [])))
    walking = build_people(a.scene_dir)
    build_drones(a.scene_dir)
    camera_file = a.scene_dir / "camera.json"
    camera_data = json.loads(camera_file.read_text()) if camera_file.exists() else {}
    build_light(scene, bool(camera_data.get("eye_level")))
    frames = camera_data.get("frames", [])
    follow = camera_data.get("follow")
    if follow and follow["agent"] in AGENTS:  # no camera inside its own car
        AGENTS[follow["agent"]]["hide"] = follow["index"]
        set_agents(0)
    if frames:
        cam_obj = build_perspective_camera(px)
    else:
        build_camera(scene, px)
    cycles = bpy.context.scene
    cycles.render.engine = "CYCLES"
    cycles.cycles.device = "GPU" if use_gpu(a.device) else "CPU"
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
    cycles.render.use_persistent_data = True  # keep the BVH and meshes between a strip's frames
    t1 = time.time()
    say(
        f"scene_built buildings={n} trees={nt} vehicles={nv} moving={moving} people={walking} "
        f"poles={npoles} props={nprops} walls={nwalls} marks={nmarks} wires={ncables} "
        f"seconds={t1 - t0:.1f}"
    )
    if frames:  # OUT is a directory of frame_0000.png, frame_0001.png, ...
        a.out.mkdir(parents=True, exist_ok=True)
        for pose in frames:
            pose_camera(cam_obj, pose)
            set_agents(pose["index"])
            cycles.render.filepath = str((a.out / f"frame_{pose['index']:04d}.png").resolve())
            bpy.ops.render.render(write_still=True)
            say(f"frame {pose['index'] + 1}/{len(frames)} t={time.time() - t1:.1f}s")
    else:
        cycles.render.filepath = str(a.out.resolve())
        bpy.ops.render.render(write_still=True)
    if a.labels:
        t2 = time.time()
        paint_labels(scene, a.scene_dir)
        t3 = time.time()
        use_label_look(cycles)
        say(f"labels_prepared paint={t3 - t2:.1f}s look={time.time() - t3:.1f}s")
        t4 = time.time()
        if frames:
            a.labels.mkdir(parents=True, exist_ok=True)
            for pose in frames:
                ta = time.time()
                pose_camera(cam_obj, pose)
                set_agents(pose["index"])
                tb = time.time()
                name = f"frame_{pose['index']:04d}.png"
                cycles.render.filepath = str((a.labels / name).resolve())
                bpy.ops.render.render(write_still=True)
                say(
                    f"label {pose['index'] + 1}/{len(frames)} t={time.time() - t4:.1f}s "
                    f"agents={tb - ta:.2f}s render={time.time() - tb:.2f}s"
                )
        else:
            cycles.render.filepath = str(a.labels.resolve())
            bpy.ops.render.render(write_still=True)
        say("labels_rendered")
    if a.depth:
        use_depth_look(cycles, a.depth_unit)
        if frames:
            a.depth.mkdir(parents=True, exist_ok=True)
            for pose in frames:
                pose_camera(cam_obj, pose)
                set_agents(pose["index"])
                cycles.render.filepath = str((a.depth / f"frame_{pose['index']:04d}.png").resolve())
                bpy.ops.render.render(write_still=True)
        else:
            cycles.render.filepath = str(a.depth.resolve())
            bpy.ops.render.render(write_still=True)
        say("depth_rendered")
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
