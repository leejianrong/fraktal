"""Mesh clean-up, scale, checks and contact sheets for generated props (FRK-122).

Pure trimesh, numpy and matplotlib, so it runs on a laptop as well as on a pod. A prop leaves here
in metres, z up, standing on z = 0 with its footprint centred on the origin, at most ``MAX_FACES``
triangles.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import trimesh

MAX_FACES = 20000
RENDER_FACES = 3000
MIN_FACES = 200
FLOATER_SHARE = 0.02  # a connected piece with fewer faces than this share of the total is dropped
MAX_PIECES = 8  # more kept pieces than this means the mesh is shattered
PROPORTION_TOLERANCE = 1.7  # each sorted extent may differ from its expected value by this factor
MAX_ASPECT = 25.0  # longest over shortest extent; beyond this a mesh is a plank or a needle


def clean(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, int]:
    """Drop degenerate faces, merge vertices, drop floaters under ``FLOATER_SHARE`` of the faces.

    Parts that are not connected (wheels, a canopy) stay. Returns the mesh, the number of pieces
    before and the number kept.
    """
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    pieces = m.split(only_watertight=False)
    if len(pieces) <= 1:
        return m, max(len(pieces), 1), max(len(pieces), 1)
    total = sum(len(p.faces) for p in pieces)
    kept = [p for p in pieces if len(p.faces) >= FLOATER_SHARE * total]
    return trimesh.util.concatenate(kept), len(pieces), len(kept)


def to_z_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """glTF is y up; our tile frame is z up. Rotate a quarter turn about x."""
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    return m


def place(mesh: trimesh.Trimesh, longest_m: float) -> trimesh.Trimesh:
    """Scale so the longest extent is ``longest_m``, stand on z = 0, centre the footprint."""
    m = mesh.copy()
    m.apply_scale(longest_m / float(max(m.extents)))
    low, high = m.bounds
    m.apply_translation([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
    return m


def decimate(mesh: trimesh.Trimesh, faces: int = MAX_FACES) -> trimesh.Trimesh:
    if len(mesh.faces) <= faces:
        return mesh
    out = mesh.simplify_quadric_decimation(face_count=faces)
    return out


def proportions_off(
    extents: list[float], expected_lwh: tuple[float, float, float] | None
) -> str | None:
    """Compare sorted extents with the class's expected length, width and height.

    The expected box is scaled so its longest side matches the mesh's, since the mesh is placed by
    its longest side. A real vehicle is not a cube, and a "sign" that is a wireframe cube fails.
    """
    if expected_lwh is None:
        return None
    got = sorted(extents, reverse=True)
    want = sorted(expected_lwh, reverse=True)
    want = [w * got[0] / want[0] for w in want]
    worst = max(max(g / w, w / g) for g, w in zip(got[1:], want[1:], strict=True))
    if worst > PROPORTION_TOLERANCE:
        shown = ([round(g, 2) for g in got], [round(w, 2) for w in want])
        return f"proportions off: got {shown[0]}, expected {shown[1]}"
    return None


def check(
    raw: trimesh.Trimesh,
    final: trimesh.Trimesh,
    pieces: int,
    kept: int,
    expected_lwh: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """A pass or fail with reasons, and the numbers behind it."""
    extents = [float(e) for e in final.extents]
    aspect = max(extents) / max(min(extents), 1e-9)
    reasons = []
    if not np.isfinite(final.vertices).all():
        reasons.append("non-finite vertices")
    if len(final.faces) < MIN_FACES:
        reasons.append(f"only {len(final.faces)} faces")
    if kept > MAX_PIECES:
        reasons.append(f"shattered: {kept} pieces kept of {pieces}")
    if aspect > MAX_ASPECT:
        reasons.append(f"extreme aspect {aspect:.1f}")
    off = proportions_off(extents, expected_lwh)
    if off:
        reasons.append(off)
    return {
        "pass": not reasons,
        "reasons": reasons,
        "faces_raw": int(len(raw.faces)),
        "faces": int(len(final.faces)),
        "watertight": bool(final.is_watertight),
        "pieces": pieces,
        "pieces_kept": kept,
        "extents_m": [round(e, 3) for e in extents],
        "aspect": round(aspect, 2),
    }


def process(
    raw: trimesh.Trimesh,
    longest_m: float,
    expected_lwh: tuple[float, float, float] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    cleaned, pieces, kept = clean(raw)
    final = decimate(place(to_z_up(cleaned), longest_m))
    return final, check(raw, final, pieces, kept, expected_lwh)


VIEWS = ((22, -55), (0, -90), (90, -90))  # three-quarter, side, top


def render_views(mesh: trimesh.Trimesh, px: int = 220) -> list[Any]:
    """Three small shaded views as PIL images, drawn with matplotlib (no OpenGL needed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    m = (
        mesh
        if len(mesh.faces) <= RENDER_FACES
        else mesh.simplify_quadric_decimation(face_count=RENDER_FACES)
    )
    tris = m.vertices[m.faces]
    light = np.array([0.4, -0.5, 0.8])
    light /= np.linalg.norm(light)
    shade = np.clip(m.face_normals @ light, 0.0, 1.0) * 0.7 + 0.3
    colours = [to_rgba((0.72 * s, 0.74 * s, 0.78 * s)) for s in shade]
    extent = float(max(m.extents)) / 2 * 1.05
    centre = m.bounds.mean(axis=0)
    out = []
    for elev, azim in VIEWS:
        fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(tris, facecolors=colours, linewidths=0))
        ax.set_xlim(centre[0] - extent, centre[0] + extent)
        ax.set_ylim(centre[1] - extent, centre[1] + extent)
        ax.set_zlim(centre[2] - extent, centre[2] + extent)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        out.append(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return out


def sheet(rows: list[tuple[str, list[Any]]], cell: int = 220) -> Any:
    """Rows of (label, images) side by side with the label drawn above each row."""
    from PIL import Image, ImageDraw

    cols = max(len(images) for _, images in rows)
    out = Image.new("RGB", (cols * cell, len(rows) * (cell + 14)), "white")
    draw = ImageDraw.Draw(out)
    for r, (label, images) in enumerate(rows):
        y = r * (cell + 14)
        draw.text((4, y + 1), label, fill=(0, 0, 0))
        for c, image in enumerate(images):
            out.paste(image.resize((cell, cell)), (c * cell, y + 14))
    return out
