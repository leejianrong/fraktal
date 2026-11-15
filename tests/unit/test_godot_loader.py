"""The Godot viewer reads the arrays `fraktal scene` writes. CI has no Godot, so these tests tie the
GDScript loader to the export contract by reading both as text."""

from pathlib import Path

ROOT = Path(__file__).parents[2]
TILE = (ROOT / "godot/scripts/tile.gd").read_text()
EXPORT = (ROOT / "src/fraktal/blender_export.py").read_text()

# prefix of the packed arrays -> where scene.json keeps that mesh's materials
PACKED = {
    "markings": 'scene["markings"]["materials"]',
    "props": 'scene["prop_materials"]',
    "walls": 'scene["wall_materials"]',
    "cables": 'scene["cables"]["materials"]',
}


def test_the_viewer_loads_every_packed_mesh_the_exporter_writes() -> None:
    for prefix, materials in PACKED.items():
        assert f'"{prefix}_vertices.npy"' not in EXPORT  # written through a loop, by prefix
        assert f'np.save(out_dir / f"{prefix}_{{key}}.npy"' in EXPORT, prefix
        assert f'_build_packed(scene_dir, assets_dir, "{prefix}"' in TILE, prefix
        assert materials in TILE, prefix


def test_the_loader_reads_the_files_the_packed_layout_has() -> None:
    for name in ("_vertices.npy", "_faces.npy", "_face_material.npy"):
        assert f'prefix + "{name}"' in TILE, name
    assert "MAX_SURFACES" in TILE  # a Godot mesh holds 256 surfaces and props use more materials


def test_walls_take_their_plaster_textures_and_cables_are_double_sided() -> None:
    assert 'scene.get("wall_textures", {})' in TILE
    assert '"cables", "Cables", scene["cables"]["materials"], {}, true' in TILE
