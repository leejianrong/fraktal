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


def test_people_and_drones_are_blocked_by_what_the_exporter_writes() -> None:
    """Walkers stop at compound walls, trunks and poles from scene.json, and at buildings."""
    assert (
        'scene.get("walls", [])' in TILE
        and 'wall["segments"]' in TILE
        and 'wall["height_m"]' in TILE
    )
    assert 'scene["trees"]' in TILE and 'scene["poles"]' in TILE
    assert "func person_blocked(" in TILE and "roof_at(" in TILE
    for key in ('"walls"', '"trees"', '"poles"'):
        assert key in EXPORT or key.strip('"') in EXPORT, key  # the exporter writes them


def test_the_viewer_has_three_modes_with_their_keys() -> None:
    main = (ROOT / "godot/scripts/main.gd").read_text()
    for name in ("walker.gd", "drone.gd"):
        assert (ROOT / "godot/scripts" / name).is_file(), name
    for fragment in (
        'KEY_1:\n\t\t\t\t_set_mode("car")',
        '_set_mode("person")',
        '_set_mode("drone")',
    ):
        assert fragment in main, fragment
    assert "_selftest_person" in main and "_selftest_drone" in main  # headless checks cover both


def test_the_replay_reads_the_tracks_scene_writes() -> None:
    agents = (ROOT / "godot/scripts/agents.gd").read_text()
    cli = (ROOT / "src/fraktal/cli.py").read_text()
    for name in ("agents.json", "people.json", "drones.json"):
        assert f'"{name}"' in agents and f'"{name}"' in cli, name
    for key in ('d["ids"]', 'd["t"]', 'd["x"]', 'd["y"]', 'd["heading"]'):
        assert key in agents, key  # the fields Tracks.as_dict writes
    assert "--agents" in cli
