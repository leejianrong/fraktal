import json
from pathlib import Path

import pytest

from fraktal import assets
from fraktal.cli import main

REPO_ASSETS = Path(__file__).resolve().parents[2] / "assets"


def make(root: Path, **changes: object) -> Path:
    """An assets directory with one good owned row, then ``changes`` applied to that row."""
    (root / "materials").mkdir(parents=True)
    file = root / "materials" / "m.json"
    file.write_text("{}\n")
    row: dict[str, object] = {
        "id": "m1", "file": "materials/m.json", "kind": "material", "source": "fraktal",
        "licence": "owned", "author": "me", "url": "https://example.org",
        "sha256": assets.sha256(file), "verified": True,
    }  # fmt: skip
    row.update(changes)
    (root / "manifest.json").write_text(json.dumps({"assets": [row]}))
    return root


def test_the_real_library_is_clean() -> None:
    assert assets.check(REPO_ASSETS) == []


def test_a_good_row_passes_and_cc0_is_allowed(tmp_path: Path) -> None:
    assert assets.check(make(tmp_path / "a")) == []
    assert assets.check(make(tmp_path / "b", licence="CC0-1.0")) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"licence": "CC-BY-4.0"}, "not allowed"),
        ({"licence": "CC-BY-NC-4.0"}, "not allowed"),
        ({"licence": ""}, "missing licence"),
        ({"verified": False}, "not verified"),
        ({"author": None}, "missing author"),
        ({"sha256": "0" * 64}, "sha256 does not match"),
        ({"file": "materials/gone.json"}, "does not exist"),
    ],
)
def test_a_bad_row_fails(tmp_path: Path, change: dict[str, object], expected: str) -> None:
    problems = assets.check(make(tmp_path / "x", **change))
    assert any(expected in p for p in problems), problems


def test_a_file_without_a_row_fails(tmp_path: Path) -> None:
    root = make(tmp_path / "x")
    (root / "materials" / "stray.png").write_bytes(b"png")
    assert any("stray.png: no manifest row" in p for p in assets.check(root))


def test_a_missing_manifest_fails(tmp_path: Path) -> None:
    assert assets.check(tmp_path) and "cannot read" in assets.check(tmp_path)[0]


def test_the_cli_exits_non_zero_on_a_bad_row(tmp_path: Path) -> None:
    assert main(["assets", "--root", str(make(tmp_path / "ok"))]) == 0
    assert main(["assets", "--root", str(make(tmp_path / "bad", licence="CC-BY-4.0"))]) != 0


GENERATOR = {
    "model": "stabilityai/stable-diffusion-xl-base-1.0", "version": "1.0",
    "licence": "CreativeML Open RAIL++-M", "prompt": "p", "seed": 1, "date": "2026-11-05",
}  # fmt: skip


def test_a_generated_asset_needs_a_full_generator_block(tmp_path: Path) -> None:
    good = make(tmp_path / "g", licence="generated-permissive", generator=GENERATOR)
    assert assets.check(good) == []
    none = assets.check(make(tmp_path / "n", licence="generated-permissive"))
    assert any("needs a generator block" in p for p in none), none
    thin = {k: v for k, v in GENERATOR.items() if k not in ("prompt", "seed")}
    partial = assets.check(make(tmp_path / "p", licence="Generated-Permissive", generator=thin))
    assert any("missing prompt" in p for p in partial) and any("missing seed" in p for p in partial)


def mesh_row(root: Path, generator: object) -> Path:
    make(root)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["assets"][0]["kind"] = "mesh"
    if generator is not None:
        manifest["assets"][0]["generator"] = generator
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


GOOD_MESH = {
    "model": "fraktal.propgen.container", "version": "1", "seed": 3, "date": "2026-11-08",
    "components": [{"name": "fraktal.propgen", "version": "1", "licence": "OWNED"}],
}  # fmt: skip


def test_a_mesh_without_a_generator_block_fails(tmp_path: Path) -> None:
    assert any("needs a generator block" in p for p in assets.check(mesh_row(tmp_path / "a", None)))
    assert assets.check(mesh_row(tmp_path / "b", GOOD_MESH)) == []


def test_a_mesh_generator_needs_every_key_and_component_fields(tmp_path: Path) -> None:
    bad = {k: v for k, v in GOOD_MESH.items() if k != "seed"}
    assert any("missing seed" in p for p in assets.check(mesh_row(tmp_path / "a", bad)))
    bad = {**GOOD_MESH, "components": [{"name": "x"}]}
    problems = assets.check(mesh_row(tmp_path / "b", bad))
    assert any("component is missing version" in p for p in problems)
    assert any("component is missing licence" in p for p in problems)
    bad = {**GOOD_MESH, "components": []}
    assert any("missing components" in p for p in assets.check(mesh_row(tmp_path / "c", bad)))


@pytest.mark.parametrize("name", ["TripoSG", "TRELLIS.2", "facebook/dinov3-vitl16", "RMBG-2.0"])
def test_a_mesh_made_with_an_excluded_component_fails(tmp_path: Path, name: str) -> None:
    generator = {**GOOD_MESH, "components": [{"name": name, "version": "1", "licence": "MIT"}]}
    assert any("excluded by ADR-0013" in p for p in assets.check(mesh_row(tmp_path, generator)))
