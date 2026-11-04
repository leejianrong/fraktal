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
