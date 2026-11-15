"""scripts/godot/make_demo_zip.py: it refuses a Godot zip that does not match the published checksum
and its launcher points at the bundled project and tile. (CI has no Godot, so the build itself is
checked by hand: see the PR for FRK-116.)"""

import hashlib
import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[2]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_demo_zip", ROOT / "scripts/godot/make_demo_zip.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_godot(tmp_path: Path, digest: str | None) -> Path:
    archive = tmp_path / "godot.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Godot_v4.7.2-stable_win64.exe", b"exe")
    real = hashlib.sha512(archive.read_bytes()).hexdigest()
    (tmp_path / "SHA512-SUMS.txt").write_text(
        f"{digest or real}  Godot_v4.7.2-stable_win64.exe.zip\n"
    )
    return archive


def test_a_godot_zip_that_does_not_match_the_checksum_is_refused(tmp_path: Path) -> None:
    script = _script()
    script.check_godot(_fake_godot(tmp_path, None))  # the right sum passes
    with pytest.raises(SystemExit, match="does not match"):
        script.check_godot(_fake_godot(tmp_path, "0" * 128))


def test_a_missing_checksum_file_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "godot.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(SystemExit, match="SHA512-SUMS.txt"):
        _script().check_godot(archive)


def test_the_launcher_and_readme_name_the_bundled_parts() -> None:
    script = _script()
    assert '--path "%~dp0project"' in script.BAT and '--scene "%~dp0tile"' in script.BAT
    assert "Fraktal.exe" in script.BAT
    for fragment in ("Run Fraktal.bat", "car", "person", "drone", "Fraktal.console.exe"):
        assert fragment in script.README
    assert "ground_rough.npy" in script.SKIP_TILE  # the one big file the viewer never reads
