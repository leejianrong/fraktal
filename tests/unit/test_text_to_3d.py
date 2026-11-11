import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "props"


@pytest.fixture(scope="module")
def t3d():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("text_to_3d")
    finally:
        sys.path.remove(str(SCRIPTS))


def _write(tmp_path: Path, rows: object) -> Path:
    path = tmp_path / "p.json"
    path.write_text(json.dumps(rows))
    return path


def test_the_shipped_prompt_file_is_valid(t3d) -> None:  # type: ignore[no-untyped-def]
    rows = t3d.load_prompts(SCRIPTS / "prompts-20.json")
    assert len(rows) == 20 and all(r["expected_lwh"] for r in rows)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        {"id": "a"},
        [{"id": "a", "prompt": "x"}],
        [{"id": "a", "prompt": " ", "size_m": 1}],
        [{"id": "a", "prompt": "x", "size_m": 0}],
        [{"id": "a/b", "prompt": "x", "size_m": 1}],
        [{"id": "a", "prompt": "x", "size_m": 1}, {"id": "a", "prompt": "y", "size_m": 1}],
    ],
)
def test_bad_prompt_files_are_refused(t3d, tmp_path: Path, rows: object) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(t3d.PromptError):
        t3d.load_prompts(_write(tmp_path, rows))


def test_records_carry_prompt_seeds_times_and_the_qa_verdict(t3d) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {"id": "a", "prompt": "a red bike", "size_m": 1.9},
        {"id": "b", "prompt": "a blue tank", "size_m": 1.7},
    ]
    images = {"per_image_s": {"a": 7.0, "b": 6.0}}
    cutouts = {"per_image_s": {"a": 0.5, "b": 0.5}}
    meshes = {
        "per_mesh_s": {"a": 70.0},
        "qa": {"a": {"pass": True, "reasons": [], "faces": 20000}},
    }
    one, two = t3d.build_records(rows, 5, images, cutouts, meshes)
    assert (one["image_seed"], one["mesh_seed"]) == (1005, 2005)
    assert (two["image_seed"], two["mesh_seed"]) == (1006, 2006)
    assert one["pass"] and one["seconds"] == 77.5 and one["faces"] == 20000
    assert not two["pass"] and two["reasons"] == ["no mesh"] and two["mesh_s"] is None
    again = t3d.build_records(rows, 5, images, cutouts, meshes)
    assert again == [one, two]  # a fixed seed gives the same record


def test_cost_per_prop_is_the_mean_seconds_times_the_hourly_rate(t3d) -> None:  # type: ignore[no-untyped-def]
    records = [{"seconds": 70.0, "pass": True}, {"seconds": 90.0, "pass": False}]
    got = t3d.summarise(records, 0.72)
    assert got["mean_seconds_per_prop"] == 80.0 and got["passed_qa"] == 1
    assert got["usd_per_prop"] == pytest.approx(80.0 / 3600 * 0.72, abs=1e-5)
    assert t3d.summarise(records, None)["usd_per_prop"] is None
    assert t3d.summarise([], 0.72)["mean_seconds_per_prop"] == 0.0
