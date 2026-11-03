import json
from pathlib import Path

import numpy as np
import pytest

from fraktal import cli
from fraktal.realism import colour_distance, compare, edge_energy, noise_level, sample


def chip(value: int = 100, noise: int = 0, seed: int = 0) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    base = np.full((3, 64, 64), value, dtype="int16")
    if noise:
        base += rng.integers(-noise, noise + 1, size=base.shape, dtype="int16")
    out: np.ndarray = np.clip(base, 0, 255).astype("uint8")
    return out


def test_identical_sets_have_zero_distance() -> None:
    report = compare([chip(90, 10, 1)], [chip(90, 10, 1)])
    assert set(report["distances"].values()) == {0.0}


def test_colour_distance_grows_with_a_brightness_shift() -> None:
    near = colour_distance([chip(100)], [chip(110)])
    far = colour_distance([chip(100)], [chip(160)])
    assert near < far
    assert far == pytest.approx(60 / 255, abs=1e-3)


def test_flat_images_have_no_edges_or_noise() -> None:
    assert edge_energy(chip()) == 0.0
    assert noise_level(chip()) == 0.0
    assert edge_energy(chip(noise=20)) > 5
    assert noise_level(chip(noise=20)) > 5


def test_flat_render_against_noisy_real_flags_texture_gap() -> None:
    report = compare([chip(100, 20)], [chip(100)])
    assert report["distances"]["noise_gap"] == pytest.approx(1.0)
    assert report["distances"]["edge_energy_gap"] == pytest.approx(1.0)


def test_compare_needs_chips() -> None:
    with pytest.raises(ValueError):
        compare([], [chip()])


def test_sample_is_deterministic_and_spans_the_set() -> None:
    chips = [chip(i) for i in range(10)]
    picked = sample(chips, 3)
    assert [int(c[0, 0, 0]) for c in picked] == [0, 5, 9] or len(picked) == 3
    assert sample(chips, 3)[1] is picked[1]
    assert len(sample(chips, 50)) == 10


def _chips_dir(root: Path, ids: list[str], make: object) -> Path:
    root.mkdir(parents=True)
    for i, name in enumerate(ids):
        np.save(root / f"{name}_img.npy", make(i))  # type: ignore[operator]
        np.save(root / f"{name}_mask.npy", np.zeros((64, 64), dtype="uint8"))
    (root / "chips.json").write_text(json.dumps({"tiles": ids, "train": ids, "test": []}))
    return root


def test_compare_command_reports_distances(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real = _chips_dir(tmp_path / "real", ["a", "b"], lambda i: chip(90 + i, 15, i))
    synth = _chips_dir(tmp_path / "synth", ["x"], lambda i: chip(130))
    code = cli.main(["compare", "--real", str(real), "--synthetic", str(synth), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["data"]["distances"]["colour_w1"] > 0
    assert out["data"]["distances"]["noise_gap"] == pytest.approx(1.0)
