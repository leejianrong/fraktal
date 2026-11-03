import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from fraktal.benchmark import (  # noqa: E402
    Recipe,
    build_unet,
    choose_subset,
    evaluate,
    run_arms,
    train,
    write_report,
)


class Squares:
    """Synthetic tiles: bright squares on a dark noisy background, mask = the squares."""

    def __init__(self, n: int, size: int = 64) -> None:
        self.n, self.size = n, size

    def __len__(self) -> int:
        return self.n

    def get(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        rng = np.random.default_rng(index)
        img = rng.integers(0, 60, (3, self.size, self.size), dtype=np.uint8)
        msk = np.zeros((self.size, self.size), dtype=np.uint8)
        for _ in range(3):
            r, c = rng.integers(4, self.size - 20, 2)
            msk[r : r + 14, c : c + 14] = 1
        img[:, msk > 0] = 200
        return img, msk


TINY = Recipe(steps=60, batch_size=4, patch_px=32, base_channels=8, eval_patch_px=32, lr=3e-3)


def test_subset_is_seeded_sorted_and_nonempty() -> None:
    a = choose_subset(100, 0.1, 5)
    assert a == choose_subset(100, 0.1, 5)
    assert a != choose_subset(100, 0.1, 6)
    assert len(a) == 10 and a == sorted(a)
    assert len(choose_subset(3, 0.01, 0)) == 1
    assert choose_subset(7, 1.0, 0) == list(range(7))
    with pytest.raises(ValueError):
        choose_subset(5, 0.0, 0)


def test_unet_output_matches_input_size() -> None:
    import torch

    model = build_unet(8)
    assert model(torch.zeros(2, 3, 32, 48)).shape == (2, 1, 32, 48)


def test_training_learns_the_synthetic_task() -> None:
    src = Squares(8)
    model, trace = train(src, TINY, seed=1, device="cpu")
    assert trace[-1] < trace[0]
    assert evaluate(model, Squares(4), TINY, "cpu") > 0.5


def test_same_seed_same_result() -> None:
    iou = []
    for _ in range(2):
        model, _ = train(Squares(6), TINY, seed=3, device="cpu")
        iou.append(evaluate(model, Squares(3), TINY, "cpu"))
    assert abs(iou[0] - iou[1]) < 0.01


def test_report_records_recipe_seed_and_both_arms(tmp_path: Path) -> None:
    report = run_arms(Squares(10), Squares(3), TINY, seed=2, device="cpu")
    assert set(report["arms"]) == {"real_10pct", "real_100pct"}
    assert report["arms"]["real_10pct"]["train_tiles"] == 1
    assert report["arms"]["real_100pct"]["train_tiles"] == 10
    assert report["seed"] == 2 and report["recipe"]["steps"] == 60
    path = tmp_path / "r.json"
    write_report(report, path)
    assert json.loads(path.read_text())["arms"]["real_100pct"]["iou"] >= 0


def test_too_small_tile_is_an_error() -> None:
    class Tiny:
        def __len__(self) -> int:
            return 1

        def get(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            return np.zeros((3, 16, 16), dtype=np.uint8), np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="smaller"):
        train(Tiny(), TINY, seed=0, device="cpu")


def test_synthetic_arm_reports_a_comparison() -> None:
    report = run_arms(
        Squares(10), Squares(3), TINY, seed=2, device="cpu", synthetic=Squares(5, size=96)
    )
    arm = report["arms"]["synthetic_pretrain_then_real_10pct"]
    assert arm["train_tiles"] == report["arms"]["real_10pct"]["train_tiles"]  # same real data
    assert arm["synthetic_tiles"] == 5
    assert arm["pretrain_steps"] == arm["finetune_steps"] == 60
    assert set(report["comparison"]) == {"arm3_minus_arm1_iou_points", "arm3_share_of_arm2"}
    assert "synthetic_only_zero_shot_iou" in report["diagnostics"]


def test_report_has_hardware_and_arm_timings() -> None:
    report = run_arms(Squares(6), Squares(2), TINY, seed=0, device="cpu")
    assert report["hardware"] == {"device": "cpu"}
    assert all(arm["seconds"] >= 0 for arm in report["arms"].values())
