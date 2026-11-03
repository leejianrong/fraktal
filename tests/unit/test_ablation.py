import json
from pathlib import Path
from typing import Any

import pytest

from fraktal.ablation import (
    RANDOMISED,
    VARIANTS,
    VariantNotReady,
    run_ablation,
    subset_arm,
    variant_config,
    verdict,
    wait_for_variant,
)
from fraktal.render import KHARTOUM_TONE, RenderConfig

BASE = RenderConfig(seed=3)


def test_variants_add_one_component_each() -> None:
    flat, _ = variant_config("v0_flat", BASE)
    assert (flat.variation, flat.texture, flat.noise_sigma, flat.haze) == (0, 0, 0, 0)
    assert flat.tone == (1.0, 1.0, 1.0)
    tone, _ = variant_config("v1_tone", BASE)
    assert tone.tone == KHARTOUM_TONE and tone.variation == 0
    texture, _ = variant_config("v3_texture", BASE)
    assert texture.variation == 0.15 and texture.texture == 0.7 and texture.noise_sigma == 0
    atmos, randomised = variant_config("v5_atmosphere", BASE)
    assert atmos.noise_sigma == 8.0 and atmos.haze == 0.1 and not randomised
    assert variant_config(RANDOMISED, BASE)[1] is True


def test_variants_keep_the_base_seed_and_grid() -> None:
    for name in VARIANTS:
        config, _ = variant_config(name, BASE)
        assert (config.seed, config.gsd_m, config.tile_px) == (3, BASE.gsd_m, BASE.tile_px)


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        variant_config("v9_magic", BASE)


def test_wait_returns_when_chips_appear_and_reports_progress(tmp_path: Path) -> None:
    chips = tmp_path / "v0" / "chips"
    chips.mkdir(parents=True)
    calls = []

    def sleep(_: float) -> None:
        calls.append(1)
        (chips / "chips.json").write_text("{}")

    assert wait_for_variant(tmp_path, "v0", 100, 0.0, sleep) == chips
    assert len(calls) == 1


def test_wait_fails_fast_on_a_failed_synth_and_on_timeout(tmp_path: Path) -> None:
    (tmp_path / "v0").mkdir()
    (tmp_path / "v0" / "FAILED").write_text("x")
    with pytest.raises(VariantNotReady, match="failed"):
        wait_for_variant(tmp_path, "v0", 100, 0.0, lambda _: None)
    with pytest.raises(VariantNotReady, match="not ready after"):
        wait_for_variant(tmp_path, "never", -1, 0.0, lambda _: None)


def arms(a1: float, a2: float, arm3: float) -> dict[str, Any]:
    return {
        "arms": {"real_1pct": {"iou": a1}, "real_100pct": {"iou": a2}},
        "variants": {"v0": {"arm3_iou": arm3}},
    }


def test_subset_arm_names_follow_the_fraction() -> None:
    assert subset_arm(0.1) == "real_10pct"
    assert subset_arm(0.01) == "real_1pct"
    assert subset_arm(0.025) == "real_2.5pct"


def test_verdict_is_undecidable_unless_arm2_beats_arm1_by_ten_points() -> None:
    v = verdict([arms(0.60, 0.65, 0.70)], ["v0"], 0.01)
    assert v["decidable"] is False and v["variants"]["v0"]["bar"] == "undecidable"


def test_verdict_applies_the_five_point_and_ninety_percent_bar() -> None:
    met = verdict([arms(0.50, 0.70, 0.66)], ["v0"], 0.01)
    assert met["decidable"] and met["variants"]["v0"]["bar"] == "met"
    short = verdict([arms(0.50, 0.70, 0.54)], ["v0"], 0.01)  # +4 points
    assert short["variants"]["v0"]["bar"] == "missed"
    low_share = verdict([arms(0.30, 0.80, 0.40)], ["v0"], 0.01)  # +10 points but 50% of arm 2
    assert low_share["variants"]["v0"]["bar"] == "missed"


def test_verdict_averages_seeds_and_skips_unfinished_variants() -> None:
    v = verdict([arms(0.50, 0.70, 0.60), arms(0.52, 0.72, 0.58)], ["v0", "v1"], 0.01)
    assert set(v["variants"]) == {"v0"}
    assert v["arm2_minus_arm1_points"] == pytest.approx(20.0)


torch = pytest.importorskip("torch")
from test_benchmark import TINY, Squares  # noqa: E402


def test_run_ablation_reports_every_variant_and_keeps_partial_results(tmp_path: Path) -> None:
    out = tmp_path / "ablation.json"
    names = ["v0_flat", "v1_tone"]
    report = run_ablation(
        Squares(10), Squares(4), TINY, [0], "cpu", lambda _n: Squares(5), names, out,
        real_chips=[Squares(2).get(0)[0]], log_every=1000,
    )  # fmt: skip
    run = report["runs"][0]
    assert set(run["arms"]) == {"real_10pct", "real_100pct"}
    assert set(run["variants"]) == set(names)
    v = run["variants"]["v0_flat"]
    assert v["arm3_minus_arm1_points"] == pytest.approx(
        100 * (v["arm3_iou"] - run["arms"]["real_10pct"]["iou"])
    )
    assert "colour_w1" in v["gauge"] and 0 <= v["zero_shot_iou"] <= 1
    assert report["status"] == "complete"
    assert json.loads(out.read_text())["status"] == "complete"


def test_a_failing_variant_leaves_finished_results_on_disk(tmp_path: Path) -> None:
    out = tmp_path / "ablation.json"

    def chips_for(name: str) -> Squares:
        if name == "v1_tone":
            raise VariantNotReady("boom")
        return Squares(5)

    with pytest.raises(VariantNotReady):
        run_ablation(
            Squares(10),
            Squares(4),
            TINY,
            [0],
            "cpu",
            chips_for,
            ["v0_flat", "v1_tone"],
            out,
            log_every=1000,
        )
    saved = json.loads(out.read_text())
    assert saved["status"] == "running"
    assert (
        "v0_flat" in saved["runs"][0]["variants"] and "v1_tone" not in saved["runs"][0]["variants"]
    )


def test_a_scarce_fraction_names_the_arm_and_records_the_verdict(tmp_path: Path) -> None:
    out = tmp_path / "a.json"
    report = run_ablation(
        Squares(20), Squares(4), TINY, [0], "cpu", lambda _n: Squares(5), ["v0_flat"], out,
        fraction=0.05, log_every=1000,
    )  # fmt: skip
    assert set(report["runs"][0]["arms"]) == {"real_5pct", "real_100pct"}
    assert report["fraction"] == 0.05
    assert report["q14"]["variants"]["v0_flat"]["bar"] in {"met", "missed", "undecidable"}
    assert json.loads(out.read_text())["q14"] == report["q14"]


def test_extremes_variant_builds_on_the_atmosphere_variant_without_changing_randomised() -> None:
    from fraktal.ablation import EXTREMES

    atmosphere, _ = variant_config("v5_atmosphere", BASE)
    extremes, randomised = variant_config(EXTREMES, BASE)
    assert randomised is False
    assert (extremes.haze, extremes.noise_sigma) == (atmosphere.haze, atmosphere.noise_sigma)
    assert (
        extremes.blank_prob > 0 and extremes.contrast > 1 and extremes.ambient < atmosphere.ambient
    )
    assert atmosphere.blank_prob == 0 and atmosphere.contrast == 1
    random_config, is_random = variant_config(RANDOMISED, BASE)
    assert is_random and random_config.blank_prob == 0 and random_config.contrast == 1
