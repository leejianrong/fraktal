"""Realism ablation (FRK-18): which V3 components help the downstream model?

Variants add realism components cumulatively, from the flat V1 render to the randomised one.
Arms 1 and 2 do not depend on synthetic data, so they run once per seed; each variant adds an
arm 3 (synthetic pretrain, then the real 10% subset) and a synthetic-only zero-shot score. The
report is rewritten after every result, so a failed run keeps what it finished.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from fraktal import benchmark, realism
from fraktal.benchmark import Recipe, Subset, TileSource
from fraktal.log import emit
from fraktal.render import AMBIENT, KHARTOUM_TONE, RenderConfig

NEUTRAL: dict[str, Any] = {
    "variation": 0.0,
    "tone": (1.0, 1.0, 1.0),
    "texture": 0.0,
    "psf_sigma_px": 0.0,
    "noise_sigma": 0.0,
    "haze": 0.0,
    "penumbra_m": 0.0,
    "ambient": AMBIENT,
    "contrast": 1.0,
    "bright_roofs": 0.0,
    "blank_prob": 0.0,
    "objects": 0.0,
    "roof_detail": 0.0,
}
# Each variant adds one component to the one before it. The last draws them all per region.
STEPS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("v0_flat", {}),
    ("v1_tone", {"tone": KHARTOUM_TONE}),
    ("v2_reflectance", {"variation": 0.15}),
    ("v3_texture", {"texture": 0.7}),
    ("v4_sensor", {"psf_sigma_px": 0.8, "noise_sigma": 8.0}),
    ("v5_atmosphere", {"haze": 0.1, "penumbra_m": 0.3}),
)
RANDOMISED = "v6_randomised"
# Brightness and darkness extremes on top of the atmosphere variant (FRK-21). Not part of the
# randomised variant, which keeps its original definition so earlier results stay comparable.
EXTREMES = "v7_extremes"
EXTREME_FIELDS: dict[str, Any] = {
    "ambient": 0.18,
    "contrast": 1.25,
    "bright_roofs": 0.15,
    "blank_prob": 0.25,
}
# Trees, vehicles and roof stripes on top of the extremes variant (FRK-22).
OBJECTS = "v8_objects"
OBJECT_FIELDS: dict[str, Any] = {"objects": 1.0, "roof_detail": 0.12}
VARIANTS: tuple[str, ...] = (*(name for name, _ in STEPS), RANDOMISED, EXTREMES, OBJECTS)


def variant_config(name: str, base: RenderConfig) -> tuple[RenderConfig, bool]:
    """The render config for a variant and whether its parameters are drawn per region."""
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose from {', '.join(VARIANTS)}")
    fields: dict[str, Any] = dict(NEUTRAL)
    for step, change in STEPS:
        fields.update(change)
        if step == name:
            break
    if name in (EXTREMES, OBJECTS):
        fields.update(EXTREME_FIELDS)
    if name == OBJECTS:
        fields.update(OBJECT_FIELDS)
    if name == RANDOMISED:
        return dataclasses.replace(base, **fields), True
    return dataclasses.replace(base, **fields), False


def subset_arm(fraction: float) -> str:
    """Report key of the real-subset arm: ``real_10pct`` at 0.1, ``real_1pct`` at 0.01."""
    return f"real_{fraction * 100:g}pct"


def verdict(
    runs: Sequence[dict[str, Any]], names: Sequence[str], fraction: float
) -> dict[str, Any]:
    """The ADR-0006 bar over the seeds so far: met, missed or undecidable, per variant.

    The test only counts where arm 2 beats arm 1 by 10+ points; below that it says so.
    """
    key = subset_arm(fraction)
    arm1 = sum(r["arms"][key]["iou"] for r in runs) / len(runs)
    arm2 = sum(r["arms"]["real_100pct"]["iou"] for r in runs) / len(runs)
    gap = 100 * (arm2 - arm1)
    out: dict[str, Any] = {"arm2_minus_arm1_points": gap, "decidable": gap >= 10, "variants": {}}
    for name in names:
        done = [r["variants"][name]["arm3_iou"] for r in runs if name in r["variants"]]
        if not done:
            continue
        arm3 = sum(done) / len(done)
        lift, share = 100 * (arm3 - arm1), arm3 / arm2 if arm2 else 0.0
        state = "undecidable" if gap < 10 else ("met" if lift >= 5 and share >= 0.9 else "missed")
        out["variants"][name] = {
            "arm3_minus_arm1_points": lift,
            "share_of_arm2": share,
            "bar": state,
        }
    return out


class VariantNotReady(RuntimeError):
    pass


def wait_for_variant(
    root: Path,
    name: str,
    timeout_s: float,
    poll_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Chips dir of a variant, waiting for the synth job that writes it. Fails fast on FAILED."""
    chips = root / name / "chips"
    started = time.monotonic()
    while not (chips / "chips.json").is_file():
        if (root / name / "FAILED").exists():
            raise VariantNotReady(f"synth for {name} failed; see its log")
        waited = time.monotonic() - started
        if waited > timeout_s:
            raise VariantNotReady(f"{name} not ready after {int(waited)} s")
        emit("ablation_wait", variant=name, waited_s=int(waited))
        sleep(poll_s)
    return chips


def _write(report: dict[str, Any], path: Path) -> None:
    benchmark.write_report(report, path)


def run_ablation(
    train_source: TileSource,
    test_source: TileSource,
    recipe: Recipe,
    seeds: Sequence[int],
    device: str,
    chips_for: Callable[[str], TileSource],
    names: Sequence[str],
    out: Path,
    *,
    real_chips: Sequence[Any] = (),
    fraction: float = 0.1,
    log_every: int = 100,
) -> dict[str, Any]:
    device = benchmark.resolve_device(device)
    report: dict[str, Any] = {
        "schema": 1,
        "recipe": dataclasses.asdict(recipe),
        "device": device,
        "variants": list(names),
        "fraction": fraction,
        "runs": [],
        "status": "running",
    }
    emit("ablation_start", seeds=list(seeds), variants=list(names), test_tiles=len(test_source))
    for seed in seeds:
        run: dict[str, Any] = {"seed": seed, "arms": {}, "variants": {}}
        report["runs"].append(run)
        subset = benchmark.choose_subset(len(train_source), fraction, seed)
        arm1_key = subset_arm(fraction)
        for arm, indices in (
            (arm1_key, subset),
            ("real_100pct", list(range(len(train_source)))),
        ):
            started = time.monotonic()
            emit("arm_start", arm=arm, seed=seed, train_tiles=len(indices))
            if arm == arm1_key:
                emit("ablation_subset", fraction=fraction, tiles=len(indices), of=len(train_source))
            model, trace = benchmark.train(
                Subset(train_source, indices), recipe, seed, device, label=arm, log_every=log_every
            )
            iou = benchmark.evaluate(model, test_source, recipe, device, label=arm)
            run["arms"][arm] = {"iou": iou, "final_loss": trace[-1]}
            emit(
                "arm_done",
                arm=arm,
                seed=seed,
                iou=round(iou, 4),
                seconds=round(time.monotonic() - started),
            )
            emit("ablation_result", seed=seed, arm=arm, iou=round(iou, 4))
            _write(report, out)
        base = run["arms"][arm1_key]["iou"]
        for name in names:
            started = time.monotonic()
            source = chips_for(name)
            emit("arm_start", arm=name, seed=seed, synthetic_tiles=len(source))
            pre, _ = benchmark.train(
                source, recipe, seed, device, label=f"{name}:pretrain", log_every=log_every
            )
            state = {k: v.detach().cpu().clone() for k, v in pre.state_dict().items()}
            zero_shot = benchmark.evaluate(
                pre, test_source, recipe, device, label=f"{name}:zero_shot"
            )
            model, trace = benchmark.train(
                Subset(train_source, subset),
                recipe,
                seed,
                device,
                init_state=state,
                label=f"{name}:finetune",
                log_every=log_every,
            )
            iou = benchmark.evaluate(model, test_source, recipe, device, label=name)
            entry: dict[str, Any] = {
                "arm3_iou": iou,
                "arm3_minus_arm1_points": 100 * (iou - base),
                "zero_shot_iou": zero_shot,
                "synthetic_tiles": len(source),
                "final_loss": trace[-1],
                "seconds": round(time.monotonic() - started, 1),
            }
            if real_chips:
                synthetic = [source.get(i)[0] for i in range(min(len(source), 100))]
                entry["gauge"] = realism.compare(list(real_chips), synthetic)["distances"]
            run["variants"][name] = entry
            emit(
                "ablation_result",
                seed=seed,
                variant=name,
                arm3_iou=round(iou, 4),
                vs_arm1_points=round(entry["arm3_minus_arm1_points"], 2),
                zero_shot=round(zero_shot, 4),
            )
            report["q14"] = verdict(report["runs"], names, fraction)
            _write(report, out)
    report["status"] = "complete"
    report["hardware"] = benchmark.hardware_stats(device)
    _write(report, out)
    emit("ablation_done", runs=len(report["runs"]))
    return report
