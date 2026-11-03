"""The downstream benchmark (ADR-0006): a fixed small U-Net and a fixed recipe.

Only the training data changes between arms. Every arm gets the same number of optimiser steps,
so a 10% arm is not handicapped by seeing fewer batches. PyTorch is imported lazily so the rest
of the package works without the optional ``train`` dependency group.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fraktal.ids import substream
from fraktal.log import emit


class TileSource(Protocol):
    """Chips for one split: uint8 images (3, H, W) and uint8 masks (H, W) with values 0 or 1."""

    def __len__(self) -> int: ...

    def get(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]: ...


class Subset:
    def __init__(self, source: TileSource, indices: Sequence[int]) -> None:
        self.source = source
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def get(self, index: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        return self.source.get(self.indices[index])


@dataclass(frozen=True)
class Recipe:
    """Everything that must be identical across arms. Recorded in every report."""

    steps: int = 3000
    batch_size: int = 16
    patch_px: int = 256
    base_channels: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_patch_px: int = 512
    threshold: float = 0.5


def require_torch() -> Any:
    try:
        import torch
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "PyTorch is not installed; run `uv sync --group train` (see CONTRIBUTING.md)"
        ) from err
    return torch


def choose_subset(n: int, fraction: float, seed: int) -> list[int]:
    """A seeded, sorted subset of ``n`` items; at least one item."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = min(n, max(1, round(n * fraction)))
    picked = substream(seed, "benchmark-subset").permutation(n)[:k]
    return sorted(int(i) for i in picked)


def build_unet(base: int = 16) -> Any:
    """A four-level U-Net with batch norm; about 2M parameters at ``base=16``."""
    torch = require_torch()
    from torch import nn

    def block(cin: int, cout: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):  # type: ignore[misc,unused-ignore]
        def __init__(self) -> None:
            super().__init__()
            widths = [base, base * 2, base * 4, base * 8]
            self.down = nn.ModuleList()
            cin = 3
            for w in widths:
                self.down.append(block(cin, w))
                cin = w
            self.mid = block(widths[-1], widths[-1] * 2)
            self.up = nn.ModuleList()
            self.merge = nn.ModuleList()
            cin = widths[-1] * 2
            for w in reversed(widths):
                self.up.append(nn.ConvTranspose2d(cin, w, 2, stride=2))
                self.merge.append(block(w * 2, w))
                cin = w
            self.head = nn.Conv2d(base, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x: Any) -> Any:
            skips = []
            for down in self.down:
                x = down(x)
                skips.append(x)
                x = self.pool(x)
            x = self.mid(x)
            for up, merge in zip(self.up, self.merge, strict=True):
                x = up(x)
                x = merge(torch.cat([x, skips.pop()], dim=1))
            return self.head(x)

    return UNet()


def _loss(logits: Any, target: Any) -> Any:
    torch = require_torch()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def _batch(
    source: TileSource, recipe: Recipe, rng: np.random.Generator
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    p = recipe.patch_px
    images = np.empty((recipe.batch_size, 3, p, p), dtype=np.float32)
    masks = np.empty((recipe.batch_size, 1, p, p), dtype=np.float32)
    for b in range(recipe.batch_size):
        img, msk = source.get(int(rng.integers(len(source))))
        _, h, w = img.shape
        if h < p or w < p:
            raise ValueError(f"tile {h}x{w} is smaller than the {p}px patch")
        r, c = int(rng.integers(h - p + 1)), int(rng.integers(w - p + 1))
        img_p, msk_p = img[:, r : r + p, c : c + p], msk[r : r + p, c : c + p]
        k = int(rng.integers(4))
        img_p, msk_p = np.rot90(img_p, k, axes=(1, 2)), np.rot90(msk_p, k)
        if rng.integers(2):
            img_p, msk_p = img_p[:, :, ::-1], msk_p[:, ::-1]
        images[b] = img_p.astype(np.float32) / 255.0
        masks[b, 0] = msk_p
    return images, masks


def _gpu_gb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(float(require_torch().cuda.max_memory_allocated()) / 2**30, 2)


def train(
    source: TileSource,
    recipe: Recipe,
    seed: int,
    device: str,
    *,
    init_state: dict[str, Any] | None = None,
    label: str = "train",
    log_every: int = 100,
) -> tuple[Any, list[float]]:
    """Train for ``recipe.steps`` optimiser steps. Returns the model and a loss trace."""
    torch = require_torch()
    if len(source) == 0:
        raise ValueError("no training tiles")
    init_seed = int(substream(seed, "benchmark-init").integers(2**31 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = build_unet(recipe.base_channels)
    if init_state is not None:
        model.load_state_dict(init_state)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe.steps)
    rng = substream(seed, "benchmark-batches")
    trace: list[float] = []
    started = time.monotonic()
    emit("train_start", label=label, steps=recipe.steps, tiles=len(source), device=device)
    for _step in range(recipe.steps):
        images, masks = _batch(source, recipe, rng)
        x = torch.from_numpy(images).to(device)
        y = torch.from_numpy(masks).to(device)
        loss = _loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        trace.append(float(loss.item()))
        done = _step + 1
        if log_every and (done % log_every == 0 or done == recipe.steps):
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = done / elapsed
            emit(
                "train_progress",
                label=label,
                step=done,
                steps=recipe.steps,
                loss=round(float(np.mean(trace[-log_every:])), 4),
                steps_per_s=round(rate, 2),
                eta_s=round((recipe.steps - done) / rate),
                gpu_peak_gb=_gpu_gb(device),
            )
    emit("train_done", label=label, seconds=round(time.monotonic() - started, 1))
    return model, trace


def evaluate(
    model: Any, source: TileSource, recipe: Recipe, device: str, *, label: str = "eval"
) -> float:
    """Pixel IoU of the building class over all tiles, accumulated globally."""
    torch = require_torch()
    model = model.to(device).eval()
    tp = fp = fn = 0
    p = recipe.eval_patch_px
    emit("eval_start", label=label, tiles=len(source))
    with torch.no_grad():
        for i in range(len(source)):
            if i and i % 25 == 0:
                emit("eval_progress", label=label, tile=i, tiles=len(source))
            img, msk = source.get(i)
            _, h, w = img.shape
            for r in range(0, h, p):
                for c in range(0, w, p):
                    rh, cw = min(p, h - r), min(p, w - c)
                    patch = img[:, r : r + rh, c : c + cw].astype(np.float32) / 255.0
                    pad_h, pad_w = -rh % 16, -cw % 16
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                    logits = model(torch.from_numpy(patch[None]).to(device))
                    prob = torch.sigmoid(logits)[0, 0, :rh, :cw].cpu().numpy()
                    pred = prob >= recipe.threshold
                    truth = msk[r : r + rh, c : c + cw] > 0
                    tp += int((pred & truth).sum())
                    fp += int((pred & ~truth).sum())
                    fn += int((~pred & truth).sum())
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    emit("eval_done", label=label, iou=round(iou, 4))
    return iou


def resolve_device(requested: str) -> str:
    torch = require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def run_arms(
    train_source: TileSource,
    test_source: TileSource,
    recipe: Recipe,
    seed: int,
    device: str = "auto",
    fraction: float = 0.1,
    synthetic: TileSource | None = None,
    log_every: int = 100,
) -> dict[str, Any]:
    """Arm 1 (real at ``fraction``), arm 2 (real 100%) and, given synthetic tiles, arm 3.

    Arm 3 pretrains on the synthetic tiles for ``recipe.steps`` steps, then fine-tunes on the
    same real subset as arm 1 for another ``recipe.steps``. The extra pretraining compute is
    the thing being tested, and the report says so.
    """
    device = resolve_device(device)
    arms: dict[str, Any] = {}
    subset = choose_subset(len(train_source), fraction, seed)
    for name, indices in (
        ("real_10pct", subset),
        ("real_100pct", list(range(len(train_source)))),
    ):
        started = time.monotonic()
        emit("arm_start", arm=name, seed=seed, train_tiles=len(indices))
        model, trace = train(
            Subset(train_source, indices), recipe, seed, device, label=name, log_every=log_every
        )
        arms[name] = {
            "iou": evaluate(model, test_source, recipe, device, label=name),
            "train_tiles": len(indices),
            "final_loss": trace[-1],
            "seconds": round(time.monotonic() - started, 1),
        }
        emit("arm_done", arm=name, iou=round(arms[name]["iou"], 4), seconds=arms[name]["seconds"])
    report: dict[str, Any] = {
        "schema": 1,
        "seed": seed,
        "device": device,
        "recipe": dataclasses.asdict(recipe),
        "test_tiles": len(test_source),
        "arms": arms,
    }
    if synthetic is not None:
        started = time.monotonic()
        emit("arm_start", arm="synthetic_pretrain_then_real_10pct", seed=seed)
        pre, pre_trace = train(
            synthetic, recipe, seed, device, label="synthetic_pretrain", log_every=log_every
        )
        state = {k: v.detach().cpu().clone() for k, v in pre.state_dict().items()}
        zero_shot = evaluate(pre, test_source, recipe, device, label="synthetic_only")
        model, trace = train(Subset(train_source, subset), recipe, seed, device, init_state=state)
        arms["synthetic_pretrain_then_real_10pct"] = {
            "iou": evaluate(model, test_source, recipe, device, label="arm3"),
            "train_tiles": len(subset),
            "synthetic_tiles": len(synthetic),
            "pretrain_steps": recipe.steps,
            "finetune_steps": recipe.steps,
            "pretrain_final_loss": pre_trace[-1],
            "final_loss": trace[-1],
            "seconds": round(time.monotonic() - started, 1),
        }
        emit("arm_done", arm="synthetic_pretrain_then_real_10pct", iou=round(arm3_iou(arms), 4))
        report["diagnostics"] = {"synthetic_only_zero_shot_iou": zero_shot}
        report["comparison"] = compare(arms)
    report["hardware"] = hardware_stats(device)
    return report


def hardware_stats(device: str) -> dict[str, Any]:
    """The device and, on CUDA, the peak memory any arm allocated (the 16 GB fit check)."""
    torch = require_torch()
    if not device.startswith("cuda"):
        return {"device": device}
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }


def arm3_iou(arms: dict[str, Any]) -> float:
    return float(arms["synthetic_pretrain_then_real_10pct"]["iou"])


def compare(arms: dict[str, Any]) -> dict[str, float]:
    """The ADR-0006 bar: arm 3 over arm 1 in IoU points, and arm 3 as a share of arm 2."""
    a1, a2 = arms["real_10pct"]["iou"], arms["real_100pct"]["iou"]
    a3 = arms["synthetic_pretrain_then_real_10pct"]["iou"]
    return {
        "arm3_minus_arm1_iou_points": 100 * (a3 - a1),
        "arm3_share_of_arm2": a3 / a2 if a2 else 0.0,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
