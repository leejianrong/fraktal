"""Realism report: charts from an ablation JSON plus, when chips are at hand, distribution
plots and chip strips, assembled into a PDF.

Needs the optional ``report`` dependency group (``make install-report``). Real imagery ends up in
the output, so write it under the git-ignored ``data/`` directory (ADR-0008).
"""

from __future__ import annotations

import json
import statistics as st
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fraktal.realism import sample

Chip = np.ndarray[Any, Any]
INK, MUTE, BLUE, ORANGE = "#1b1f24", "#6b7280", "#2e6fdb", "#e4572e"
PALETTE = ["#e4572e", "#f3a712", "#29a3a3", "#7a5cc7", "#2e6fdb", "#8a8f98", "#17a34a"]
LABELS = {
    "v0_flat": "flat (V1)",
    "v1_tone": "+ tone",
    "v2_reflectance": "+ reflectance",
    "v3_texture": "+ texture",
    "v4_sensor": "+ sensor",
    "v5_atmosphere": "+ haze/shadow",
    "v6_randomised": "randomised",
}


def _plt() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise RuntimeError("matplotlib is missing; run `make install-report`") from err
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def load_ablation(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    if not report.get("runs"):
        raise ValueError(f"{path} has no finished runs")
    return report


def _arm1_key(report: Mapping[str, Any]) -> str:
    return f"real_{report.get('fraction', 0.1) * 100:g}pct"


def _label(name: str) -> str:
    return LABELS.get(name, name)


def summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Headline numbers, computed from the report so the text can never drift from the data."""
    runs, names, key = report["runs"], report["variants"], _arm1_key(report)
    arm1 = st.mean(r["arms"][key]["iou"] for r in runs)
    arm2 = st.mean(r["arms"]["real_100pct"]["iou"] for r in runs)
    means = {
        v: st.mean(r["variants"][v]["arm3_iou"] for r in runs if v in r["variants"])
        for v in names
        if any(v in r["variants"] for r in runs)
    }
    zero = {
        v: st.mean(r["variants"][v]["zero_shot_iou"] for r in runs if v in r["variants"])
        for v in means
    }
    best = max(means, key=lambda v: means[v])
    return {
        "arm1": arm1,
        "arm2": arm2,
        "gap_points": 100 * (arm2 - arm1),
        "best": best,
        "best_iou": means[best],
        "best_vs_arm1_points": 100 * (means[best] - arm1),
        "zero_shot_range": (min(zero.values()), max(zero.values())),
        "seeds": len(runs),
        "fraction": report.get("fraction", 0.1),
        "q14": report.get("q14"),
    }


def arm3_chart(report: Mapping[str, Any], path: Path) -> None:
    plt = _plt()
    runs, key = report["runs"], _arm1_key(report)
    names = [v for v in report["variants"] if any(v in r["variants"] for r in runs)]
    arm1 = st.mean(r["arms"][key]["iou"] for r in runs)
    arm2 = st.mean(r["arms"]["real_100pct"]["iou"] for r in runs)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, v in enumerate(names):
        ys = [r["variants"][v]["arm3_iou"] for r in runs if v in r["variants"]]
        ax.scatter([i] * len(ys), ys, color=BLUE, s=28, alpha=0.6, zorder=3)
        ax.scatter([i], [st.mean(ys)], marker="_", s=600, color=BLUE, zorder=4, linewidths=3)
    ax.axhline(arm1, color=INK, lw=1.2, ls="--")
    ax.text(
        -0.45, arm1, f"real {report.get('fraction', 0.1):.0%} (arm 1)", fontsize=8.5, va="bottom"
    )
    ax.axhline(arm2, color=ORANGE, lw=1.2, ls="--")
    ax.text(-0.45, arm2, "real 100% (arm 2)", fontsize=8.5, color=ORANGE, va="bottom")
    ax.set_xticks(range(len(names)), [_label(v) for v in names], rotation=20, ha="right")
    ax.set_ylabel("IoU on held-out tiles")
    ax.set_title("Synthetic pretraining, then the real subset", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def zero_shot_chart(report: Mapping[str, Any], path: Path) -> None:
    plt = _plt()
    runs = report["runs"]
    names = [v for v in report["variants"] if any(v in r["variants"] for r in runs)]
    z = [[r["variants"][v]["zero_shot_iou"] for r in runs if v in r["variants"]] for v in names]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.bar(range(len(names)), [st.mean(x) for x in z], color=BLUE, width=0.55, alpha=0.85)
    for i, x in enumerate(z):
        ax.scatter([i] * len(x), x, color=INK, s=14, zorder=3)
    ax.set_xticks(range(len(names)), [_label(v) for v in names], rotation=20, ha="right")
    ax.set_ylabel("IoU, synthetic-only model")
    ax.set_title(
        "Trained on synthetic chips alone, scored on real tiles", loc="left", fontsize=10.5
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def gauge_chart(report: Mapping[str, Any], path: Path) -> bool:
    """Gauge distances beside the downstream gap. False when the report has no gauge."""
    plt = _plt()
    runs, key = report["runs"], _arm1_key(report)
    names = [v for v in report["variants"] if "gauge" in runs[0]["variants"].get(v, {})]
    if not names:
        return False
    arm1 = st.mean(r["arms"][key]["iou"] for r in runs)
    fig, axs = plt.subplots(1, 4, figsize=(10, 3.2))
    x = list(range(len(names)))
    for ax, (k, title) in zip(
        axs,
        [
            ("colour_w1", "Colour distance"),
            ("edge_energy_gap", "Edge-energy gap"),
            ("noise_gap", "Noise gap"),
        ],
        strict=False,
    ):
        ax.plot(x, [runs[0]["variants"][v]["gauge"][k] for v in names], marker="o", color=ORANGE)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylim(0, None)
    gap = [100 * (st.mean(r["variants"][v]["arm3_iou"] for r in runs) - arm1) for v in names]
    axs[3].plot(x, gap, marker="o", color=BLUE)
    axs[3].axhline(0, color=MUTE, lw=0.8, ls="--")
    axs[3].set_title("Arm 3 vs arm 1 (points)", fontsize=9.5, loc="left")
    for ax in axs:
        ax.set_xticks(x, [str(i) for i in x])
    fig.supxlabel(
        "variant index; lower is closer to real for the first three", fontsize=8.5, color=MUTE
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def _hist(chips: Sequence[Chip], band: int) -> np.ndarray[Any, Any]:
    counts = np.zeros(256)
    for c in chips:
        counts += np.bincount(c[band].ravel(), minlength=256)[:256]
    out: np.ndarray[Any, Any] = counts / counts.sum()
    return out


def _gradients(chips: Sequence[Chip]) -> np.ndarray[Any, Any]:
    out = []
    for c in chips:
        g = c.astype("float32").mean(axis=0)
        out.append(np.abs(np.diff(g, axis=1))[::4, ::4].ravel())
    return np.concatenate(out)


def distribution_chart(
    real: Sequence[Chip], variants: Mapping[str, Sequence[Chip]], path: Path
) -> None:
    """Colour, gradient, per-chip colour and contrast distributions, real against each variant."""
    plt = _plt()
    sets: dict[str, Sequence[Chip]] = {"real": real, **{_label(k): v for k, v in variants.items()}}
    colours = {
        k: ("#111" if k == "real" else PALETTE[i % len(PALETTE)]) for i, k in enumerate(sets)
    }
    width = {k: 2.2 if k == "real" else 1.3 for k in sets}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for band, name in enumerate("RGB"):
        for k, cs in sets.items():
            h = np.convolve(_hist(cs, band), np.ones(5) / 5, "same")
            ax[0, band].plot(range(256), h, label=k, color=colours[k], lw=width[k])
        ax[0, band].set_title(f"{name} channel pixel distribution")
        ax[0, band].set_xlabel("digital number")
        ax[0, band].set_ylim(0, None)
    ax[0, 0].legend(fontsize=8)
    bins = np.linspace(0, 60, 61)
    for k, cs in sets.items():
        h, _ = np.histogram(_gradients(cs), bins=bins, density=True)
        ax[1, 0].plot(bins[1:], h, color=colours[k], lw=width[k])
        means = np.array([c.reshape(3, -1).mean(axis=1) for c in cs])
        ax[1, 1].scatter(means[:, 0], means[:, 1], s=14, color=colours[k], alpha=0.7)
        spread = [c.astype("float32").mean(axis=0).std() for c in cs]
        ax[1, 2].hist(
            spread, bins=np.linspace(0, 90, 30), histtype="step", color=colours[k], lw=width[k]
        )
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Horizontal gradient magnitude (log)")
    ax[1, 0].set_xlabel("|dI| per pixel")
    ax[1, 1].set_title("Per-chip mean colour")
    ax[1, 1].set_xlabel("mean R")
    ax[1, 1].set_ylabel("mean G")
    ax[1, 2].set_title("Per-chip contrast (grey std)")
    ax[1, 2].set_xlabel("std of grey level")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _crop(chip: Chip, n: int = 260) -> np.ndarray[Any, Any]:
    h, w = chip.shape[1:]
    r, c = max(0, (h - n) // 2), max(0, (w - n) // 2)
    out: np.ndarray[Any, Any] = np.moveaxis(chip[:, r : r + n, c : c + n], 0, -1)
    return out


def chip_strip(rows: Mapping[str, Sequence[Chip]], path: Path) -> None:
    """One row of chip crops per label, e.g. real, then each variant at the same locations."""
    plt = _plt()
    n_cols = max(len(v) for v in rows.values())
    fig, axs = plt.subplots(
        len(rows), n_cols, figsize=(1.9 * n_cols, 1.9 * len(rows)), squeeze=False
    )
    for i, (label, chips) in enumerate(rows.items()):
        for j in range(n_cols):
            axs[i, j].axis("off")
            if j < len(chips):
                axs[i, j].imshow(_crop(chips[j]))
        axs[i, 0].set_title(label, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_figures(
    report: Mapping[str, Any],
    out_dir: Path,
    real: Sequence[Chip] = (),
    variants: Mapping[str, Sequence[Chip]] | None = None,
) -> dict[str, Path]:
    """Write every figure the inputs allow and return their paths by name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    arm3_chart(report, out_dir / "arm3.png")
    figures["arm3"] = out_dir / "arm3.png"
    zero_shot_chart(report, out_dir / "zeroshot.png")
    figures["zeroshot"] = out_dir / "zeroshot.png"
    if gauge_chart(report, out_dir / "gauge.png"):
        figures["gauge"] = out_dir / "gauge.png"
    if real and variants:
        shown = {k: sample(v, 60) for k, v in variants.items()}
        distribution_chart(sample(real, 60), shown, out_dir / "distributions.png")
        figures["distributions"] = out_dir / "distributions.png"
        chip_strip(
            {"real": sample(real, 6), **{_label(k): v[:6] for k, v in variants.items()}},
            out_dir / "chips.png",
        )
        figures["chips"] = out_dir / "chips.png"
    return figures


def build_pdf(report: Mapping[str, Any], figures: Mapping[str, Path], path: Path) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as err:
        raise RuntimeError("reportlab is missing; run `make install-report`") from err
    from PIL import Image as PilImage

    s = summary(report)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=0
    )
    h2 = ParagraphStyle(
        "h2",
        parent=ss["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        spaceBefore=12,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.6, spaceAfter=5
    )
    small = ParagraphStyle(
        "s", parent=body, fontSize=8.2, leading=11, textColor=colors.HexColor(MUTE)
    )
    bullet = ParagraphStyle("u", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3)

    def picture(name: str) -> Image:
        w, h = PilImage.open(figures[name]).size
        return Image(str(figures[name]), width=174 * mm, height=174 * mm * h / w)

    pct = f"{s['fraction']:.0%}"
    lo, hi = s["zero_shot_range"]
    story: list[Any] = [
        Paragraph("Realism ablation report", h1),
        Paragraph(f"{s['seeds']} seed(s), real subset {pct}; IoU on the held-out tiles.", small),
        Paragraph("Summary", h2),
    ]
    points = [
        f"Real {pct} (arm 1) scored {s['arm1']:.3f} and real 100% (arm 2) {s['arm2']:.3f}: "
        f"a gap of {s['gap_points']:.2f} points.",
        f"Best synthetic variant: {_label(s['best'])} at {s['best_iou']:.3f} "
        f"({s['best_vs_arm1_points']:+.2f} points against arm 1).",
        f"Synthetic-only models scored {lo:.3f} to {hi:.3f} on real tiles.",
    ]
    if s["q14"]:
        bars = {v["bar"] for v in s["q14"]["variants"].values()}
        points.append(f"Q14 bar: {', '.join(sorted(bars))} (decidable: {s['q14']['decidable']}).")
    story += [Paragraph(t, bullet, bulletText="•") for t in points]
    rows = [["Variant", "Mean", "vs arm 1 (pts)", "Zero-shot"]]
    key = _arm1_key(report)
    arm1 = st.mean(r["arms"][key]["iou"] for r in report["runs"])
    for v in report["variants"]:
        done = [r["variants"][v] for r in report["runs"] if v in r["variants"]]
        if done:
            m = st.mean(d["arm3_iou"] for d in done)
            rows.append(
                [
                    _label(v),
                    f"{m:.3f}",
                    f"{100 * (m - arm1):+.2f}",
                    f"{st.mean(d['zero_shot_iou'] for d in done):.3f}",
                ]
            )
    table = Table(rows, colWidths=[60 * mm, 30 * mm, 40 * mm, 30 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.6),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 8.6),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5d9e0")),
            ]
        )
    )
    story += [Paragraph("Results", h2), table, Spacer(1, 6)]
    for name, title in [
        ("arm3", None),
        ("zeroshot", "Synthetic-only model"),
        ("gauge", "Realism gauge against the benchmark"),
        ("chips", "What the chips look like"),
        ("distributions", "How the distributions differ"),
    ]:
        if name in figures:
            if title:
                story.append(Paragraph(title, h2))
            story.append(picture(name))
    story += [
        Paragraph("Notes", h2),
        Paragraph(
            "Real imagery appears in this report; keep it local (ADR-0008). Differences of about "
            "one point are within seed noise. Edge and noise gauge statistics respond directly "
            "to added noise.",
            body,
        ),
    ]

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTE))
        canvas.drawString(18 * mm, 10 * mm, "Fraktal realism report")
        canvas.drawRightString(192 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()

    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Fraktal realism report",
        author="Jian",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return path
