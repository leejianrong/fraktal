import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("reportlab")

from fraktal import cli  # noqa: E402
from fraktal.report import build_figures, build_pdf, load_ablation, summary  # noqa: E402


def entry(iou: float, zero: float) -> dict[str, Any]:
    gauge = {"colour_w1": 0.1, "edge_energy_gap": 0.5, "noise_gap": 0.3}
    return {"arm3_iou": iou, "zero_shot_iou": zero, "gauge": gauge}


def ablation(fraction: float = 0.1) -> dict[str, Any]:
    key = f"real_{fraction * 100:g}pct"
    runs = [
        {
            "seed": s,
            "arms": {key: {"iou": 0.60 + s / 100}, "real_100pct": {"iou": 0.65}},
            "variants": {"v0_flat": entry(0.58 + s / 100, 0.0), "v3_texture": entry(0.62, 0.3)},
        }
        for s in (0, 1)
    ]
    return {
        "status": "complete",
        "variants": ["v0_flat", "v3_texture"],
        "fraction": fraction,
        "runs": runs,
    }


def chip(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out: np.ndarray = rng.integers(0, 255, (3, 64, 64), dtype=np.uint8)
    return out


def test_summary_is_computed_from_the_data() -> None:
    s = summary(ablation())
    assert s["arm1"] == pytest.approx(0.605) and s["arm2"] == pytest.approx(0.65)
    assert s["gap_points"] == pytest.approx(4.5)
    assert s["best"] == "v3_texture" and s["zero_shot_range"] == (0.0, 0.3)


def test_summary_reads_the_arm_named_for_a_scarce_fraction() -> None:
    assert summary(ablation(0.01))["fraction"] == 0.01


def test_figures_and_pdf_are_written_without_chips(tmp_path: Path) -> None:
    figures = build_figures(ablation(), tmp_path / "f")
    assert set(figures) == {"arm3", "zeroshot", "gauge"}
    pdf = build_pdf(ablation(), figures, tmp_path / "r.pdf")
    assert pdf.read_bytes().startswith(b"%PDF")


def test_distribution_and_chip_figures_need_real_and_variant_chips(tmp_path: Path) -> None:
    real = [chip(i) for i in range(4)]
    variants = {"v0_flat": [chip(10 + i) for i in range(4)]}
    figures = build_figures(ablation(), tmp_path / "f", real, variants)
    assert {"distributions", "chips"} <= set(figures)
    assert all(p.stat().st_size > 1000 for p in figures.values())


def test_an_ablation_without_runs_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"runs": []}))
    with pytest.raises(ValueError, match="no finished runs"):
        load_ablation(path)


def test_report_command_writes_a_pdf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps(ablation()))
    code = cli.main(["report", "--ablation", str(path), "--data-dir", str(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert Path(out["data"]["pdf"]).read_bytes().startswith(b"%PDF")


def test_report_command_fails_cleanly_on_a_missing_file(tmp_path: Path) -> None:
    assert (
        cli.main(["report", "--ablation", str(tmp_path / "none.json"), "--data-dir", str(tmp_path)])
        == 4
    )
