"""Gauge numbers, distribution plots and chip strips for synthetic chip sets against real chips.

Usage: python review_figs.py REAL_CHIPS_DIR OUT_DIR LABEL=CHIPS_DIR [LABEL=CHIPS_DIR ...]
Writes OUT_DIR/gauge.json, distributions.png and chips.png. Needs the `report` extra.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fraktal import chips, realism, report


def _load(path: Path, real: bool = False) -> list[realism.Chip]:
    data = chips.load_splits(path)[0] if real else chips.load_all(path)
    return [data.get(i)[0] for i in range(len(data))]


def main(argv: list[str]) -> int:
    real_dir, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    real = realism.sample(_load(real_dir, real=True), 60)
    sets = {}
    for spec in argv[2:]:
        label, _, path = spec.partition("=")
        sets[label] = realism.sample(_load(Path(path)), 60)
    gauge = {label: realism.compare(real, cs) for label, cs in sets.items()}
    (out / "gauge.json").write_text(json.dumps(gauge, indent=2) + "\n")
    report.distribution_chart(real, sets, out / "distributions.png")

    def spread(cs: list[realism.Chip]) -> list[realism.Chip]:
        return cs[:: max(1, len(cs) // 6)][:6]  # across regions, not just the first one

    strip = {"real": real[:6], **{k: spread(v) for k, v in sets.items()}}
    report.chip_strip(strip, out / "chips.png")
    for label, g in gauge.items():
        print(label, json.dumps(g["distances"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
