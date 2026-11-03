"""Summarise `runpodctl pod logs` JSON-lines from stdin: stage, heartbeat, progress, problems."""

from __future__ import annotations

import contextlib
import json
import sys

PROGRESS = (
    "train_progress",
    "download_progress",
    "chips_progress",
    "synth_region_done",
    "ablation_wait",
)


def _gap_s(earlier: str, later: str) -> float:
    from datetime import datetime

    def parse(ts: str) -> datetime:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")[:32])

    try:
        return (parse(later) - parse(earlier)).total_seconds()
    except ValueError:
        return 0.0


def main() -> None:
    rows = []
    for raw in sys.stdin:
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        rows.append((d.get("ts", ""), d.get("line", "")))
    lines = [text for _, text in rows]
    print(next((x for x in reversed(lines) if x.startswith("STAGE ")), "STAGE (none yet)"))
    print(next((x for x in reversed(lines) if x.startswith("hb ")), "hb (none yet)"))
    print("last log line at", rows[-1][0] if rows else "?")

    events = []
    for line in lines:
        if line.startswith("fraktal-log "):
            with contextlib.suppress(ValueError):
                events.append(json.loads(line.removeprefix("fraktal-log ")))
    latest = {}
    for e in events:
        if e["event"] in PROGRESS:
            latest[e.get("label", e["event"])] = e
    for key, e in latest.items():
        detail = {k: v for k, v in e.items() if k not in ("event", "label")}
        print(f"  {e['event']:<18} {key:<28} {detail}")
    for e in (e for e in events if e["event"] == "arm_done"):
        print(f"  arm_done {e['arm']}: iou={e['iou']}")

    for e in (e for e in events if e["event"] == "ablation_result"):
        detail = {k: v for k, v in e.items() if k != "event"}
        print(f"  result {detail}")
    for x in [x for x in lines if x.startswith(("synth_start", "synth_done"))][-3:]:
        print("  " + x)

    warns = [x for x in lines if x.startswith("WARN ")]
    if warns:
        print(f"WARNINGS ({len(warns)}, latest first):")
        for x in reversed(warns[-3:]):
            print("  " + x[:160])
    heartbeats = [ts for ts, text in rows if text.startswith("hb ")]
    if (
        heartbeats
        and rows
        and rows[-1][0] > heartbeats[-1]
        and _gap_s(heartbeats[-1], rows[-1][0]) > 180
    ):
        print(
            f"WARNING: last heartbeat {heartbeats[-1]} is far behind the newest line {rows[-1][0]}"
        )

    markers = (
        "Traceback",
        "STAGE failed",
        "STAGE finished with FAILURE",
        "ERROR ",
        "self-terminating",
    )
    bad = [x for x in lines if any(m in x for m in markers)]
    if bad:
        print("PROBLEM:")
        for x in bad[-5:]:
            print("  " + x[:200])


if __name__ == "__main__":
    main()
