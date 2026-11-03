"""Summarise `runpodctl pod logs` JSON-lines from stdin: stage, heartbeat, progress, problems."""

from __future__ import annotations

import contextlib
import json
import sys

PROGRESS = ("train_progress", "download_progress", "chips_progress", "synth_region_done")


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

    markers = ("Traceback", "STAGE failed", "self-terminating")
    bad = [x for x in lines if any(m in x for m in markers)]
    if bad:
        print("PROBLEM:")
        for x in bad[-5:]:
            print("  " + x[:200])


if __name__ == "__main__":
    main()
