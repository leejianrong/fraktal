"""Machine-readable progress lines for long jobs, written to stderr.

Each line is ``fraktal-log {json}``. RunPod's log API timestamps every line, so no clock is read
here, and stdout stays reserved for the JSON result envelope. Grep ``fraktal-log`` to follow a run.
"""

from __future__ import annotations

import json
import sys
from typing import Any

PREFIX = "fraktal-log "


def emit(event: str, **fields: Any) -> None:
    line = PREFIX + json.dumps({"event": event, **fields}, sort_keys=True)
    print(line, file=sys.stderr, flush=True)
