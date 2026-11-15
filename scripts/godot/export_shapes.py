#!/usr/bin/env python
"""Write godot/shapes/agent_shapes.json from the part tables in blender_scene.py (FRK-94).

The Blender script is the one place the moving cars, people and drones are defined. This reads its
CAR_PARTS, PERSON_PARTS and DRONE_BOXES (without importing it: it needs bpy) so the Godot viewer
draws the same shapes. A test fails if the committed JSON differs from what this writes.

Each part is {"role", "box": [shift, half_l, half_w, z0, z1], "opt": {top_l, top_w, top_shift,
z1_front, side}}; the car frame is x forward, y left, z up, in metres.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "fraktal" / "blender_scene.py"
OUT = ROOT / "godot" / "shapes" / "agent_shapes.json"
WANTED = ("CAR_PARTS", "PERSON_PARTS", "DRONE_BOXES")


def shapes() -> dict[str, Any]:
    keep = [
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in WANTED for t in node.targets)
    ]
    scope: dict[str, Any] = {}
    exec(compile(ast.Module(keep, []), str(SOURCE), "exec"), scope)  # noqa: S102

    def part(role: str, box: tuple[Any, ...]) -> dict[str, Any]:
        return {"role": role, "box": list(box[:5]), "opt": dict(box[5]) if len(box) > 5 else {}}

    return {
        "car": [part(role, box) for role, box in scope["CAR_PARTS"]],
        "person": [part(role, box) for role, box in scope["PERSON_PARTS"]],
        "drone": [
            part("body" if i == 0 else "arm", box) for i, box in enumerate(scope["DRONE_BOXES"])
        ],
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(shapes(), indent=1, sort_keys=True) + "\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
