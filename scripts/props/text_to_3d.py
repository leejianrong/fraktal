"""Text to 3D props (FRK-123, ADR-0013): a list of prompts in, GLB meshes out, each with a sidecar.

Usage (on a GPU pod):
    python text_to_3d.py OUT_DIR --prompts prompts.json --step1x /path/to/Step1X-3D [--seed 0]

``prompts.json`` is a list of ``{"id", "prompt", "size_m"}`` objects, with an optional
``"expected_lwh"`` (length, width, height in metres) for the proportion check. Each prop goes
prompt -> SDXL image -> BiRefNet cut-out -> Step1X-3D geometry -> clean, scale, check
(``props_trial`` does the work). Every mesh gets ``meshes/<id>.json`` and the run gets
``manifest.json``: the prompt, both seeds, the QA verdict, the seconds each stage took, and the
model revisions and library versions, so an asset found to be problematic later can be traced to
the models that made it (ADR-0013 decision 5). A fixed ``--seed`` and the same models give the same
meshes. Only models ADR-0013 allows are used; nothing here reads or prints a secret.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import props_trial  # noqa: E402  (heavy imports inside its functions only)

MODELS = {
    "image": props_trial.SDXL,
    "cutout": props_trial.BIREFNET,
    "geometry": "stepfun-ai/Step1X-3D",
}
PACKAGES = ("torch", "diffusers", "transformers", "trimesh", "numpy")


class PromptError(ValueError):
    pass


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """The prompt list, checked: unique ids, non-empty prompts, positive sizes."""
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise PromptError("the prompt file must be a non-empty list")
    seen: set[str] = set()
    for row in rows:
        for key in ("id", "prompt", "size_m"):
            if key not in row:
                raise PromptError(f"a prompt is missing {key!r}: {row}")
        name = str(row["id"])
        if not name or "/" in name or name in seen:
            raise PromptError(f"prompt id {name!r} is empty, has a slash or is repeated")
        if not str(row["prompt"]).strip():
            raise PromptError(f"prompt {name!r} is empty")
        if not float(row["size_m"]) > 0:
            raise PromptError(f"prompt {name!r} needs a positive size_m")
        seen.add(name)
    return rows


def build_records(
    rows: list[dict[str, Any]], seed: int, images: dict[str, Any], cutouts: dict[str, Any],
    meshes: dict[str, Any],
) -> list[dict[str, Any]]:  # fmt: skip
    """One record per prompt: what was asked, the seeds, the stage times and the QA verdict."""
    out = []
    for k, row in enumerate(rows):
        name = str(row["id"])
        qa = meshes.get("qa", {}).get(name, {})
        record = {
            "id": name,
            "prompt": row["prompt"],
            "size_m": float(row["size_m"]),
            "image_seed": props_trial.image_seed(k, seed),
            "mesh_seed": props_trial.mesh_seed(k, seed),
            "image_s": images.get("per_image_s", {}).get(name),
            "cutout_s": cutouts.get("per_image_s", {}).get(name),
            "mesh_s": meshes.get("per_mesh_s", {}).get(name),
            "pass": bool(qa.get("pass", False)),
            "reasons": qa.get("reasons", ["no mesh"]),
            "faces": qa.get("faces"),
        }
        record["seconds"] = round(
            sum(record[key] or 0.0 for key in ("image_s", "cutout_s", "mesh_s")), 2
        )
        out.append(record)
    return out


def summarise(records: list[dict[str, Any]], usd_per_hour: float | None) -> dict[str, Any]:
    """Per-prop seconds and, given the pod's hourly rate, the generation cost per prop.

    Model loading and installing are not in it: they are paid once per run, not per prop.
    """
    n = len(records)
    seconds = sum(r["seconds"] for r in records)
    mean = seconds / n if n else 0.0
    return {
        "props": n,
        "passed_qa": sum(r["pass"] for r in records),
        "mean_seconds_per_prop": round(mean, 2),
        "usd_per_hour": usd_per_hour,
        "usd_per_prop": None if usd_per_hour is None else round(mean / 3600 * usd_per_hour, 5),
    }


def model_versions() -> dict[str, Any]:
    """Hugging Face revision hashes of the three models, the Step1X-3D code commit and packages."""
    revisions: dict[str, str] = {}
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for role, repo in MODELS.items():
            revisions[role] = f"{repo}@{api.model_info(repo).sha}"
    except Exception as err:  # noqa: BLE001 - a missing revision is recorded, not fatal
        revisions["error"] = f"{type(err).__name__}: {err}"[:200]
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return {
        "models": revisions,
        "step1x_code_commit": os.environ.get("STEP1X_SHA", "unknown"),
        "packages": versions,
        "geometry_checkpoint": "Step1X-3D-Geometry-1300m",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--step1x", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="added to every per-prop seed")
    parser.add_argument(
        "--usd-per-hour", type=float, default=None, help="the pod's rate, for cost per prop"
    )
    args = parser.parse_args()
    rows = load_prompts(args.prompts)
    for sub in ("images", "cutouts", "meshes", "meshes_raw"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    props = [(str(r["id"]), str(r["prompt"]), float(r["size_m"])) for r in rows]
    expected = {str(r["id"]): tuple(r["expected_lwh"]) for r in rows if r.get("expected_lwh")}
    props_trial.EXPECTED.update(expected)  # the proportion check reads this table
    stages: dict[str, Any] = {}
    t0 = time.perf_counter()
    for stage, fn in (
        ("images", lambda: props_trial.make_images(args.out, props, args.seed)),
        ("cutouts", lambda: props_trial.cut_out(args.out, props)),
        ("meshes", lambda: props_trial.make_meshes(args.out, props, args.step1x, args.seed)),
    ):
        t = time.perf_counter()
        try:
            stages[stage] = fn()
        except Exception as err:  # noqa: BLE001 - say which stage died and keep what exists
            stages[stage] = {"error": f"{type(err).__name__}: {err}"[:500]}
            props_trial.log(event="text_to_3d_stage_failed", stage=stage, error=str(err)[:300])
            break
        stages[f"{stage}_total_s"] = round(time.perf_counter() - t, 1)
    records = build_records(
        rows, args.seed, stages.get("images", {}), stages.get("cutouts", {}),
        stages.get("meshes", {}),
    )  # fmt: skip
    versions = model_versions()
    for record in records:
        (args.out / "meshes" / f"{record['id']}.json").write_text(
            json.dumps({**record, **versions}, indent=1) + "\n"
        )
    manifest = {
        "seed": args.seed,
        "total_s": round(time.perf_counter() - t0, 1),
        "stage_totals_s": {k: v for k, v in stages.items() if k.endswith("_total_s")},
        "stage_loads_s": {k: v.get("load_s") for k, v in stages.items() if isinstance(v, dict)},
        "vram_gb": {k: v.get("vram_gb") for k, v in stages.items() if isinstance(v, dict)},
        "summary": summarise(records, args.usd_per_hour),
        "props": records,
        **versions,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("RP_REPORT_BEGIN")
    print(json.dumps({k: v for k, v in manifest.items() if k != "props"}, indent=1))
    print("RP_REPORT_END")
    failed = [k for k, v in stages.items() if isinstance(v, dict) and "error" in v]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
