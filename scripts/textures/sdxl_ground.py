"""Ground and plaster textures from SDXL, with derived and learned normal maps (FRK-126).

Usage: python sdxl_ground.py OUT_DIR [--seeds 4] [--steps 20] [--size 1024]

Per role and seed it writes ``OUT/<role>_<seed>.jpg`` (512 px, seamless through circular padding),
its ``-normal`` and ``-rough`` maps from ``fraktal.pbr``, and for seed 0 a Marigold normals
prediction (``marigold-normals-v1-1``, OpenRAIL++-M) so the two can be compared. Writes
``OUT/ground.json``: seam score, seconds, and the correlation between the derived and the
learned normal (x and y). Generated images, not real photographs: nothing here reads real data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import MarigoldNormalsPipeline, StableDiffusionXLPipeline
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fraktal import pbr  # noqa: E402  (numpy only; no bpy)

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
NORMALS_MODEL = "prs-eth/marigold-normals-v1-1"
STYLE = (
    "top-down flat orthographic photograph, even lighting, seamless tileable texture, no objects"
)
ROLES = {
    "soil_red": "dry red laterite earth, fine dust and small gravel, faint cracks",
    "soil_tan": "dry tan sandy soil, scattered small pebbles, wind-smoothed",
    "soil_scrub": "dry red-brown soil with sparse dead grass tufts and tiny dry shrubs",
    "plaster_sand": "rough sand-coloured cement render on a wall, uneven, slightly stained",
}
NEGATIVE = "text, letters, logo, watermark, people, objects, border, frame, perspective, vignette"


def circular(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.padding_mode = "circular"


def plain(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.padding_mode = "zeros"


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--size", type=int, default=1024)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    circular(pipe.unet)
    circular(pipe.vae)
    print(f"RP_TEXTURE loaded SDXL on {torch.cuda.get_device_name(0)}", flush=True)
    rows: list[dict[str, object]] = []
    images: dict[str, Image.Image] = {}
    for role, prompt in ROLES.items():
        for seed in range(a.seeds):
            t = time.time()
            gen = torch.Generator("cuda").manual_seed(seed)
            big = pipe(
                f"{prompt}, {STYLE}", negative_prompt=NEGATIVE, num_inference_steps=a.steps,
                height=a.size, width=a.size, generator=gen,
            ).images[0]  # fmt: skip
            img = big.resize((512, 512), Image.LANCZOS)
            name = f"{role}_{seed}"
            img.save(a.out / f"{name}.jpg", quality=92)
            rgb = np.asarray(img.convert("RGB"))
            normal, rough = pbr.derive(rgb, 4.0, 24.0, 0.8)
            Image.fromarray(normal).save(a.out / f"{name}-normal.jpg", quality=92)
            Image.fromarray(rough).save(a.out / f"{name}-rough.jpg", quality=92)
            images[name] = img
            rows.append({
                "name": name, "seconds": round(time.time() - t, 1),
                "seam": round(pbr.seam_score(rgb), 3),
            })  # fmt: skip
            print(f"RP_TEXTURE {name} {rows[-1]['seconds']}s seam={rows[-1]['seam']}", flush=True)
    del pipe
    torch.cuda.empty_cache()
    normals = MarigoldNormalsPipeline.from_pretrained(
        NORMALS_MODEL, variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    for role in ROLES:
        name = f"{role}_0"
        out = normals(images[name], num_inference_steps=4, ensemble_size=1)
        vis = normals.image_processor.visualize_normals(out.prediction)[0]
        vis.save(a.out / f"{name}-marigold.jpg", quality=92)
        learned = np.asarray(out.prediction[0], dtype="float32")  # numpy (H, W, 3) in [-1, 1]
        derived = np.asarray(Image.open(a.out / f"{name}-normal.jpg").convert("RGB")).astype(
            "float32"
        )
        derived = derived / 127.5 - 1.0
        derived = (
            np.asarray(
                Image.fromarray(((derived + 1) * 127.5).astype("uint8")).resize(
                    (learned.shape[1], learned.shape[0]), Image.BILINEAR
                )
            ).astype("float32")
            / 127.5
            - 1.0
        )
        corr = {
            "x": round(correlation(derived[..., 0], learned[..., 0]), 3),
            "y": round(correlation(derived[..., 1], learned[..., 1]), 3),
            "y_flipped": round(correlation(derived[..., 1], -learned[..., 1]), 3),
        }
        row = next(r for r in rows if r["name"] == name)
        row["marigold_corr"] = corr
        print(f"RP_TEXTURE {name} marigold corr {corr}", flush=True)
    (a.out / "ground.json").write_text(json.dumps({"model": MODEL, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
