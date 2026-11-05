"""Texture trial (FRK-73): SDXL base makes seamless material tiles, timed, for five roles.

Usage: python sdxl_trial.py OUT_DIR [--seeds 4] [--steps 20] [--size 1024]

Circular convolution padding makes the output wrap, so a tile repeats without a seam. Writes
``OUT/<role>_<seed>.jpg`` (512 px), ``OUT/<role>_3x3.jpg`` (the first seed tiled 3x3),
``OUT/sheet.jpg`` and ``OUT/trial.json`` (seconds per image, a seam score, the model revision).
Outputs of SDXL (CreativeML Open RAIL++-M): see docs/research/texture-model-spike.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
STYLE = (
    "top-down flat orthographic photograph, even lighting, seamless tileable texture, no objects"
)
ROLES = {
    "plaster_wall": "weathered cement plaster wall, sun-bleached beige, hairline cracks",
    "rough_concrete": "rough poured concrete surface, grey, pitted, stained",
    "corrugated_tin": "old corrugated galvanised steel roof sheets, rust spots, parallel ridges",
    "clay_roof_tile": "terracotta clay roof tiles in overlapping rows, weathered, moss patches",
    "weathered_paint": "flaking painted wall, faded blue paint over bare plaster",
}
NEGATIVE = "text, letters, logo, watermark, people, objects, border, frame, perspective, vignette"


def circular(module: torch.nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.padding_mode = "circular"


def seam_score(image: np.ndarray) -> float:
    """Wrap-around step over the typical step: about 1 means no visible seam."""
    grey = image.astype("float32").mean(axis=2)
    wrap = np.abs(grey[:, 0] - grey[:, -1]).mean()
    inner = np.abs(np.diff(grey, axis=1)).mean()
    wrap_v = np.abs(grey[0] - grey[-1]).mean()
    inner_v = np.abs(np.diff(grey, axis=0)).mean()
    return float(((wrap / inner) + (wrap_v / inner_v)) / 2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--size", type=int, default=1024)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    circular(pipe.unet)
    circular(pipe.vae)
    load_s = round(time.time() - t0, 1)
    print(f"RP_TEXTURE loaded model in {load_s}s on {torch.cuda.get_device_name(0)}", flush=True)
    rows: list[dict[str, object]] = []
    sheet_rows = []
    for role, prompt in ROLES.items():
        row_imgs = []
        for seed in range(a.seeds):
            t = time.time()
            gen = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                f"{prompt}, {STYLE}", negative_prompt=NEGATIVE, num_inference_steps=a.steps,
                height=a.size, width=a.size, generator=gen,
            ).images[0]  # fmt: skip
            seconds = round(time.time() - t, 2)
            small = image.resize((512, 512), Image.LANCZOS)
            small.save(a.out / f"{role}_{seed}.jpg", quality=92)
            arr = np.asarray(small)
            mean_rgb = [round(float(v), 1) for v in arr.reshape(-1, 3).mean(0)]
            seam = round(seam_score(arr), 3)
            rows.append(
                {"role": role, "seed": seed, "seconds": seconds, "seam": seam, "mean_rgb": mean_rgb}
            )
            row_imgs.append(arr[::2, ::2])
            if seed == 0:
                tiled = np.tile(arr, (3, 3, 1))
                Image.fromarray(tiled).resize((768, 768), Image.LANCZOS).save(
                    a.out / f"{role}_3x3.jpg", quality=90
                )
            print(f"RP_TEXTURE {role} seed {seed} {seconds}s seam {rows[-1]['seam']}", flush=True)
        sheet_rows.append(np.concatenate(row_imgs, axis=1))
    Image.fromarray(np.concatenate(sheet_rows, axis=0)).save(a.out / "sheet.jpg", quality=90)
    total = round(time.time() - t0, 1)
    gen_s = [float(r["seconds"]) for r in rows]  # type: ignore[arg-type]
    result = {
        "model": MODEL, "steps": a.steps, "size": a.size, "load_s": load_s, "total_s": total,
        "mean_s_per_image": round(sum(gen_s) / len(gen_s), 2), "images": rows,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }  # fmt: skip
    (a.out / "trial.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "RP_TEXTURE_DONE",
        json.dumps({k: result[k] for k in ("load_s", "total_s", "mean_s_per_image")}),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
