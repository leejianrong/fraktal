"""Image-to-3D props trial (FRK-122, ADR-0013): prompt -> SDXL image -> BiRefNet cut-out ->
Step1X-3D geometry -> clean, scale, check. Everything is timed; failures are recorded, not hidden.

Usage (on a GPU pod): python props_trial.py OUT_DIR --step1x /path/to/Step1X-3D [--count 20]

Licences, from the primary texts read on 2026-11-07 (docs/adr/0013): SDXL base 1.0 (OpenRAIL++-M),
BiRefNet (MIT), Step1X-3D geometry (Apache-2.0, DINOv2 encoder). Not used on purpose: Step1X-3D's
texture stage (nvdiffrast, Hunyuan3D tools), diso (CC BY-NC) and any Bria background remover.
Real reference photos are never an input: prompts are written text only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
BIREFNET = "ZhengPeng7/BiRefNet"
STYLE = (
    "single object, centred, three-quarter view, whole object in frame, plain light grey studio "
    "background, soft even lighting, no text, no logos, product photograph"
)
NEGATIVE = (
    "text, letters, logo, watermark, people, multiple objects, cropped, blurry, "
    "cluttered background"
)
# (id, prompt, longest side in metres)
PROPS = [
    ("motorbike-red", "a red commuter motorbike", 1.9),
    (
        "motorbike-scooter",
        "a black motorised moped scooter with a seat, one headlight and two small wheels",
        1.8,
    ),
    ("motorbike-cargo", "a blue motorbike with a cargo box on the back", 2.1),
    ("tuktuk-yellow", "a yellow three-wheeled auto rickshaw with a canopy", 2.7),
    ("tuktuk-green", "a green three-wheeled taxi with a folding roof", 2.7),
    ("tuktuk-blue", "a blue and white three-wheeler passenger taxi", 2.8),
    ("minibus-white", "a white passenger minibus van", 5.5),
    ("minibus-colour", "a colourfully painted minibus", 5.6),
    ("pickup-white", "a white pickup truck", 5.3),
    ("pickup-red", "one single dusty red pickup truck with an open back, only one vehicle", 5.3),
    ("stall-tarp", "a wooden market stall with a blue tarpaulin roof", 2.5),
    ("stall-kiosk", "a small street kiosk with a corrugated metal roof", 2.4),
    ("stall-umbrella", "a fruit stall with a large umbrella", 2.6),
    ("tank-black", "a black plastic rooftop water tank on a metal stand", 1.8),
    ("tank-blue", "a blue cylindrical water storage tank", 1.7),
    ("container-rust", "a rusty shipping container", 6.1),
    ("container-blue", "a blue shipping container", 6.1),
    ("sign-billboard", "one large rectangular billboard on a steel frame, one object only", 4.0),
    ("sign-shop", "a small hand-painted shop signboard on a wooden frame", 2.0),
    ("sign-street", "a green street name sign on a short metal pole, blank sign", 3.0),
]


# Expected length, width, height in metres, for the proportion check in meshqa. Rough, from what
# these things are; the check is a filter and a person still looks at every mesh.
EXPECTED = {
    "motorbike-red": (1.9, 0.8, 1.1), "motorbike-scooter": (1.8, 0.7, 1.1),
    "motorbike-cargo": (2.1, 0.8, 1.3), "tuktuk-yellow": (2.7, 1.4, 1.7),
    "tuktuk-green": (2.7, 1.4, 1.7), "tuktuk-blue": (2.8, 1.4, 1.7),
    "minibus-white": (5.5, 2.0, 2.5), "minibus-colour": (5.6, 2.0, 2.5),
    "pickup-white": (5.3, 1.9, 1.8), "pickup-red": (5.3, 1.9, 1.8),
    "stall-tarp": (2.5, 1.5, 2.3), "stall-kiosk": (2.4, 2.0, 2.4),
    "stall-umbrella": (2.6, 1.6, 2.4),
    "tank-black": (1.8, 1.0, 1.0), "tank-blue": (1.7, 1.1, 1.1),
    "container-rust": (6.1, 2.4, 2.6), "container-blue": (6.1, 2.4, 2.6),
    "sign-billboard": (4.0, 0.8, 3.0), "sign-shop": (2.0, 0.3, 1.5), "sign-street": (3.0, 0.8, 0.5),
}  # fmt: skip


# Second attempt at the props that failed the proportion check (7 flagged by meshqa on the first
# run, plus the cube-shaped umbrella stall). Same size and expected box; the prompts now say what
# shape the object is and show it side-on, which is what its length-to-width ratio shows best.
RETRY = {
    "minibus-white": "a white passenger minibus van seen from the side, long boxy body, "
    "four wheels on the ground, side-on three-quarter view",
    "minibus-colour": "a colourfully painted passenger minibus seen from the side, long body, "
    "four wheels on the ground, side-on three-quarter view",
    "pickup-red": "a single dusty red pickup truck with an open cargo bed, seen from the front "
    "three-quarter, one vehicle only, four wheels on the ground",
    "container-rust": "a rusty corrugated steel shipping container, a long low rectangular box "
    "about three times longer than it is wide, closed doors at one end, side-on view",
    "container-blue": "a blue corrugated steel shipping container, a long low rectangular box "
    "about three times longer than it is wide, closed doors at one end, side-on view",
    "sign-shop": "a flat rectangular wooden shop signboard on two short wooden posts, wide and "
    "thin, blank face, seen from the front three-quarter",
    "sign-street": "a green rectangular street name sign, blank, bolted to the top of a tall "
    "thin grey metal pole, whole pole visible, seen from the front three-quarter",
    "stall-umbrella": "a market fruit stall: a low wooden table with baskets of fruit under a "
    "large round umbrella on a central pole",
}
SEEDS = 2


def image_seed(k: int, base: int = 0) -> int:
    """The SDXL seed of prop ``k``; ``base`` 0 gives the 1000 + k the first trial used."""
    return 1000 + k + base


def mesh_seed(k: int, base: int = 0) -> int:
    """The Step1X-3D seed of prop ``k``; ``base`` 0 gives the first trial's 2000 + k."""
    return 2000 + k + base


def retry_props() -> list[tuple[str, str, float]]:
    """name-s0, name-s1, ... for every retried prop, with the new prompt and the old size."""
    sizes = {n: s for n, _, s in PROPS}
    return [(f"{n}-s{k}", RETRY[n], sizes[n]) for n in RETRY for k in range(SEEDS)]


def expected(name: str) -> tuple[float, float, float] | None:
    return EXPECTED.get(name) or EXPECTED.get(name.rsplit("-s", 1)[0])


def vram_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def log(**fields: object) -> None:
    print("fraktal-log " + json.dumps(fields, sort_keys=True), file=sys.stderr, flush=True)


def make_images(out: Path, props: list[tuple[str, str, float]], seed: int = 0) -> dict[str, float]:
    import torch
    from diffusers import StableDiffusionXLPipeline

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    load = time.perf_counter() - t0
    per = {}
    for k, (name, prompt, _) in enumerate(props):
        t = time.perf_counter()
        image = pipe(
            prompt=f"{prompt}, {STYLE}", negative_prompt=NEGATIVE, num_inference_steps=30,
            guidance_scale=7.0, generator=torch.Generator("cuda").manual_seed(image_seed(k, seed)),
        ).images[0]  # fmt: skip
        image.save(out / "images" / f"{name}.png")
        per[name] = round(time.perf_counter() - t, 2)
        log(event="props_image", name=name, seconds=per[name])
    stats = {"load_s": round(load, 1), "per_image_s": per, "vram_gb": vram_gb()}
    del pipe
    torch.cuda.empty_cache()
    return stats


def cut_out(out: Path, props: list[tuple[str, str, float]]) -> dict[str, float]:
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForImageSegmentation.from_pretrained(BIREFNET, trust_remote_code=True)
    model = model.to("cuda").half().eval()
    prep = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    per = {}
    for name, _, _ in props:
        t = time.perf_counter()
        image = Image.open(out / "images" / f"{name}.png").convert("RGB")
        with torch.no_grad():
            pred = model(prep(image).unsqueeze(0).to("cuda").half())[-1].sigmoid().cpu()[0, 0]
        mask = transforms.ToPILImage()(pred.float()).resize(image.size)
        rgba = image.copy()
        rgba.putalpha(mask)
        box = rgba.split()[-1].point(lambda v: 255 if v > 128 else 0).getbbox()
        if box:  # crop to the object with a margin, pad to a square
            x0, y0, x1, y1 = box
            pad = int(0.08 * max(x1 - x0, y1 - y0))
            rgba = rgba.crop((max(x0 - pad, 0), max(y0 - pad, 0), x1 + pad, y1 + pad))
        side = max(rgba.size)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
        square.save(out / "cutouts" / f"{name}.png")
        per[name] = round(time.perf_counter() - t, 2)
    stats = {"per_image_s": per, "vram_gb": vram_gb()}
    del model
    torch.cuda.empty_cache()
    return stats


def make_meshes(
    out: Path, props: list[tuple[str, str, float]], step1x: Path, seed: int = 0
) -> dict[str, object]:
    import torch
    import trimesh

    sys.path.insert(0, str(step1x))
    from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline

    sys.path.insert(0, str(Path(__file__).parent))
    import meshqa

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = Step1X3DGeometryPipeline.from_pretrained(
        "stepfun-ai/Step1X-3D", subfolder="Step1X-3D-Geometry-1300m"
    ).to("cuda")
    load = time.perf_counter() - t0
    log(event="props_geometry_loaded", seconds=round(load, 1), vram_gb=vram_gb())
    per, reports, rows = {}, {}, []
    for k, (name, prompt, size_m) in enumerate(props):
        t = time.perf_counter()
        try:
            generated = pipe(
                str(out / "cutouts" / f"{name}.png"),  # a path: check_inputs rejects a PIL image
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.Generator(device=pipe.device).manual_seed(mesh_seed(k, seed)),
                force_remove_background=False,
                do_remove_floater=False,
                do_shade_smooth=False, surface_extractor_type="mc", do_reduce_face=False,
                max_facenum=200000,
            )  # fmt: skip
            raw = generated.mesh[0]
            if isinstance(raw, trimesh.Scene):
                raw = trimesh.util.concatenate(list(raw.geometry.values()))
            seconds = round(time.perf_counter() - t, 2)
            raw.export(out / "meshes_raw" / f"{name}.glb")  # kept even if our own QA fails
            final, report = meshqa.process(raw, size_m, expected(name))
            meshqa.to_gltf(final).export(out / "meshes" / f"{name}.glb")  # glTF is y up
            report.update(prompt=prompt, gen_s=seconds)
            rows.append(
                (
                    f"{name}  {report['faces']} faces  {'ok' if report['pass'] else 'FAIL'}",
                    meshqa.render_views(final),
                )
            )
        except Exception as err:  # noqa: BLE001 - a failed prop is a result
            seconds = round(time.perf_counter() - t, 2)
            report = {
                "pass": False,
                "reasons": [f"error: {type(err).__name__}: {err}"[:200]],
                "gen_s": seconds,
            }
        per[name], reports[name] = seconds, report
        log(
            event="props_mesh",
            name=name,
            seconds=seconds,
            ok=report["pass"],
            reasons=report["reasons"],
        )
    if rows:
        meshqa.sheet(rows).save(out / "sheet_meshes.jpg", quality=88)
    return {"load_s": round(load, 1), "per_mesh_s": per, "vram_gb": vram_gb(), "qa": reports}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--step1x", type=Path, required=True)
    parser.add_argument("--count", type=int, default=len(PROPS))
    parser.add_argument("--retry", action="store_true", help="the failed props, new prompts")
    args = parser.parse_args()
    for sub in ("images", "cutouts", "meshes", "meshes_raw"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    props = retry_props() if args.retry else PROPS[: args.count]
    result: dict[str, object] = {"prompts": {n: p for n, p, _ in props}}
    t0 = time.perf_counter()
    for stage, fn in (
        ("images", lambda: make_images(args.out, props)),
        ("cutouts", lambda: cut_out(args.out, props)),
        ("meshes", lambda: make_meshes(args.out, props, args.step1x)),
    ):
        t = time.perf_counter()
        try:
            result[stage] = fn()
        except Exception as err:  # noqa: BLE001
            result[stage] = {"error": f"{type(err).__name__}: {err}"[:500]}
            log(event="props_stage_failed", stage=stage, error=str(err)[:300])
            break
        result[f"{stage}_total_s"] = round(time.perf_counter() - t, 1)
    result["total_s"] = round(time.perf_counter() - t0, 1)
    (args.out / "trial.json").write_text(json.dumps(result, indent=1) + "\n")
    print("RP_REPORT_BEGIN")
    print(json.dumps({k: v for k, v in result.items() if k != "prompts"}, indent=1))
    print("RP_REPORT_END")
    return 0 if "error" not in str(result.get("meshes", ""))[:200] else 1


if __name__ == "__main__":
    raise SystemExit(main())
