# Where things live

## `docs/img/` (committed, our own renders only)

Evidence images, one folder per topic. Link them from the doc that makes the claim, with a relative path
(`../img/<topic>/<file>`). Never commit real reference imagery here.

| Folder | What goes in it |
|---|---|
| `pipeline-3d/` | The Blender path: batches, labels, heights, surfaces, typologies (ADR-0010) |
| `realism/` | Before/after and gap evidence for the realism work (`research/realism-gap-analysis.md`) |
| `moving-agents/` | Traffic, people, dashcam and eye-level views |
| `generative-assets/` | Prop, mesh and texture generation trials |
| `biomes/` | Biome overviews |

New topic: add a folder and a row here. Name files for what they show, not for the card.

## `data/` (git-ignored, local only)

| Path | What it is | Safe to delete? |
|---|---|---|
| `eval/` | Real reference imagery and side-by-side sheets. Eval only, never committed or shared | No |
| `spacenet2/`, `synthetic/`, `dataset_v0/`, `terrain/`, `worlds/` | Inputs and built datasets that docs and scripts name | No |
| `review/`, `proc_review/`, `reports/` | Review figures and reports | Regenerable |
| `scratch/` | Deleted 2026-11-09 (see below). New one-off pod output goes here | Yes, once the doc image is committed |
| `flights/` | Fly-through, dashcam and eye-level renders | Regenerable |
| `old-synth/` | Deleted 2026-11-09 (see below) | n/a |

Put new one-off run output under `scratch/<topic>`, not at the top level of `data/`.

## Removed on 2026-11-09

Deleted to free 5 GB. Everything here can be regenerated from a commit, and the evidence images the docs
use were already committed under `docs/img/`.

### `old-synth/`: the V3 realism ablation sets (FRK-18)

Nine sets, each 3 regions of 11x11 tiles (135 chips), rendered over the real Khartoum building skeleton by the
V3 realism ablation (FRK-18 onwards; results in `docs/adr/0006` and `data/reports/fraktal-v3-realism-report.pdf`,
driver `scripts/runpod/ablation.sh`). The variant definitions are in `src/fraktal/ablation.py`, which is the
source of truth. All used seed 0, sun azimuth 135 and elevation 50. The settings below come from each set's
`render.json`:

| Set | Settings |
|---|---|
| `synth_v3a` | tone `[0.66, 0.94, 0.31]`, variation 0.15, no texture |
| `synth_v3b` | as v3a, texture 0.3 |
| `synth_v3c` | as v3a, texture 0.7 |
| `synth_v3d` | texture 0.7, haze 0.1, penumbra 0.3 m |
| `synth_v3e` | texture 0.7, sensor only: noise sigma 8, PSF sigma 0.8 px |
| `synth_v5` | `v5_atmosphere`: texture 0.7, sensor, haze 0.1, penumbra 0.3 m |
| `synth_v7` | `v7_extremes`: v5 plus ambient 0.18, contrast 1.25, bright roofs 0.15, blank prob 0.25 |
| `synth_v8` | `v8_objects`: v7 plus objects 1.0, roof detail 0.12 |
| `synth_v9` | `v9_randomised_wide`: v8 with the extremes drawn per region from the wide ranges, palette on |

To rebuild one, check out the commit of that ablation run and run the `synth` step with those values.

### `scratch/`: pod outputs behind the realism evidence

Each folder is the pod artifact of one RunPod run (`driver.log`, `pod.log` and the renders). The runs are named in
`research/realism-gap-analysis.md` and `research/generative-assets-plan.md`, and the scripts that made them are
in `scripts/`.

| Folders | Made by | Evidence image |
|---|---|---|
| `props_trial*`, `propgen*`, `props_retry` | `scripts/props/props_trial.py`, `propgen_*.py`; `scripts/runpod/props.sh`, `propgen.sh` | `img/generative-assets/` |
| `propscene*` | `render3d-batch` with `PROPS=1` via `scripts/runpod/stage0_run.sh` | `img/realism/props-in-scene.jpg` |
| `roofs*`, `walls` | `stage0_run.sh` | `img/realism/roofs-drone-scale.jpg`, `compound-walls.jpg` |
| `lowsun*`, `probe*` | `stage0_run.sh` with `SUN_ELEV`; `scripts/props/material_probe.py` via `probe_run.sh` | `img/realism/tin-ribs-low-sun.jpg` |
| `pbr_with`, `pbr_without` | `stage0_run.sh` before and after the PBR maps | `img/realism/pbr-ab.jpg` |
| `groundtex`, `detail` | `scripts/textures/sdxl_ground.py` via `ground_textures_run.sh` | `img/realism/ground-textures.jpg` |
| `roadviews` | `scripts/roads_preview.py` | `img/realism/organic-roads.jpg` |

Remote branches for the merged PRs (#115 to #125) were deleted at the same time.
