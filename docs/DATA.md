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
| `scratch/` | Pod outputs behind the evidence images: props trials, low-sun and probe runs, PBR A/B, texture and road views | Yes, once the doc image is committed |
| `flights/` | Fly-through, dashcam and eye-level renders | Regenerable |
| `old-synth/` | Early `synth_v*` training sets | Ask first |

Put new one-off run output under `scratch/<topic>`, not at the top level of `data/`.
