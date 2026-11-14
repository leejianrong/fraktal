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

Nothing here is in git. A fresh clone has no `data/`; this table says what each folder is, where it came
from and how to get it back. "Rebuild" is the command to run on a new machine; "Pod run" means it was made on
RunPod (see [RUNPOD.md](RUNPOD.md)) and only the downloaded artifact lives here. `FRAKTAL_DATA_DIR` moves the
whole tree. Folders marked **irreplaceable** cannot be regenerated from this repo.

### Inputs and real reference imagery (eval only: never committed, trained on, conditioned on or shared)

| Path | What it is | Source and licence | Rebuild |
|---|---|---|---|
| `spacenet2/` | SpaceNet 2 AOI 5 Khartoum, 650 px chips plus the train and test split | AWS Open Data `spacenet-dataset`, CC BY-SA 4.0 (ADR-0008) | `fraktal ingest --what imagery` (5 GB), then `fraktal split --seed 0` |
| `terrain/` | Copernicus GLO-30 DEM and ESA WorldCover clipped to the Khartoum AOI | Copernicus (free, open), ESA WorldCover CC BY 4.0 | `fraktal ingest --what terrain` |
| `worlds/` | `khartoum-skeleton.parquet`, the real-footprint world | Built from the train-split labels only (no test footprints, ADR-0006) | `fraktal generate --seed 0` |
| `eval/ground-reference/` | 247 files: about 120 street-level photos of 11 cities with a `.json` each (source, licence, author, URL) | Wikimedia Commons (own licence per file) and Mapillary (CC BY-SA 4.0, needs `MAPILLARY_TOKEN`) | `uv run python scripts/refboard/pull_ground.py --out data/eval/ground-reference` (the pull is not byte-repeatable: Mapillary results change) |
| `eval/maxar-reference/` | 47 windows of Maxar satellite, pre and post disaster | Maxar Open Data, CC BY-NC 4.0 | `uv run python scripts/refboard/pull_maxar.py --out data/eval/maxar-reference` |
| `eval/open-cities-10/` | 90 drone chips averaged to 0.5 m | Open Cities AI (GFDRR, ODbL-1.0), citation in its `EVAL_ONLY.txt` | `fraktal eval-set` |
| `eval/segmentation/` | Cached class shares from the evaluation segmenter, keyed by pixels and model | OneFormer COCO Swin-L (MIT), see `research/eval-segmentation.md` | Created by `fraktal streets --segment` (needs `make install-seg`; the 840 MB model downloads into `~/.cache/huggingface`; a city takes about 12 minutes on CPU) |
| `eval/reference-board.html`, `eval/side-by-side-*.jpg` | The looking board and sheets | Made from the sets above | `refboard` module (`research/reference-board.md`) |

`EVAL_ONLY.txt` sits beside each real set and states the terms. Keep it there.

### Our own synthetic output (regenerable, but pod time costs money)

| Path | What it is | Rebuild |
|---|---|---|
| `synthetic/`, `dataset_v0/` | V1 benchmark chips (`fraktal synth`) and the first multi-biome Blender batch (6 biomes, 20 tiles each, contact sheet) | `synth`; `dataset_v0` is a pod run of `render3d-batch` per biome via `scripts/runpod/stage0_run.sh` (`MODE=batch`) |
| `review/` | Review sheets, street strips and progress files. `review/street<N>_<biome>/` is one strip render of one commit (`scripts/runpod/city_strips.sh`). `review/city_baseline/` holds the compared sheets of the city-lookalike work, named after the card | Pod run; the commit of each strip is in its `driver.log` |
| `flights/` | Fly-through, dashcam and eye-level renders and the browser demo | Pod runs of `stage0_run.sh` with `MODE=fly` |
| `trees1/`, `trees2/`, `trees3/`, `walls2/` | Pod artifacts behind the tree and wall work (FRK-111, 132, 134) | Pod runs; `strip.jpg` and `before-after.jpg` are the evidence |
| `proc_review/`, `reports/` | Procedural-world review renders and the V3 realism report PDF (`fraktal report`) | `fraktal report` from an ablation result; see `scripts/runpod/ablation.sh` |

Anything under `docs/img/` that matters was copied there from these folders and is in git.

### Other things a new machine needs that git does not hold

- **`.env`** (git-ignored) holds only `MAPILLARY_TOKEN`, a read-only Mapillary client token (free developer
  account). It is read by `pull_ground.py`. Nothing else reads `.env`.
- **`~/.runpod/config.toml`**: the RunPod API key, written by `runpodctl config`. Read by reference in
  `scripts/runpod/stage0_run.sh`. The account balance must be positive or no pod starts.
- **`~/.cache/huggingface`**: the OneFormer model (above).
- **Pandan board 41** (`pandan` CLI, a token from `pandan login`): the work tracker. Not a file.

Put new one-off run output under `data/scratch/<topic>`, not at the top level of `data/`.

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
