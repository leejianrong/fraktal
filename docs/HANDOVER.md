# Handover: picking this project up on a new machine

Written 2026-11-14 for a developer or agent starting on another machine (for example one with a GPU).
Read [AGENTS.md](../AGENTS.md) first for the rules, then this page for state and setup. Where this page and
the code disagree, trust the code and fix this page.

## 1. Set up (about 15 minutes)

```bash
git clone https://github.com/leejianrong/fraktal.git && cd fraktal
make install          # uv sync --frozen; Python 3.11 or newer
make hooks            # pre-push runs `make check-fast` (ruff + mypy --strict)
make test-fast        # sanity: unit tests without the slow ones
uv run fraktal --help
```

Needed tools and accounts. None of the secrets are in git; see [DATA.md](DATA.md) for the full list.

| What | For | How to get it |
|---|---|---|
| `gh` CLI, logged in | PRs and watching CI | `gh auth login` |
| `pandan` CLI, logged in, board 41 (key `FRK`) | The work tracker. Cards are the source of truth for what is done | `pandan login`; `pandan` prints the board |
| `runpodctl` and `~/.runpod/config.toml` | Every Blender render and any GPU job. The API key is read by reference, never printed | `runpodctl config`; top up the balance (see section 3) |
| `MAPILLARY_TOKEN` in `.env` | Only to re-pull the ground photos | Free Mapillary developer account |
| `make install-seg` | Only to run `fraktal streets --segment` (torch and transformers) | Model downloads on first use |
| Blender | **Not needed locally.** It runs on pods; `blender_scene.py` runs inside Blender's own Python | See `scripts/runpod/stage0.sh` |

The old machine had no GPU and 8 GB of RAM, so all heavy work went to RunPod CPU pods. With a local GPU you can
render Blender scenes directly (`fraktal render3d`), but keep determinism and the "pods render pushed commits"
habit for anything you compare across runs. GPU training is a separate matter: see the standing rules.

**WSL with the Windows Blender (checked 2026-11-14, Blender 5.0.0, RTX 3080 Ti Laptop):** set
`FRAKTAL_BPY_PYTHON=$PWD/scripts/wsl/winblender.sh` and pass `--device gpu` to `fraktal render3d`. Cycles uses OptiX on
the Windows driver, which does not work from inside WSL. A 15-frame street strip at 32 samples took 90 s on the GPU and
140 s on the CPU; the beauty pass is 3x faster but the labels pass (about 60 s) is the same either way. GPU pixels differ
from CPU pixels, and the pods' `bpy` version is not pinned to this Blender, so do not compare across them. In WSL, torch
needs a CUDA 12.x build: the lockfile's `cu132` build segfaults on `torch.cuda.is_available()` with this driver.

Rebuild the data you need with the table in [DATA.md](DATA.md). For the city work you need only
`data/eval/ground-reference` (pull it with `scripts/refboard/pull_ground.py`); the segmentation cache refills itself.

## 2. Where the project stands

Goal: the Blender world generator should produce realistic lookalikes of 11 cities (Accra, Cebu, Dar es Salaam,
Jakarta, Kampala, Khartoum, Khartoum-Bahri, Mandalay, Manila, Manila-Quezon, Nairobi), judged against real street
photos in `data/eval/ground-reference` (evaluation only). Three pilots first: Khartoum (`khartoum_city`),
Manila-Quezon (`manila_quezon`), Nairobi centre (`nairobi_centre`). Per-city profiles and measured photo statistics are in
[research/city-profiles.md](research/city-profiles.md); the harness is `fraktal streets` ([research/eval-segmentation.md](research/eval-segmentation.md)).

Done on `main`: eye-level camera and review harness, segmentation for evaluation, city biomes, exposure (FRK-152), road
markings and kerbs (153), street trees (154), per-city light and canopy colour (155), street life: jeepneys, cars, kerb stalls,
overhead cables (147), crowns and leaf cards (156), the camera chain (165), wall speckle and denoise (166), weathering (167),
light variety and visible clouds (168, 174), pavement and kerb wear (175), per-strip metering (171), mesh cars with Kenney's
vehicles (169, 178), visible people and denser traffic (163), cumulus-like sky (149), and the Godot viewer and playable world
(ADR-0012, [../godot/README.md](../godot/README.md)).

Measured on `main` at 1addc7d (seed 7, one strip per city, local CPU render, eye-level strips metered, mesh cars, 600 people and
60 cars per km; photos in brackets). A strip's noise is about 0.02, so read differences under that as nothing.

| | Manila-Quezon (14 photos) | Nairobi centre | Khartoum city (3, a hint) |
|---|---|---|---|
| road share | 0.344 (0.241) | 0.393 (0.244) | 0.344 (0.259) |
| building share | 0.044 (0.089) | 0.337 (0.233) | 0.111 (0.107) |
| vegetation share | 0.193 (0.335) | 0.034 (0.098) | 0.049 (0.015) |
| vehicle share | 0.064 (0.077) | 0.039 (0.081) | 0.055 (0.031) |
| person share | 0.005 (0.006) | 0.005 (0.017) | 0.008 (0.200, one market photo) |
| brightness | 0.448 (0.480) | 0.459 (0.487) | 0.604 (0.540) |
| edge energy | 0.024 (0.041) | 0.021 (0.021) | 0.019 (0.043) |

Most shares are inside one photo standard deviation. What is still off: Khartoum contrast (z 2.8) and vegetation (z 1.5), Manila
saturation (z 1.0) and `seg_other` (z 0.9), edge energy in Manila and Khartoum. By eye: the road is smooth and wide (road share does
not respond to road width, it is set by the camera and field of view), people are block figures, parked props are boxes, the crowns
are stylised, and building faces are flat. Local CPU strips are usable for iteration, and pod and local pixels differ, so compare
like with like.

Next, in order: the pilot cycles FRK-149 (Manila-Quezon, remainder: road surface grain and edge energy), FRK-150 (Nairobi
centre) and FRK-148 (Khartoum: contrast and vegetation); then FRK-115 (facades and street furniture) once they show what is most
missing; then the other eight cities (FRK-151).

Parked until Jian says go: live agent simulation (FRK-184), zebra crossings (185), pavement as a dataset toggle (186), the SAR
epic (FRK-179 to 182), HDRIs, the Infinigen install, FRK-176 (owned street photos), FRK-117 (benchmark rebuild), FRK-45 (3D
benchmark) and any GPU training. Ask before starting any of them.

## 3. How a render run works

```bash
git push                                   # pods render pushed commits only
scripts/runpod/city_strips.sh <full sha> <tag> manila_quezon nairobi_centre khartoum_city
cat data/review/cities<tag>_progress.txt   # ends with ALL_DONE
fraktal streets data/review/street<tag>_manila_quezon/out/review/fly/frames \
  --city 'manila-quezon' --data-dir data --limit 30 --segment --sheet <sheet>.png
```

Locally (WSL, Windows Blender, `--device cpu` while the GPU is shared: 3 to 15 minutes a city depending on load), with `PATH=$PWD/.venv/bin:$PATH` and
`FRAKTAL_BPY_PYTHON=$PWD/scripts/wsl/winblender.sh`:

```bash
.venv/bin/python scripts/local/city_strips.py <tag> manila_quezon nairobi_centre khartoum_city --device cpu
```

`--eye-sensor phone|dashcam` (FRK-165) runs the camera chain over the frames: exposure and highlight shoulder,
softness, fringe, vignette, shot and read noise, JPEG loss, and for `dashcam` a bonnet with the labels blanked under
it. The raw frames stay in `frames_raw/`, the rig in `eye_sensor.json`. It is off by default; compare chain-on strips
with chain-on photos stats only (it moves brightness, contrast and edge energy).

Eye-level beauty frames are denoised (OpenImageDenoise, FRK-166): at 32 samples the walls otherwise show coloured grain.
Labels and depth are separate passes and are unchanged. Aerial renders are not denoised.

Road wear and wall grime (FRK-167): `markings.build(..., seed)` adds tyre tracks, oil, patches, manholes, cracks and gutter
grime as flat parts under the paint, labelled ground like the marks. Textured `wall-*` materials get base grime and rain
streaks in the shader (`add_wall_wear`). Both are procedural, so there are no new assets. About 0.003% of label pixels
(wheel and foot contact) flip from vehicle or person to ground under the 4 mm wear boxes.

Pavement and kerb wear (FRK-175): `markings._edge_wear` adds dirt up the kerb face, scuffs on its top and, where there is a
raised pavement (Nairobi), stains, cracks, missing pavers and weeds, as seeded flat parts labelled ground.

Light variety (FRK-168, eye-level frames only): `lighting.variety` adds a per-seed sun height offset (-5 to +8 degrees,
strength-compensated, warmer when low), cloud shadows (an invisible plane that only the sun's shadow rays see, cover 0 to 0.3,
40% of the sun blocked) and haze (a bounded scattering box; thicker for Khartoum). `city_strips.py --seed N` picks the seed.
Brightness varied a lot between seeds (Manila seed 13 rendered at 0.31 without any variety): sky fill and how much sky is in
view are about equal causes. Eye-level strips are now metered (FRK-171, `metering.py`, `--no-meter` to skip): one gamma pulls the
strip's mean 70% of the way to 0.46, which took the Manila-Quezon std across 8 seeds from 0.053 to 0.016
([research/seed-brightness.md](research/seed-brightness.md)). No HDRIs yet.
Visible clouds (FRK-174): when a seed has cloud cover, the sky shader draws clouds for camera rays only (`add_sky_clouds`),
at the seed's scale and offset, so a cloudier seed has more of them and of their shadows. Sky light and the label and depth
passes are unchanged. No HDRIs yet.

Moving cars and people: with `--car-meshes` (the default in `city_strips.py`) the traffic is nine CC0 models (see below); people and the
box-car fallback are built from boxes that may be tapered, sloped or offset (`CAR_PARTS`, `PERSON_PARTS` in `blender_scene.py`). Each
person gets a look from their index (`person_look`: height, build, skin, shirt, trousers), with no random state. Parked cars and
jeepneys come from `propgen.py`, not from here.

Poles stand clear of every road, not only the one they line (`POLE_ROAD_CLEARANCE_M`), and Manila-Quezon and Khartoum now have a pavement strip (1.8 m, 1.5 m) so the people of the Godot replay have somewhere to walk. This changes the Blender renders of those two cities slightly (a raised strip beside the kerb).

Leaf cards cover the crown (FRK-156): each crown gets enough cards to cover its surface about once (`card_budget`, up to 220,
half-size 0.6 to 1.3 m, hugging the crown), and the canopy greens are lighter. On Manila-Quezon (seed 7, 32 samples) the segmenter's
vegetation share went 0.136 to 0.206 (photos 0.337), building 0.064 to 0.042, and `seg_other` (the smooth blobs read as food and
couches) 0.096 to 0.057, for +11% strip time. Bigger crowns made it worse (`seg_other` 0.126); more, smaller cards than cover the
surface did nothing measurable (a strip's noise is about 0.02). Khartoum's vegetation went 0.036 to 0.043 (photos 0.015, 3 photos).

Render options (FRK-169, see [research/trial-assets.md](research/trial-assets.md)): `--car-meshes` (off in `render3d`) swaps the box cars
for nine CC0 car models (six Quaternius, plus Kenney's truck, van and delivery, FRK-178; on by default in `scripts/local/city_strips.py`), `--leaf-cards` (on by default for street strips, `--no-leaf-cards` to turn it off) adds leaf quads to tree crowns; the label and depth passes cut them by the leaf alpha.
The pilot strips default to 60 cars and 600 people per km (the pods use 25 and 30): people are spread evenly over the whole street network, so at 200 per km only 0.7 were within 15 m of the camera and person share was 0.002 to 0.003 (FRK-163). At 600 and with a per-person look (height, build, skin, shirt, trousers) person share is 0.005 and vehicle share 0.039 (Nairobi), 0.064 (Manila), 0.055 (Khartoum, a little high).
Sky clouds (FRK-149): the sky noise is coarse with few octaves and soft edges (`CLOUD_SKY_NOISE`, `_DETAIL`, `_ROUGH`, `_SOFT` in `blender_scene.py`); at the old fine noise the sky read as hundreds of small flecks, not the photos' broad soft cumulus. The cloud shadows on the ground are separate and unchanged. Street-tree greens are 18% greyer (crowns were the most saturated class).
Objects that exist only for the beauty pass (`haze`, `clouds`, `treecards`) must be listed in `LOOK_HIDDEN` or they smear the label and depth passes.

`--override '{"buildings": {"front_setback_m": 6}}'` tries biome settings without editing `biomes.py`. It writes the same
`data/review/street<tag>_<biome>/out/review/fly/frames` layout and a `commit.txt` per city
(HEAD, and "dirty" if the tree had changes), so the `fraktal streets` line above works unchanged. Pod and local
pixels differ, so compare like with like.

- One batch at a time. Delete the old `driver.log` first, and confirm `pods left: 0` in the new one.
- Only touch pods this project started. Other projects share the RunPod account.
- A refusal with "RunPod balance is too low to rent a pod" means a top-up is needed; do not retry in a loop.
- Cost control and tear-down are in [RUNPOD.md](RUNPOD.md) and the `runpod-jobs` skill (dead-man's switch).
- Look at the renders before calling a visual card done. Report what only partly worked.

## 4. Habits that are not obvious from the code

These are in [AGENTS.md](../AGENTS.md) too, repeated here because they bite first:

- Branch per slice, PR only, rebase-merge when CI is green. **CI is the full gate**: locally run only ruff, mypy on
  changed files and the tests for touched code. A force-push to your own PR branch needs Jian's approval.
- Determinism only through `fraktal.ids.substream`. New config fields go in `procedural.LATER_FIELDS`. Do not repin
  Khartoum's world id unless told.
- `blender_scene.py` runs in Blender's Python and must not import `fraktal`; its tests pull functions out of its source with `ast`. No PIL in core code or tests.
- The GPU is shared with Jian's other jobs: `nvidia-smi` first, `--device cpu` for Blender and `FRAKTAL_SEG_DEVICE=cpu` for segmentation, GPU jobs only through `pueue` (the `gpu-job-queue` skill), and never kill or reorder a job you did not add. Never `pkill -f` or `pgrep -f` with a pattern in your own command line.
- Assets are owned or CC0 with a manifest row. Real photos are looking-and-evaluation only.
- No co-author or tool attributions in commits or PRs. After each PR: update the Pandan card, then report what
  changed, what was measured, what is still open and what needs deciding.
- Check the board for an existing card before filing a new one.
