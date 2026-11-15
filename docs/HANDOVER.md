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

Done on `main`: eye-level camera and review harness, segmentation for evaluation, city biomes, exposure fix
(FRK-152), road markings and kerbs (153), street trees (154), per-city light and canopy colour (155), street life:
jeepneys, cars with wheels, kerb stalls, overhead cables (147, code merged).

Measured on current `main` (commit 5ffeebd, local GPU render, 2026-11-14; photos in brackets, z in the harness
output): the three pilots all show the street-life code (cables, jeepneys, stalls, motorbikes), and all three share the
same gaps.

| | Manila-Quezon (12 photos) | Nairobi centre (18) | Khartoum city (3, a hint) |
|---|---|---|---|
| road share | 0.362 (0.268) | 0.411 (0.244) | 0.365 (0.259) |
| building share | 0.038 (0.053) | 0.331 (0.233) | 0.134 (0.107) |
| vegetation share | 0.223 (0.337) | 0.067 (0.098) | 0.024 (0.015) |
| vehicle share | 0.031 (0.090) | 0.011 (0.081) | 0.013 (0.031) |
| person share | 0.000 (0.006) | 0.001 (0.017) | 0.000 (0.200, one market photo) |
| brightness | 0.436 (0.486) | 0.466 (0.487) | 0.627 (0.540) |

Manila-Quezon and Nairobi above include FRK-161, 162 and 164 (Manila setback 10 m, 45% and 30% built in core and edge; it was 2.5 m and 92% and
80%, which gave building 0.230, vegetation 0.072). Road share does not respond to road width (it stayed near 0.35
with roads 16 m and 7 m wide); it is set by the camera and field of view, and it is already within one photo standard
deviation for Manila (Nairobi and Khartoum are at about one). Narrower roads made building share worse.

Looking at the sheets (before FRK-161 for Manila): Nairobi reads as a canyon of tall uniform blocks with sparse boxy cars and a wide empty road;
Khartoum is closer in feel but its vehicles are blocky and its road is too wide. The local numbers for Manila-Quezon
match the pod's from commit ac52cc5 to the third decimal, so local GPU strips are usable for iteration.

Next, in order:

1. Finish FRK-163: vehicles are still boxes and people box figures. The mesh and leaf-card trial is FRK-169.
2. The camera and surface realism steps from [research/hyperrealism-methods.md](research/hyperrealism-methods.md):
   eye-level camera chain (FRK-165), wall speckle (FRK-166), weathering and decals (FRK-167), light variety (FRK-168).
3. Pilot iteration cycles: FRK-148 (Khartoum), FRK-149 (Manila-Quezon), FRK-150 (Nairobi centre).
4. The other eight cities (FRK-151).

Parked until Jian says go: the Godot driving design (FRK-90 browser demo is the simpler route), FRK-117 (benchmark
rebuild), FRK-45 (3D benchmark) and any GPU training. Ask before starting any of them.

## 3. How a render run works

```bash
git push                                   # pods render pushed commits only
scripts/runpod/city_strips.sh <full sha> <tag> manila_quezon nairobi_centre khartoum_city
cat data/review/cities<tag>_progress.txt   # ends with ALL_DONE
fraktal streets data/review/street<tag>_manila_quezon/out/review/fly/frames \
  --city 'manila-quezon' --data-dir data --limit 30 --segment --sheet <sheet>.png
```

Locally (WSL, Windows Blender on the GPU, about 45 seconds a city), with `PATH=$PWD/.venv/bin:$PATH` and
`FRAKTAL_BPY_PYTHON=$PWD/scripts/wsl/winblender.sh`:

```bash
.venv/bin/python scripts/local/city_strips.py <tag> manila_quezon nairobi_centre khartoum_city
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
Brightness varies a lot between seeds already (Manila seed 13 renders at 0.31 without any variety), so compare against the
same seed with variety off. No HDRIs yet.
Visible clouds (FRK-174): when a seed has cloud cover, the sky shader draws clouds for camera rays only (`add_sky_clouds`),
at the seed's scale and offset, so a cloudier seed has more of them and of their shadows. Sky light and the label and depth
passes are unchanged. No HDRIs yet.

Moving cars and people (FRK-163) are still built from boxes, but a box may now be tapered, sloped or offset to one side
(`CAR_PARTS`, `PERSON_PARTS` in `blender_scene.py`): a car has a sloping bonnet, a tapered glass cabin, a roof panel and four
wheels; a person has two legs, arms, shoulders and a head. Parked cars and jeepneys come from `propgen.py`, not from here.

Trial options (FRK-169, off by default, see [research/trial-assets.md](research/trial-assets.md)): `--car-meshes` swaps the box cars
for six CC0 car models, `--leaf-cards` (on by default for street strips, `--no-leaf-cards` to turn it off) adds leaf quads to tree crowns; the label and depth passes cut them by the leaf alpha. Objects that exist only for
the beauty pass (`haze`, `clouds`, `treecards`) must be listed in `LOOK_HIDDEN` or they smear the label and depth passes.

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
- `blender_scene.py` runs in Blender's Python and must not import `fraktal`. No PIL in core code or tests.
- Assets are owned or CC0 with a manifest row. Real photos are looking-and-evaluation only.
- No co-author or tool attributions in commits or PRs. After each PR: update the Pandan card, then report what
  changed, what was measured, what is still open and what needs deciding.
- Check the board for an existing card before filing a new one.
