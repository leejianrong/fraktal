# Plan: generative textures and 3D props for the asset library (epic FRK-E10)

Written 2026-11-07. Goal: widen what the renders contain (vehicles, stalls, roof clutter, facades, soil and
roof materials) with diffusion-made textures and image-to-3D and text-to-3D props, without breaking the
rules that make a dataset sellable. Builds on [texture-model-spike.md](texture-model-spike.md) (FRK-59, FRK-73),
[3d-assets-and-realism.md](3d-assets-and-realism.md), [infinigen-spike.md](infinigen-spike.md) and ADR-0010
decisions 5 and 6. The model table below is from 2026 web summaries; the licence files of the shortlisted
models were read the same day and are in [ADR-0013](../adr/0013-which-generative-models-may-make-assets-for-sold-datasets.md),
which overrides this table where they differ (notably: TRELLIS.2 depends on non-commercial NVIDIA renderers,
and TripoSG's default background remover needs a commercial agreement).

## Where the library stands

- `assets/` has 24 material specs and 9 textures: 5 CC0 colour maps from ambientCG and 4 SDXL-generated
  ones (`gen-*`, licence `generated-permissive`, FRK-73 and FRK-76). The manifest has 33 rows.
- There are no meshes in the library. Cars, people, trees, poles, tanks and buildings are procedural, built in
  code. Infinigen was deferred.
- Textures are colour only: no normal, roughness or height maps.

## Guardrails

1. **Reference imagery never feeds a generator.** The board (Open Cities, Maxar, SpaceNet, ground photos) is for
   looking. No fine-tuning, no image conditioning (img2img, IP-Adapter and similar), no reference to a real
   photo in a prompt pipeline. Prompts are written from looking at the board and from general knowledge.
2. **Licences gate models, and their dependencies.** Allowed means: weights and outputs usable in a sold
   dataset, with no territory, revenue or user-count carve-out. The relaxed stance on reference imagery does
   not apply here because generated assets ship in datasets.
3. **Every generated asset has a manifest row with a `generator` block** (model, version, prompt, seed,
   sampler settings, date). Generated output is probably not copyrightable (FRK-59), so we ship it but cannot
   claim to own it; assets are not what the dataset is sold for.
4. **Generation is offline.** Assets are files with hashes. A scene picks them by a seeded sub-stream and lists
   asset ids and hashes, so determinism (ADR-0003) is unchanged.
5. **Labels come from the catalogue.** A mesh's class is set when it enters the library, never inferred from the
   model, so label exactness (R5) holds.
6. **No logos or text.** Prompts ask for none, and QA looks for them (a generated tuk-tuk can carry a brand).

## Models, as reported (unverified until FRK-121)

| Model | Job | Licence reported | Plan |
|---|---|---|---|
| TRELLIS.2 (Microsoft) | image to 3D with materials | MIT code and weights; imports nvdiffrast (NVIDIA non-commercial) only for texture baking, previews and GLB export, not for geometry; default background remover RMBG-2.0 is excluded; image encoder DINOv3 is gated; 24 GB or more of GPU memory | **excluded** (Jian, 2026-11-08): DINOv3 licence concerns; TRELLIS v1 has non-commercial dependencies (ADR-0013) |
| Step1X-3D | image to 3D, geometry and texture | Apache-2.0 code and weights; texture stage reuses Hunyuan3D tools and nvdiffrast | **used**, geometry stage only (FRK-122) |
| TripoSG | image to 3D, geometry only | MIT at the repository root, but `triposg/` carries the Tencent FlashVDM licence (no use in the EU, UK or South Korea; no use of outputs to improve other AI models) and it needs diso (CC BY-NC) | **excluded** (ADR-0013) |
| Pixal3D, Direct3D-S2 | image to 3D | MIT | watch, not first |
| Hunyuan3D 2.1 | image to 3D | Tencent community licence, not valid in the EU, UK or South Korea, with a user threshold | excluded (ADR-0010) |
| Stable Fast 3D | image to 3D | Stability community licence, free under $1M revenue | excluded |
| Cube 3D | text to 3D | research only | excluded |
| FLUX.1 schnell | text to image | Apache-2.0 (read in FRK-59) | the prompt-to-image step |
| SDXL base 1.0 | text to image, textures | OpenRAIL++-M (read in FRK-59) | textures, as in FRK-73 |
| FLUX.1 dev | text to image | non-commercial model terms (read in FRK-59) | excluded |
| DreamPBR, MatLat, FabricDiffusion | PBR material generation | research code; weights and licences unknown | evaluate only if weights are released and the base model licence is fine |
| Meshy, Tripo, Rodin and other hosted APIs | text and image to 3D | per-service terms, not read | out unless terms are read |

Text-to-3D models of the open kind are weaker than a two-step pipeline, so "text to 3D" here means text to
image (FLUX.1 schnell) then image to 3D.

## Assets written as code (the LLM route)

An LLM can write the asset instead of a diffusion model drawing it. The tools that exist: BlenderMCP, a
community plugin that lets any LLM drive Blender; LL3M (University of Chicago), a multi-agent system that writes
and revises Blender Python with a critic that looks at renders; and BlenderLLM, a 7B model fine-tuned on
Blender and CAD scripts. None of their licences were read, and we do not need them: the useful part is the loop,
not the tool.

For Fraktal this route already exists in the repo: the building grammar, trees, cars and poles are code. What
an LLM adds is writing more parametric generators (a motorbike, a tuk-tuk, a minibus, a stall) as Python
functions with parameters and a seed, which a person reviews and tests.

| | Code-written | Image-to-3D |
|---|---|---|
| Licence | none, we own the code (the LLM's output terms apply) | per ADR-0013, with dependency traps |
| Determinism and variation | exact, parameters give variants for free | one mesh per generation |
| Labels and parts | exact, each part is a named object | one mesh, parts unknown |
| Look | primitives, extrusions and curves: blocky, fine at aerial scale | more organic, baked lighting and holes |
| GPU | none (a render loop to look at it) | 8 to 29 GB per run |
| Best for | vehicles, stalls, tanks, walls, signs at aerial scale | organic or cluttered ground-level props |

Proposal: do both and compare (FRK-130 for the code route, FRK-122 for image-to-3D). The code route is the better
first bet for aerial-scale props because it has no licence risk, is deterministic and parametric, and is cheap to
test. The loop needs Blender to render what the LLM wrote; locally there is none, so that loop runs on a pod
unless Jian allows a headless Blender on this machine.

## What to generate and what to keep procedural

| Generate | Keep procedural |
|---|---|
| vehicles beyond boxes: motorbikes, tuk-tuks, minibuses, trucks, pickups | buildings (the grammar), walls, fences |
| market stalls, signs, street furniture | storage tanks, containers, silos (simple shapes, cheaper to vary and label) |
| roof clutter: dishes, panels, drying racks | terrain, roads, land cover |
| soil, plaster, paint, rust and tile textures | corrugated tin relief (geometry or normal map, FRK-109) |
| facade materials | trees until FRK-111 shows procedural is not enough; generated trees are usually poor |

People stay boxes at aerial scale; ground-level people need rigs and are out of scope.

## Pipelines

**Textures (FRK-126).** Prompt from the gap list ("galvanised corrugated sheet, rusted, seen from above,
seamless") with SDXL or FLUX schnell, made tileable, then derive height, normal and roughness (a model or a
heuristic; compare both), remove baked light, check the seam and the histogram, write with manifest rows.
Single-image PBR derivation is approximate, so real relief stays geometry where it matters.

**Image to 3D (FRK-122).** A prompt-made image of one object on a plain background, background removal, image
to 3D, then clean-up, scale from a size table, QA, manifest row. Background-removal models need a licence check
too (two default ones are excluded).

**Text to 3D (FRK-123).** The same, with FLUX schnell making the image, seeded and logged.

**Mesh QA and import (FRK-124).** Set scale and origin, decimate (aerial scale needs about 20k triangles at
most), reject non-manifold or tiny-detail meshes, import into the scene as instances with a catalogue class.

**Library layout (FRK-125, FRK-127).** `assets/meshes/<id>.glb` with a spec beside it (class, size in metres,
up axis, biomes, tags, generator). A library that grows past about 50 MB moves to object storage with the
hashes kept in the manifest; that is a later decision.

**Variation (FRK-128).** A seeded sampler picks assets and varies scale, rotation, tint and wear, and reports
how often each asset repeats in a tile, since twenty assets repeated everywhere look fake.

## First trial results (FRK-122, 2026-11-07)

Stack: SDXL base 1.0 (30 steps, 1024 px), BiRefNet cut-out, Step1X-3D-Geometry-1300m (50 steps, octree 384,
marching cubes), then our own clean-up and checks. 20 prompted props (motorbikes, tuk-tuks, minibuses, pickups,
stalls, water tanks, containers, signs) on one RunPod A40 (48 GB, $0.59 an hour). Images:
[props-trial-images.jpg](../img/generative-assets/props-trial-images.jpg); meshes with the QA verdict:
[props-trial-meshes.jpg](../img/generative-assets/props-trial-meshes.jpg).

| Step | Time | GPU memory |
|---|---|---|
| SDXL image | 6.8 s each (model load 10 to 14 s) | 11.2 GB |
| BiRefNet cut-out | 0.55 s each | 1.7 GB |
| Step1X-3D geometry | **72.6 s each** (65.5 to 85.5; model load 8 to 12 s) | 7.4 GB |
| Whole run | 47 minutes: 165 s install, 155 s images, 17 s cut-outs, about 24 minutes of meshes | |

Cost: about $0.46 for the successful run, and about $0.8 for the six runs it took to get one working (estimated
from pod uptime times the hourly rate, not from an invoice). At about 50 meshes per pod hour, generation alone is
about $0.012 a mesh plus 7 s for its image. How much of the 73 s is the CPU marching cubes and how much the GPU
was not measured.

What came out. Raw meshes have 650k to 1.26M faces (measured on the 11 raw meshes that arrived), so they are
decimated to 20k faces (about 350 KB each) before use. The motorbikes, tuk-tuks, water tanks, the white minibus
and pickup, and the market stalls are recognisable and about the right shape. Wrong: the shop sign is a wireframe
cube; both containers are near-cubes (5.9 x 6.1 x 5.9 m, not 6.1 x 2.4 x 2.6), because a three-quarter view
foreshortens the depth; the red pickup is two trucks (its image had two); both minibuses are too wide. The
quadric decimation of a fragmented 1M-face surface leaves holes in thin roofs (the stalls). Every mesh is grey:
this route has no colour, so colour must come from our materials, which is not built yet.

QA. With only generic checks, all 20 passed, including the cube-frame sign. With expected length, width and
height per class (`EXPECTED` in `props_trial.py`, compared after scaling to the longest side), 12 pass and 8 are
flagged: both containers, both minibuses, the two-truck pickup, the sign cube, and the street sign, which is a
false alarm from my crude expected size. A proportion prior is cheap and catches what the eye catches; a person
still has to look.

What went wrong, in order (each cost a run of a few cents):

- Step1X-3D's `check_inputs` is inverted: it raises for a PIL image or a tensor and accepts only a file path.
- It imports training code at start-up (`pytorch_lightning`, `typeguard`, `streaming` from `mosaicml-streaming`)
  and `pymeshlab`, whose file plugins fail to load on the pod image; `remove_floater` then threw. Fixed by
  installing the imports, switching its `pymeshlab` post-processing off and doing floaters and decimation
  ourselves. The job's early import check with automatic `pip install` of missing modules worked well.
- I launched run 5 while run 4's driver was alive; both use the pod name `fraktal-props`, and `rp up` reuses a
  pod by name, so they collided. Check for running drivers before a launch.
- The 288 MB artifact (raw meshes included) stalled near the end of the relay transfer, and the pod's sender then
  timed out. The truncated archive still gave the images, cut-outs and all 20 final meshes, and the per-mesh
  numbers came from the pod log. My own report printing was cut at 6,000 characters, which lost the full QA
  table. Next time the artifact leaves out raw meshes, and the full report is printed.
- A relay-get that dies (it was killed once, exit 137) tore down a healthy pod; the driver now retries while the
  pod is up.

Not done against the card: no Blender render sheet (the sheet is drawn with matplotlib; importing the meshes
into Blender is FRK-124), and TRELLIS.2 was not run, because of the gated DINOv3 weights. Its nvdiffrast question
is answered in ADR-0013.

Verdict so far: image-to-3D works well enough to be worth building on for vehicles, tanks and stalls, with a
proportion check and a human look in the loop, and it is clearly not a fit for flat things with text (signs).
The cheap, deterministic code route (FRK-130) is still untested against it.

### Retry of the flagged props (2026-11-08)

The 7 props that meshqa flagged (both minibuses, red pickup, both containers, shop sign, street sign) plus the
cube-shaped umbrella stall were re-run with prompts that state the shape and a side-on view, two seeds each
(`--retry`, 16 meshes, about 28 minutes on an A40, roughly $0.28 by uptime times rate). 11 of 16 pass the
proportion check, and by eye every prop has at least one usable seed. Containers stay a coin flip (2 of 4 are
proper boxes), and the check is too strict on a thin street sign. Holes from decimation to 20k faces remain.
A vision-model reading of the same sheet (one subagent, not blind: the sheet shows the check's FAIL flags)
rejected the same two cubes, marked 5 meshes usable and 9 as needing fixes for holes and lumps. Best of N seeds
with the check, then a look, is the working method.

## Text to 3D as a command (FRK-123, 2026-11-11)

`scripts/props/text_to_3d.py OUT --prompts prompts.json --step1x DIR [--seed N]` takes a list of
`{id, prompt, size_m, expected_lwh}` and runs the FRK-122 stack on it: SDXL base 1.0, BiRefNet,
Step1X-3D geometry (marching cubes, our own clean-up), then QA. The pod job runs it with
`MODE=text3d scripts/runpod/props_run.sh COMMIT OUT` and then a Blender stage
(`bpy_glb_sheet.py`, a clay render of each mesh at ground level and from above). Per mesh it writes
`meshes/<id>.glb` and `meshes/<id>.json` (prompt, image and mesh seeds, seconds per stage, QA verdict, face
count, SHA-256 of the GLB), and per run `manifest.json` with the Hugging Face revision of each model, the
Step1X-3D code commit, library versions and the totals (ADR-0013 decision 5). FLUX.1 schnell was not used:
SDXL is allowed and was already working.

Two runs of the 20-prop list (`prompts-20.json`, seed 0) on A40 pods at $0.59 an hour:

| | |
|---|---|
| Per prop (image 6.8 s, cut-out 0.5 s, mesh 70.6 s on average) | **77.9 s**, about **$0.0128** |
| Whole run | about 33 minutes: install and import check 159 s, models and 20 props about 28 minutes, Blender stage about 2 minutes, including installing `bpy`. About $0.33 a run with the pod's start and teardown |
| QA by proportion | 13 of 20 pass; the same seven fail as in FRK-122 (both containers, both minibuses, the two-truck pickup, the cube-frame shop sign, the street sign) |
| Determinism | all 20 GLBs have the same SHA-256 in both runs, on two different pod instances |

Cost per prop is the mean generation time times the hourly rate. Model loading and installing are paid once
per run, not per prop. At about 46 props an hour a library of 200 props is about $2.60 of generation, before
the failures that are thrown away.

`docs/img/generative-assets/text-to-3d-blender-sheet.jpg` is the Blender check: every prop, ground view and
nadir view. Vehicles, tanks, stalls and the billboard read well. The flagged ones look as flagged: containers
are near-cubes, the shop sign is a wireframe cube, the red pickup is two trucks stacked.

The Blender stage found two faults that the matplotlib sheet of FRK-122 could not show:

- **The GLBs were written z up.** glTF is y up, so Blender's importer stood every prop on its side, as would
  any other glTF reader (a viewer, Godot). `meshqa.to_gltf` now rotates the mesh a quarter turn on export, with
  a round-trip test against `to_z_up`. FRK-124 (import, scale, instancing) must read these files as y up.
- **Light grey clay clipped white** under the Nishita sky and sun, so the shapes were washed out. The clay is
  now darker and the exposure lower.

Not done: the sheet in the image was rendered with a local `bpy` from the pod's meshes after applying the same
quarter turn, not by a third pod run, because the pod's Blender stage had already worked (the first run's
install lacked `apt-get update`, fixed) and only the file's up axis and the exposure changed. The next pod run
writes y-up files directly. Texture and colour are still ours to add (FRK-126); these meshes are plain grey.

## Code-written props (FRK-130, 2026-11-08)

`src/fraktal/propgen.py` holds five parametric generators (container, water tank, market stall, signboard,
tuk-tuk) in plain numpy, seeded through `substream`. `scripts/props/bpy_build.py` builds them in Blender,
renders a ground and an aerial view with Cycles and exports a GLB each; the contact sheet of ten variants of
each is `docs/img/generative-assets/propgen-trial.jpg`. I wrote the generators myself in one session, so this measures what an
LLM-authored generator looks like, not an autonomous agent loop (BlenderMCP or LL3M style, not tried).

| | Code route (FRK-130) | Image to 3D (FRK-122) |
|---|---|---|
| Time per asset | 0.2 to 2.3 ms to generate, 0.84 s to render two views and export, 16 vCPU | 72.6 s of GPU, plus 7 s image |
| Pod cost | about $0.02 for all 50 variants of compute; three runs with boot about $0.15 (uptime x rate, not billed) | about $0.012 per mesh of GPU time |
| Size and proportions | exact by construction, tested (15 unit tests: determinism, size limits, ground contact, outward faces) | 8 of 20 off on the first run; best of two seeds fixed all 8 by eye |
| Colour and materials | flat colour with roughness, set per part | none, grey |
| Surface and detail | clean, blocky, toy-like (tuk-tuk, stall, tank); corrugation on containers is faint | organic, far more detail, but holes and lumps |
| Variation | only what the parameters give: a container has two (length, colour) | every seed differs |

What it shows: the code route makes clean, correctly scaled, coloured assets for almost nothing, but they look
simple, and variation is capped by how many parameters we write. It is a good fit for boxy, repetitive things
that must have exact dimensions (containers, tanks, signboards, stall frames) and a poor fit for vehicles, where
the tuk-tuk looks like a child's model next to the generated one.

Defects I saw on the sheet, not fixed: white signboard faces vanish against the pale sky, the tank lid and ribs
are barely visible at aerial scale, and the container corrugation needs bevels to read. Three renders were
needed to get a usable exposure; the first two clipped white and black plastic to the background.

Route by asset, as a first answer (Jian, 2026-11-08: happy with this trial; use the code route more, and make its
look richer later with textures and detail in Blender; TRELLIS and DINOv3 are excluded, see ADR-0013):

- Containers, tanks, signboards, stalls: code, with more parameters (door states, rust, dents, stencils).
- Vehicles and anything organic: image to 3D, best of N seeds, then the proportion check.
- Generic clutter that matches our look: CC0 libraries, once surveyed.
- Not tried: texturing the generated meshes with our own materials (FRK-126), which would close the colour gap.

## Asset taxonomy and coverage targets (FRK-127, 2026-11-10)

"Distinct" counts shapes a tile can show (a code-written prop has 8 shape variants per kind, each in 4 tints and 3 wear levels, so 96 looks per kind). Routes: **code** is a generator in `propgen`, `trees3d`, `building_grammar` or `compounds`; **gen-tex** is an SDXL texture on code geometry (FRK-126 route); **CC0** is a scanned texture in the manifest; **model** is image-to-3D, held back by ADR-0013 and FRK-123. The first choice is always code. Status: done, a card, or **not planned** (named so a gap is a decision, not an oversight).

| Class | Now (distinct) | Target | Route | Status |
|---|---|---|---|---|
| Cars (parked and moving) | 1 box body | 6 body shapes (hatch, saloon, SUV, pickup, van, taxi) | code | **not planned** (sim vehicles are boxes; FRK-112 added the others) |
| Tuk-tuks | 8 | 8 | code | done (FRK-130) |
| Motorbikes | 8 | 12, plus a rider variant | code | shapes done (FRK-112); rider **not planned** |
| Minibuses | 8 | 8 | code | done (FRK-112) |
| Trucks and buses | 0 | 6 trucks, 4 buses (dense urban, industrial yards) | code | **not planned** |
| Carts and animals | 0 | 4 hand and donkey carts (Khartoum, informal) | code | **not planned** |
| Standing people | 8 | 16 (clothing, bags, children) | code | shapes done (FRK-112); more is **not planned** |
| Walking people (sim) | 1 | follow the sim | code | FRK-94 (Godot replay); Blender side **not planned** |
| Poles | 1 | 3 (lamp, power pole, mast) | code | **not planned** |
| Stalls | 8 | 12 | code | done (FRK-130); more **not planned** |
| Signboards | 8 | 12 | code | done (FRK-130) |
| Bins, drums, litter | 8 each | 8 each | code | done (FRK-112) |
| Bus shelters, kiosks, benches | 0 | 6 | code | FRK-115 |
| Containers, tanks, silos | 8 each | 8 each | code | done (FRK-130, FRK-107) |
| Roof colours (tin) | 8 | 8 | gen-tex plus code | done (FRK-110) |
| Roof materials (tile, flat) | 1 tile, 4 flat | 3 tile, 6 flat | gen-tex, CC0 | **not planned** |
| Roof clutter (tanks, solar, brick stacks) | 3 | 10 | code | FRK-110 started it; more **not planned** |
| Building footprints | rectangle, L, T, shed | add courtyard and hall (done), stepped block | code | FRK-105 done; stepped block **not planned** |
| Facade materials | 8 wall materials | 16 | gen-tex, CC0 | FRK-115 |
| Compound walls | 3 kinds, 10 colours, angled and curved outlines | 5 kinds (add mud wall, chain link) | code | FRK-106 done; extra kinds **not planned** |
| Trees | 3 species | 6 (acacia, neem, date palm, mango, banana, eucalyptus) | code | FRK-111 done for 3; the rest **not planned** |
| Ground textures | 1 soil, 2 concrete | 4 soils, 3 pavements, mud, grass | gen-tex, CC0 | FRK-108 and FRK-126 done for the first set; more **not planned** |

Per biome, what matters most, in priority order (a target is the distinct shapes of that class the biome should show):

| Biome | First | Then | Barely needed |
|---|---|---|---|
| Khartoum | walls (5 kinds), tin and flat roofs, acacia and neem, motorbikes, carts | stalls, minibuses, poles | containers, silos |
| Dense urban | minibuses and trucks, bins, facade materials (16), halls and yards, poles | motorbikes, bus shelters, signboards | trees beyond 3 species |
| Informal settlement | sheet fences and tin roofs in more colours, stalls, tanks on roofs, litter, motorbikes | carts, people, tuk-tuks | halls, silos |
| Oil palm plantation | palm variety (age, height), tracks, trucks, sheds | water tanks, a few workers | walls, litter |
| Paddy terraces | water and mud textures, huts, a few people and bikes | trees | vehicles in number |
| Mangrove coast | mud and water textures, boats (not built), stilt huts | trees (mangrove species) | cars, walls |

Boats and stilt huts are **not planned**: no biome card exists for them. A target is a ceiling to aim for, not a count to hit before rendering; ADR-0009 says to build for coverage and measure to steer, so the table is reviewed when FRK-129 shows what still looks fake.

## Order of work

1. FRK-121: read the licences, write ADR-0013. No GPU, can start now.
2. FRK-126 (textures) and FRK-125 (manifest) once the licences are read: cheap, and textures feed FRK-109.
3. FRK-122 done (above); next FRK-124 (import, scale, instancing, labels) and FRK-123.
4. FRK-127 after Jian's FRK-103 choice; FRK-128 after FRK-124 and FRK-127; FRK-129 to look at the result.

Realism cards consume these: FRK-109 and FRK-110 want the textures, FRK-112 wants the props, FRK-115 wants both.

## Risks

- Generated meshes arrive with baked shadows, wrong scale, holes and internal faces; QA is most of the work.
- Cost per mesh and per texture is not known; FRK-122 measures it. GPU memory of 24 GB or more is reported for
  the first choice, which a 48 GB pod covers.
- Aerial-scale props are a few pixels, so the first payoff may be in drone and ground views, not satellite.
- A diverse library needs a lot of distinct assets; FRK-127 sets targets so the effort goes where the gaps are.
- Image-model outputs may look alike across seeds; the variation step and Jian's look at the strips are the check.
