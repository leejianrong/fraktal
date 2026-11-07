# Plan: generative textures and 3D props for the asset library (epic FRK-E10)

Written 2026-11-07. Goal: widen what the renders contain (vehicles, stalls, roof clutter, facades, soil and
roof materials) with diffusion-made textures and image-to-3D and text-to-3D props, without breaking the
rules that make a dataset sellable. Builds on [texture-model-spike.md](texture-model-spike.md) (FRK-59, FRK-73),
[3d-assets-and-realism.md](3d-assets-and-realism.md), [infinigen-spike.md](infinigen-spike.md) and ADR-0010
decisions 5 and 6. The model facts below are from 2026 web summaries, **not primary sources**; FRK-121 reads
the licence files and replaces this table with an ADR.

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
| TRELLIS.2 (Microsoft) | image to 3D with materials | MIT, code and weights; the original pipeline used nvdiffrast and nvdiffrec, which may be non-commercial, and a reimplementation replaces them; about 24 GB of GPU memory | first choice, check dependencies |
| Step1X-3D | image to 3D, geometry and texture | Apache-2.0; about 27 to 29 GB | second choice |
| TripoSG | image to 3D, geometry only | MIT | needs a texturing step |
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
to 3D, then clean-up in headless Blender, scale from a size table, QA, manifest row. Background-removal models
need a licence check too.

**Text to 3D (FRK-123).** The same, with FLUX schnell making the image, seeded and logged.

**Mesh QA and import (FRK-124).** Set scale and origin, decimate (aerial scale needs about 20k triangles at
most), reject non-manifold or tiny-detail meshes, import into the scene as instances with a catalogue class.

**Library layout (FRK-125, FRK-127).** `assets/meshes/<id>.glb` with a spec beside it (class, size in metres,
up axis, biomes, tags, generator). A library that grows past about 50 MB moves to object storage with the
hashes kept in the manifest; that is a later decision.

**Variation (FRK-128).** A seeded sampler picks assets and varies scale, rotation, tint and wear, and reports
how often each asset repeats in a tile, since twenty assets repeated everywhere look fake.

## Order of work

1. FRK-121: read the licences, write ADR-0013. No GPU, can start now.
2. FRK-126 (textures) and FRK-125 (manifest) once the licences are read: cheap, and textures feed FRK-109.
3. FRK-122 (pod trial, small cost, unknown until measured), then FRK-124 and FRK-123.
4. FRK-127 after Jian's FRK-103 choice; FRK-128 after FRK-124 and FRK-127; FRK-129 to look at the result.

Realism cards consume these: FRK-109 and FRK-110 want the textures, FRK-112 wants the props, FRK-115 wants both.

## Risks

- Generated meshes arrive with baked shadows, wrong scale, holes and internal faces; QA is most of the work.
- Cost per mesh and per texture is not known; FRK-122 measures it. GPU memory of 24 GB or more is reported for
  the first choice, which a 48 GB pod covers.
- Aerial-scale props are a few pixels, so the first payoff may be in drone and ground views, not satellite.
- A diverse library needs a lot of distinct assets; FRK-127 sets targets so the effort goes where the gaps are.
- Image-model outputs may look alike across seeds; the variation step and Jian's look at the strips are the check.
