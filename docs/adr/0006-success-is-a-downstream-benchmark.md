# ADR-0006: Success is measured by a downstream building-segmentation benchmark

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

Synthetic data can look convincing and still not help a model. The project lives or dies on sim-to-real transfer, so "it works" has to be a number from a model trained on the data, not a judgement of how the images look.

## Decision

One benchmark task, building footprint segmentation, with a fixed small segmentation network (U-Net family, PyTorch) and a fixed training recipe so that only the data changes. Three arms are compared on a held-out real test set:

1. real-only, 10% of the real training data
2. real-only, 100%
3. synthetic pretrain, then fine-tune on the same 10% real

Initial bar, to be confirmed after the first baseline: arm 3 beats arm 1 by at least 5 IoU points and reaches at least 90% of arm 2's IoU. The benchmark dataset is chosen in SLICES V1 step 1 from open candidates (SpaceNet, WHU building dataset, Open Cities AI), subject to licence and regional fit.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Visual realism or FID scores | Not tied to the customer's actual outcome |
| Several tasks at once | Dilutes the signal and multiplies work before the first result |
| Customer pilot only | Slow, and no repeatable regression check |

## Consequences

An objective pass/fail from the first slice, and a regression test for every later renderer change. The costs: a real benchmark dataset must be obtained and its licence checked (some, like xView, are non-commercial), and the chosen dataset may not be tropical, so regional transfer is a separate open risk. One task also means the MVP says nothing yet about land cover, roads or change detection.

## Update 2026-11-03: V1 results and the revised bar

V1 ran on SpaceNet 2 Khartoum (ADR-0008): 810 train and 202 held-out test tiles, a fixed U-Net (about 2M parameters), 3000 optimiser steps per stage, on one RunPod A40. Arm 3 pretrained on 141 synthetic chips (crude renderer, ADR-0004) and then fine-tuned on the same 81 real tiles as arm 1. Pixel IoU on the test tiles:

| Run | Arm 1: real 10% | Arm 2: real 100% | Arm 3: synthetic then real 10% | Arm 3 minus arm 1 | Arm 3 / arm 2 | Synthetic-only IoU |
|-----|------|------|------|------|------|------|
| seed 0 | 0.6265 | 0.6353 | 0.6101 | -1.64 pts | 0.960 | 0.000005 |
| seed 0, repeated | 0.6241 | 0.6331 | 0.6170 | -0.72 pts | 0.974 | 0.000309 |
| seed 1 | 0.6319 | 0.6422 | 0.6192 | -1.27 pts | 0.964 | 0.000008 |

The two seed-0 runs agree within 0.7 points, so run-to-run noise is under one point. Only two seeds are independent, so treat the spread as a rough guide.

**What the numbers say.**

1. **The original bar is not met.** Arm 3 sits 0.7 to 1.6 points below arm 1, not 5 above it. Synthetic pretraining from the V1 renderer slightly hurt.
2. **The 10% regime cannot discriminate.** Arm 2 beats arm 1 by under 1 point (0.9, 0.9, 1.0). Ten percent of the real tiles is already almost as good as all of them for this model and recipe, so there is no room for any data source to add 5 points, and "90% of arm 2" is passed by arm 1 alone (about 98.6%).
3. **The synthetic domain gap is very large.** A model trained only on the synthetic chips scores about 0 IoU on real tiles. It finds essentially no real buildings. The V1 renderer produces flat colours and 10 m land cover, and the network learns cues that do not exist in real imagery.

**Revised bar.** The thresholds stay, and one precondition is added so the test can say something:

- The comparison counts only in a regime where real data is scarce: arm 2 must beat arm 1 by at least 10 IoU points. Arm 1 uses the smallest real fraction that satisfies this. A first candidate is 1% (about 8 tiles); this has not been measured.
- In that regime, arm 3 must beat arm 1 by at least 5 IoU points and reach at least 90% of arm 2.
- Synthetic-only IoU on real tiles is reported in every run as a realism gauge. V1 value: about 0. A target for it is set when V3 starts, from what the realism work achieves.

**Consequences.** V1 is a valid, repeatable measurement of the pipeline and a clear first answer: the crude renderer does not transfer. The next measurement is the same three arms at the scarce-data fraction, which costs a few GPU minutes. Whether to invest in V3 realism work is decided after that run. This update does not change the plan itself.

## Update 2026-11-03: realism gauge baseline (FRK-12)

`fraktal compare` measures how far synthetic chips are from real ones without a GPU. Baseline on the V1 chips (100 per side):

| Statistic | Real | V1 synthetic |
|---|---|---|
| Channel mean (R, G, B) | 88, 118, 34 | 133, 126, 109 |
| Channel std | 62, 68, 31 | 41, 32, 25 |
| Edge energy | 6.1 | 0.85 |
| Noise level | 24.2 | 17.6 |

Distances: colour W1 0.205, edge-energy gap 0.86, noise gap 0.27. Real imagery is warm and high-contrast with strong local edges. The synthetic chips are grey and flat. The colour and edge-energy gaps are what FRK-13 to FRK-16 should close.

### FRK-13: varied reflectance and tone

Per-scene class reflectance gains (from named sub-streams), per-building roof material and brightness (from the stable feature id), and a per-channel tone matched to the real chips (the real Khartoum chips have a strong green cast, so the synthetic tone is calibrated to the benchmark's inputs, not to natural colour). Colour W1 fell from 0.205 to 0.095. Reflectance variation alone did not move it (0.213); the tone did. Edge-energy gap 0.90 and noise gap 0.47 remain, which is the work in FRK-14 to FRK-16.

### FRK-14: texture below the 10 m grid

A `value_noise` kernel (`kernels/noise.py`, golden-tested) feeds a four-octave texture field (40 m, 10 m, 2.5 m and 0.8 m cells) that modulates albedo per class. Measured on the same three regions:

| Texture | Colour W1 | Edge-energy gap | Noise gap |
|---|---|---|---|
| 0 (FRK-13) | 0.095 | 0.90 | 0.47 |
| 0.3 | 0.084 | 0.77 | 0.47 |
| 0.7 (synth default) | 0.073 | 0.61 | 0.44 |

Edge energy is still well short of real imagery, whose edges come from objects (cars, trees, roof detail) as much as from texture. Sensor effects (FRK-16) and object detail are the remaining levers.

### FRK-15: sun, haze and soft shadows

Haze (exponential transmittance, monotonic in optical depth, blending toward a sky colour) and soft shadow edges (a Gaussian blur kernel in `kernels/filters.py`) are in the renderer, off by default. Sun azimuth and elevation were already per-render parameters. Measured on the same three regions with haze 0.1 and penumbra 0.3 m:

| | Colour W1 | Edge-energy gap | Noise gap |
|---|---|---|---|
| FRK-14 (no haze, hard shadows) | 0.073 | 0.61 | 0.44 |
| With haze and penumbra | 0.080 | 0.66 | 0.68 |

Both lower contrast, and the real Khartoum chips are high-contrast, so on this gauge they move away from the real data. They stay off by default. They are for randomisation ranges (FRK-17), and the ablation (FRK-18) will say whether they help the downstream model.

### FRK-16: sensor model

`fraktal.sensor` applies a Gaussian PSF blur (sigma in output pixels) to the region and additive Gaussian noise per tile, drawn from a per-tile named sub-stream. GSD is the render grid's `gsd_m`, so 0.5 m and 3 m renders keep their masks aligned (tests at both). With the synth defaults (PSF 0.8 px, noise 8 DN) on the same three regions:

| | Colour W1 | Edge-energy gap | Noise gap |
|---|---|---|---|
| FRK-14 | 0.073 | 0.61 | 0.44 |
| FRK-16 (PSF 0.8, noise 8) | 0.070 | 0.05 | 0.13 |

Caution: edge energy and the Laplacian noise statistic respond directly to added noise, so they will drop whether or not the model gets better at the task. The gauge says the chips now have real-like fine-scale energy, not that they look real. Only the benchmark can tell (FRK-18).

### FRK-17: randomised realism parameters

`fraktal synth --randomise` draws each scene's sun azimuth and elevation, haze, penumbra, PSF, noise, texture, reflectance variation, per-channel tone jitter and render seed from default ranges (`fraktal.randomise.Ranges`). A scene is a rendered region, which is then tiled, so parameters vary per region rather than per tile. Every component has its own named sub-stream (`realism:<scene>:<component>`), so editing one range never changes another's draw, and each region's parameters are written to `chips.json` as `region_params`. The ranges are initial guesses, not tuned values.

## Update 2026-11-03: V3 realism ablation results (FRK-18)

Seven cumulative variants (`fraktal.ablation.VARIANTS`), seeds 0 and 1, 3000 steps per arm, A40, about 2 hours 10 minutes. Pixel IoU on the 202 held-out tiles; arm 3 is synthetic pretraining then the real 10% subset. Means over two seeds. Arm 1 (real 10%) is 0.6278 and arm 2 (real 100%) is 0.6373.

| Variant (cumulative) | Seed 0 | Seed 1 | Mean | vs arm 1 (points) | Zero-shot IoU | Colour W1 |
|---|---|---|---|---|---|---|
| v0 flat (V1) | 0.614 | 0.611 | 0.612 | −1.54 | 0.003 | 0.206 |
| v1 + tone | 0.606 | 0.619 | 0.613 | −1.52 | 0.031 | 0.112 |
| v2 + reflectance variation | 0.607 | 0.609 | 0.608 | −1.96 | 0.224 | 0.094 |
| v3 + texture | 0.618 | 0.623 | 0.621 | −0.72 | 0.314 | 0.075 |
| v4 + sensor (PSF, noise) | 0.608 | 0.624 | 0.616 | −1.19 | 0.199 | 0.072 |
| v5 + haze, soft shadows | 0.610 | 0.626 | 0.618 | −1.01 | 0.278 | 0.077 |
| v6 randomised per region | 0.624 | 0.630 | 0.627 | −0.09 | 0.265 | 0.080 |

**The Q14 bar is not met.** No variant beats arm 1; the best (randomised) is level with it. The revised bar's precondition also fails: arm 2 beats arm 1 by 0.95 points, not 10, so this test cannot discriminate any data source. Differences under about 1 point are within seed noise (arm 3 moved up to 1.5 points between seeds).

**What the ablation does show**

- Zero-shot IoU went from about 0 to 0.2 to 0.3. The synthetic chips carry real signal now, but fine-tuning on 10% real data removes the advantage.
- Texture is the most consistent single gain (+0.8 points over flat, same direction on both seeds).
- Randomising the parameters per region is the best variant and is level with arm 1. Variety seems to matter more than any single component.
- Tone alone, and the sensor model, did not help the downstream score. The sensor model moved the gauge's edge and noise gaps to 0.04 and 0.12 while arm 3 stayed flat, which confirms those gauge statistics respond to noise without implying realism.
- Peak GPU memory 1.48 GB.

**Consequence.** Realism work through V3 closed most of the zero-shot gap, but the benchmark as set up (10% of about 1,000 tiles) cannot show a gain, because real data is not scarce enough. A decisive test needs a smaller real fraction, where arm 2 beats arm 1 by 10 or more points. Until then the premise is not refuted, and also not supported. V4 should wait for that measurement. (Superseded by ADR-0009: V4 is not held; the 1% run below made the test decidable.)

### FRK-21: bright and dark extremes, clipping and no-data edges

Four renderer parameters, all off by default: `ambient` (light left inside shadows), `contrast` (stretch about mid-grey, which clips the brightest and darkest pixels), `bright_roofs` (a share of buildings whose roofs saturate) and `blank_prob` (a tile may have a no-data region cut off one side, like a scene footprint edge). Blank regions are black, carry no labels and are excluded from the independent mask check, so the label guarantees hold. A new ablation variant, `v7_extremes`, adds them to the atmosphere variant (the randomised variant is unchanged, so earlier results stay comparable). The gauge gained contrast, clipped-share and strong-edge statistics. Measured on the same three regions:

| | Atmosphere variant | + extremes | Real |
|---|---|---|---|
| Grey contrast (std) | 25.1 | 36.8 | 35.6 |
| Clipped pixels (share at 0 or 255 in any channel) | 0.07% | 10.7% | 21.5% |
| Strong edges (share of steps above 20 DN) | 0.8% | 1.2% | 4.9% |
| Colour W1 | 0.077 | 0.064 | |

Contrast is matched and the clipped share is about halfway. Strong edges barely move, so the edge tail is an object-level gap, not a tone one (FRK-22). Whether extremes help the downstream model is untested until the next ablation run includes `v7_extremes`.

### FRK-22: trees, vehicles and roof detail

`fraktal.objects` places trees (domes, 1.5 to 4 m radius, 4 to 9 m tall) and vehicles (4.5 by 1.9 m, 1.5 m tall, five colours) deterministically from named sub-streams (`objects:trees`, `objects:cars`), on land-cover classes where they plausibly stand and never on buildings. A numba kernel (`kernels/objects.py`, golden-tested) paints them. They join the surface model, so they cast shadows, and the shading, so they show in the image. They are render-only decoration, not world features: no IDs, no labels. Roofs get ridge and corrugation stripes whose direction, period and phase come from the building's stable id. New variant `v8_objects` adds these to `v7_extremes`. Same three regions, gauge:

| | v7 extremes | v8 objects | Real |
|---|---|---|---|
| Strong-edge share (steps above 20 DN) | 1.2% | 2.1% | 4.9% |
| Strong-edge gap | 0.76 | 0.58 | |
| Grey contrast (std) | 36.8 | 38.5 | 35.6 |
| Colour W1 | 0.064 | 0.065 | |

The strong-edge tail moves but stays under half of real. Vehicles only stand on built-up and bare ground, and the world has no roads, so they are scattered rather than lined up on streets; a road network (V2) would make them realistic. Downstream effect untested until an ablation includes `v7_extremes` and `v8_objects`.

## Update 2026-11-03: scarce-data benchmark at 1% real (FRK-19)

At 1% real (8 tiles) the test can discriminate. Three seeds, 3000 steps per arm, A40, `fraktal ablate --fraction 0.01`. Real 1% scored 0.515 and real 100% scored 0.642, a gap of 12.8 points (the revised bar needs 10 or more), so the Q14 bar applies here.

| Variant | Seed 0 | Seed 1 | Seed 2 | Mean | vs real 1% (points) | Share of real 100% | Zero-shot | Bar |
|---|---|---|---|---|---|---|---|---|
| flat (V1) | 0.520 | 0.497 | 0.445 | 0.487 | −2.73 | 76% | 0.000 | missed |
| + texture (cumulative) | 0.529 | 0.528 | 0.472 | 0.510 | −0.51 | 79% | 0.301 | missed |
| randomised per region | 0.551 | 0.514 | 0.446 | 0.504 | −1.07 | 78% | 0.264 | missed |

**The Q14 bar is missed.** It needed arm 3 at least 5 points above real 1% and at least 90% of real 100%. The best variant is 0.5 points below real 1% and at 79% of real 100%. Synthetic pretraining on these chips does not help even when real data is scarce. Real 1% itself varies a lot with the subset drawn (0.541, 0.527, 0.477), so differences of about 3 points between variants are within noise, and arm 3 beat arm 1 in only 1 of 9 seed and variant pairs (randomised, seed 0).

**Consequence.** The result agrees with the 10% run: realism work through v6 raised the synthetic-only score from 0 to about 0.3 but did not make pretraining pay in the fine-tuned model. This is the risk the plan names (the sim-to-real gap is larger than randomisation can close). ADR-0009 sets the response: keep building for coverage and use these measurements to steer. The extremes (v7) and objects (v8) variants were not part of this run. They are the remaining realism levers on the list, and a further run including them at 1% would show whether they change the answer.

**Operational note.** The relay transfer failed ("room not ready"), so the pod waited out the transfer timeout (about 25 minutes, roughly $0.25) before the wrapper terminated it. The numbers were recovered from the report printed in the pod logs, as designed. Total cost about $1.1.

### FRK-23: palette fitted from the real train chips

`fraktal palette` fits class colours and roof colours from the real train chips only (never test tiles): WorldCover classes warped onto each tile's grid give per-class mean colours away from buildings, and the pixels under building labels are clustered (seeded k-means, four colours with their shares). Values are stored as albedo, chip colour divided by the render tone, so the tone gain still applies on top. `fraktal synth --palette <file>` uses them; without a palette the hand-picked colours are unchanged. Fitted on 162 sampled train tiles: six classes had enough pixels (tree cover, grassland, cropland, built-up, bare ground, water) and four roof colours came out, from dark grey to very bright (which saturate, as real bright roofs do). Same three regions with `v8_objects` settings:

| | Hand-picked colours | Fitted palette | Real |
|---|---|---|---|
| Colour W1 | 0.065 | 0.044 | |
| Channel means (R, G, B) | 77, 115, 16 | 87, 129, 22 | 88, 118, 34 |
| Grey contrast (std) | 38.5 | 43.7 | 35.6 |
| Strong-edge share | 2.1% | 2.7% | 4.9% |

Colour distance falls by a third and red and green now match. Blue is still low (22 against 34) and contrast now overshoots, because the extremes settings (FRK-21) were tuned against the old colours. Re-tuning contrast against the palette, and a blue-channel fix, are the obvious next steps. Downstream effect untested.

## FRK-30: procedural buildings, reviewed without training

Per [ADR-0009](0009-build-for-coverage-and-measure-to-steer.md) and the rule to train only at big
milestones, this was judged with the gauge, distribution plots and chip strips, not a GPU run.
`fraktal synth --procedural` renders each region over its own generated world. The job ran on a
CPU pod (`scripts/runpod/review.sh`, 16 vCPU, about $0.56/hr), 6 regions per set, palette fitted
from train chips.

| set | chips | colour W1 | edge gap | noise gap | contrast gap | strong-edge gap |
|---|---|---|---|---|---|---|
| skeleton v8_objects | 7 | 0.039 | 0.009 | 0.132 | 0.318 | 0.425 |
| procedural v8_objects | 41 | 0.068 | 0.121 | 0.181 | 0.109 | 0.724 |
| skeleton v6_randomised | 7 | 0.058 | 0.006 | 0.060 | 0.149 | 0.630 |
| procedural v6_randomised | 41 | 0.076 | 0.005 | 0.010 | 0.507 | 0.698 |

What the plots show:
- Procedural labels are complete everywhere, so 6 regions gave 41 chips against 7 for the skeleton
  (the training-tile filter drops the rest). Coverage is the point of the procedural path.
- Procedural scenes are mostly bare ground with scattered trees and a few buildings. Real chips
  are dense blocks with roads and courtyards. Parcels line few arterials (FRK-31), and roads are
  not painted onto the render (FRK-32).
- Colour is close but a little worse than skeleton. Strong-edge share is the largest gap in every
  set, as it was before. Randomised scenes are low in contrast (0.51 gap), so the randomised
  ranges need re-tuning against the palette.
- Next generator work: building density, painted roads, then contrast ranges.

## FRK-31: fairer arterials and parcels that tolerate curves

Arterials get Douglas-Peucker simplification and three rounds of corner cutting (end points kept;
a smoothed line that would touch water falls back to the raw path), and local streets start on the
smoothed line with the junction added as a shared vertex. The bigger win was in the parcel check:
a straight parcel edge clips a sliver off a curved carriageway, and the old test rejected any
overlap above 1e-6 m², so curved roads lost most parcels. The check now allows 3% of the parcel
area. Parcels per 2 km world, FRK-27 against now (seeds 0, 1, 2): 478 to 705, 284 to 493, 220 to 532.
Buildings: 321 to 515, 215 to 357, 155 to 387. Arterial-only parcel counts are mixed (214 to 198,
175 to 112, 108 to 168 against raw paths with the new tolerance), so the gain comes from the check.

## FRK-32: roads on the render, vehicles on roads

`render_region(..., roads=layer)` paints each road as a ground class by width and class (asphalt
for arterials, dusty tan for local streets). Buildings paint over them, trees keep off them, and
the land-cover tile is untouched, so labels are byte-identical with and without roads. Without a
road layer the render is unchanged. Vehicles are placed on road pixels (14 per hectare of road at
density 1) and oriented along the road from the gradient of the blurred road mask. `synth
--procedural` passes the world's roads automatically. Lane markings and junction geometry are not
modelled.

## FRK-33: contrast pivot and wide randomised extremes

The CPU review showed two things: randomised scenes were too flat (grey-level std 17 against 34
for real chips) and the blue channel was too low in the v7 and v8 sets (mean 15 against 36).
Both have one cause. Contrast stretched about mid-grey, so a dark channel was pushed toward zero,
and the randomised ranges did not include any extremes. Contrast now stretches about each
channel's own mean, which keeps the colour balance while widening the spread. The randomised
ranges gain ambient, contrast, bright-roof share and no-data chance, neutral by default (so the
original v6 draws are unchanged), plus `randomise.WIDE` and the variant `v9_randomised_wide`
(objects, roof stripes, extremes drawn per region). Earlier v7 and v8 chips were rendered with
the old pivot, so rendered sets from before FRK-33 are not comparable at equal contrast.

## FRK-34: review re-run after roads, parcels and wide extremes

Same CPU-pod review as FRK-30 (6 regions, palette from train chips, one pod, about $0.20),
now with FRK-31 to FRK-33 in place. Real chips: grey-level std 34.2, channel means
92 / 124 / 36. "proc" is procedural buildings and roads, "skel" is the real-footprint skeleton.

| set | chips | colour W1 | edge gap | noise gap | contrast gap | strong-edge gap | grey std | channel means |
|---|---|---|---|---|---|---|---|---|
| proc v8_objects | 41 | 0.040 | 0.077 | 0.140 | 0.016 | 0.658 | 34.8 | 95 / 132 / 38 |
| proc v9_randomised_wide | 41 | 0.061 | 0.101 | 0.018 | 0.113 | 0.332 | 38.1 | 107 / 147 / 40 |
| skel v8_objects | 7 | 0.046 | 0.041 | 0.100 | 0.372 | 0.374 | 46.9 | 103 / 140 / 44 |
| skel v9_randomised_wide | 7 | 0.064 | 0.105 | 0.010 | 0.530 | 0.181 | 52.3 | 110 / 145 / 45 |

Against FRK-30 (procedural sets): contrast gap 0.109 to 0.016 (v8) and 0.507 to 0.113 (randomised);
colour distance 0.068 to 0.040 (v8); blue mean 15 to 38 for v8, against 36 real; strong-edge gap
0.72 to 0.66 (v8) and 0.70 to 0.33 (randomised). The skeleton sets over-shoot on contrast because
their regions are dense with large roofs, which the procedural scenes are not.

What is still wrong, from the chip strip:
- Procedural scenes remain sparse: bare ground with scattered trees and few buildings, where real
  chips are dense blocks. Building density per parcel and parcel coverage need raising.
- Trees are scattered uniformly; real trees line streets and cluster around buildings.
- Edge-energy gap grew (0.12 for v8) because the scenes have fewer hard edges, not more blur.
- Roof and ground textures are still smooth next to real imagery (see the 3D assets note,
  docs/research/3d-assets-and-realism.md).

## FRK-44: biome sheet and per-biome review (no training)

Two free checks, both on CPU, nothing trained. Pod costs: one that never started its container
(terminated after 15 minutes, about $0.14) and one 8-minute run, about $0.10 in all.

**Overview sheet** (`scripts/biome_overview.py`, committed as `docs/img/biome-overview.jpg`): one clear-sky
1 km window per biome, same seed. It shows each generator doing what it was built to do:
dense_urban has gridded blocks of tall roofs; oil_palm_plantation has staggered palm rows with a
track and shadows; paddy_terraces has tiled blocks in three states; mangrove_coast has sea, a mud
flat, a mangrove belt and creeks, with ponds behind it. informal_settlement shows up only along
one road in this window, so its density is low. The land cover under everything is still
pixelated at the 5 m cell size.

**Gauge against the Khartoum chips** (`scripts/runpod/biomes.sh`; 3 regions a biome, up to 30
chips; tropical biomes drawn from `v10_tropical`, the Khartoum reference from `v9_randomised_wide`).
The real chips are arid Khartoum, so for the tropical biomes these numbers measure distance from
Khartoum, not from the tropics. Without a tropical set (Q15, FRK-43) they cannot say more.

| set | colour W1 | edge gap | noise gap | contrast gap | strong-edge gap |
|---|---|---|---|---|---|
| khartoum, v9 | 0.074 | 0.051 | 0.048 | 0.136 | 0.417 |
| dense_urban | 0.292 | 0.030 | 0.043 | 0.095 | 0.547 |
| informal_settlement | 0.293 | 0.011 | 0.045 | 0.039 | 0.612 |
| oil_palm_plantation | 0.280 | 0.000 | 0.047 | 0.015 | 0.652 |
| paddy_terraces | 0.270 | 0.024 | 0.053 | 0.051 | 0.730 |
| mangrove_coast | 0.293 | 0.003 | 0.048 | 0.033 | 0.656 |

Reading it: colour distance is about 0.28 for every tropical biome against 0.07 for the Khartoum
reference, which is what the greener, hazier tone and cloud should do against arid reference
chips; the tone is an untuned guess. Noise, edge and contrast gaps are small everywhere. The
strong-edge gap is large for all of them (0.4 to 0.7): synthetic scenes still have too few hard
edges.

**Problems found, not fixed:**
- The pod chip strip is a weak review. Every biome uses the same seeds, so ground, trees and
  clouds repeat across rows, and uniform chip picks land mostly on bare ground or sea. The strip
  now takes the busiest chips, which skips smooth ones (sea, cloud). The overview sheet above is
  the better tool.
- `v10_tropical` cloud and haze hide large parts of many chips (several are near-white), which
  also depresses the gauge's edge numbers.
- Chips dropped from the Khartoum set (9 of 30) are the benchmark-leakage guards, not label
  errors: past runs record only "outside labelled training tiles" and "overlaps a held-out test
  tile". That is intended.
- Fixed in FRK-55: `dense_urban` lost tiles in label validation because snapped footprints put
  pixel centres exactly on an edge, which the rasteriser paints and the independent check did not.
  The check now accepts a label between the strict and the edge-touching mask.
- Scenes outside the built areas are still empty ground.
