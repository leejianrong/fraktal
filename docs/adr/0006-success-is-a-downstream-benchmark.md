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

**Consequence.** Realism work through V3 closed most of the zero-shot gap, but the benchmark as set up (10% of about 1,000 tiles) cannot show a gain, because real data is not scarce enough. A decisive test needs a smaller real fraction, where arm 2 beats arm 1 by 10 or more points. Until then the premise is not refuted, and also not supported. V4 should wait for that measurement.

### FRK-21: bright and dark extremes, clipping and no-data edges

Four renderer parameters, all off by default: `ambient` (light left inside shadows), `contrast` (stretch about mid-grey, which clips the brightest and darkest pixels), `bright_roofs` (a share of buildings whose roofs saturate) and `blank_prob` (a tile may have a no-data region cut off one side, like a scene footprint edge). Blank regions are black, carry no labels and are excluded from the independent mask check, so the label guarantees hold. A new ablation variant, `v7_extremes`, adds them to the atmosphere variant (the randomised variant is unchanged, so earlier results stay comparable). The gauge gained contrast, clipped-share and strong-edge statistics. Measured on the same three regions:

| | Atmosphere variant | + extremes | Real |
|---|---|---|---|
| Grey contrast (std) | 25.1 | 36.8 | 35.6 |
| Clipped pixels (share at 0 or 255 in any channel) | 0.07% | 10.7% | 21.5% |
| Strong edges (share of steps above 20 DN) | 0.8% | 1.2% | 4.9% |
| Colour W1 | 0.077 | 0.064 | |

Contrast is matched and the clipped share is about halfway. Strong edges barely move, so the edge tail is an object-level gap, not a tone one (FRK-22). Whether extremes help the downstream model is untested until the next ablation run includes `v7_extremes`.
