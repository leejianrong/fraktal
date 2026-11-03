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
