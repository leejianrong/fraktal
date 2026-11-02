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
