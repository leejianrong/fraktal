# ADR-0008: V1 benchmark uses SpaceNet 2 (Khartoum), with the tropical test set deferred to V4

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

ADR-0006 requires a real building-segmentation dataset, chosen in SLICES V1 step 1. Candidates reviewed:

| Dataset | Licence (per public listings, verify at download) | Fit |
|---------|---------|-----|
| SpaceNet 2 (Vegas, Paris, Shanghai, Khartoum) | CC-BY-SA 4.0 | Satellite, labelled polygons, public S3 bucket, known baselines. Not tropical |
| Open Cities AI (Africa) | CC-BY 4.0 | Drone imagery at a much finer GSD than our 0.5 m target. Labels derived from OSM |
| WHU building dataset | Not verified | Satellite subset covers East Asia, not SE Asia |
| Google Open Buildings V3 | CC-BY 4.0 (polygons only) | No imagery, so not a benchmark by itself |
| Maxar Open Data derived sets (e.g. Ramp) | CC BY-NC 4.0 | Non-commercial, so excluded |

No open, labelled, image-plus-mask building dataset for SE Asia turned up in a first search.

SpaceNet 2's labelled train split is publicly listable on S3 (`s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/`). Sizes: Khartoum 5.1 GB, Paris 5.7 GB, Vegas 25.6 GB, Shanghai 25.1 GB. The test tarballs are separate; the train tarballs carry the labels, so we split train ourselves.

## Decision

- V1 benchmarks on **SpaceNet 2, Khartoum train** (smallest tarball, workable on the dev laptop), split into train and held-out test by tile.
- The dataset is used only for training and evaluation inside this project. We do not redistribute it or any derivative, so CC-BY-SA's share-alike terms are not triggered. The V5 licence review re-checks this before any sale.
- The SE Asia claim stays unmeasured until V4 finds or builds a tropical test set.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Open Cities AI | GSD mismatch with the 0.5 m target, labels come from OSM |
| Shanghai or Vegas | 5x larger download for no gain in V1 |
| Wait for a tropical set | Blocks the riskiest measurement on a dataset that may not exist |

## Consequences

V1 can start now with a real baseline. Khartoum is arid, with low-rise dense buildings, so results say nothing yet about tropical transfer. If the licence review finds SpaceNet unusable for commercial claims, the benchmark switches datasets, which costs days, not weeks, because the benchmark harness takes the dataset as a config.
