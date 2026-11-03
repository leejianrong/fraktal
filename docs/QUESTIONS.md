# Questions

Statuses: `DECIDED` (user answered) · `ASSUMED` (default taken, correct it if
wrong) · `FORK` (waiting on the user) · `DEFERRED` (not needed this milestone).

Mode: fresh. Host: WSL2, 20 cores, 31 GB RAM, RTX 3080 Ti Laptop (16 GB VRAM), Python 3.10, Rust 1.97, Go 1.26.

## Open forks

None. Round 1 closed.

## Register

| ID | Question | Status | Answer or default | Landed |
|----|----------|--------|-------------------|--------|
| F1 | MVP use case | DECIDED | Satellite/EO first on a shared world model | ADR-0001 |
| F2 | Procedural vs real-conditioned | DECIDED | Hybrid: real skeleton, procedural fill | ADR-0002 |
| F3 | Image generation approach | DECIDED | Procedural render plus sensor model; refinement only if needed | ADR-0004 |
| F4 | Definition of success | DECIDED | Downstream building-segmentation benchmark, three arms | ADR-0006 |
| F5 | Core language | DECIDED (changed) | Python first, with a kernel seam for a later Rust/C++ core. The user overrode the Rust recommendation | ADR-0005 |
| F6 | Commercial shape | DECIDED | Datasets and custom projects first; engine private | ADR-0007 |
| F7 | Starting geography | DECIDED | Tropical SE Asia | ADR-0002, PLAN §Scope |
| Q1 | Primary user | ASSUMED | EO ML teams and researchers; second user is coding agents via CLI | PLAN §Users |
| Q2 | Identity and reproducibility | ASSUMED | `(seed, config, version)`; hashed stable feature IDs | ADR-0003 |
| Q3 | Storage and formats | ASSUMED | Local FS; GeoParquet, COG, STAC, COCO | PLAN §Implementation |
| Q4 | Concurrency | ASSUMED | Single-writer batch; immutable content-addressed outputs | PLAN §Implementation |
| Q5 | Interfaces | ASSUMED | CLI first, Python library second, no custom UI | PLAN §Affordances |
| Q6 | Failure behaviour | ASSUMED | Hard errors; invalid tiles dropped and listed in the manifest | PLAN §Implementation |
| Q7 | Dependencies and licences | ASSUMED | GDAL/PROJ, PyTorch, numba, rasterio; Blender only as optional subprocess; data: Copernicus DEM, WorldCover, OSM. Licence review gates first sale | ADR-0007, SLICES V5 |
| Q8 | Runtime | ASSUMED | Local WSL2 on the RTX 3080 Ti; RunPod for scale-out | PLAN §Implementation |
| Q9 | Sensor modality | ASSUMED | RGB+NIR optical at 0.5 m and 3 m; SAR deferred | PLAN §Scope |
| Q10 | Security and privacy | ASSUMED | Nothing sensitive now; aggregate-only populations later | PLAN §Implementation |
| Q11 | Versioning | ASSUMED | Generator version and config hash in every manifest; datasets immutable | ADR-0003 |
| Q12 | Agent simulation timing | DEFERRED | After the EO milestone; world schema ready for it (R8) | PLAN §Later bets |
| Q13 | Sim-to-real gap handling | ASSUMED | Domain randomisation first, learned refinement only if needed | ADR-0004 |
| Q14 | Benchmark bar | MISSED at 1% real (REVISED bar) | Same thresholds (arm 3 beats arm 1 by 5+ IoU points, at least 90% of arm 2), but only in a regime where arm 2 beats arm 1 by 10+ points. At 10% real the gap is under 1 point, so the V1 run cannot judge it (arm 3 was 0.7 to 1.6 points below arm 1). V3 ablation (seven realism variants, two seeds) also ended at or below arm 1 (best −0.09 points) with arm 2 only 0.95 points above arm 1, so it still cannot judge. At 1% real the test is decidable (real 100% beats real 1% by 12.8 points) and the bar is missed: best variant 0.5 points below real 1%, 79% of real 100%. Per ADR-0009 this steers the work rather than gating it; v7 and v8 variants not yet measured downstream | ADR-0006 |
| Q15 | Benchmark dataset | DECIDED | SpaceNet 2 Khartoum train for V1 (2026-11-02); a tropical test set is sought in V4 | ADR-0008, SLICES V4 |
| Q16 | Python stack | ASSUMED | NumPy, numba, PyTorch, rasterio, shapely, geopandas, `uv` | ADR-0005 |
| Q17 | Later bets (robotics/drone, SAR, disasters, agent sim, digital twins, refinement) | DEFERRED | Recorded in PLAN §Later bets; each needs its own ADR when started | PLAN §Later bets |
| Q18 | Whole procedural 3D worlds, a reusable 3D asset library, and imagery and video from any viewpoint | PROPOSED (ADR-0010; research in docs/research/3d-assets-and-realism.md) | Long-term direction recorded in PLAN §Later bets (item 0). Open: asset sourcing and licences (build, commission or buy), renderer choice (Blender subprocess, GPU renderer or engine), video and camera-trajectory model, and what 2.5D work carries over. ADR-0010 proposes a headless Blender renderer, our own building grammar and an owned or CC0 asset library; slices are filed once it is accepted. Not held for the V3 benchmark (ADR-0009) | PLAN §Later bets |

## Coverage

| Category | Covered by |
|----------|-----------|
| Primary user and actors | Q1 |
| Scope boundary | F1, Q9, Q12, Q17 |
| Data model and identity | Q2, F2 |
| State and storage | Q3 |
| Concurrency and conflict | Q4 |
| Interfaces and contracts | Q5 |
| Failure behaviour | Q6 |
| External dependencies | Q7, Q16, F5 |
| Runtime and deployment | Q8 |
| Measurable success | F4, Q14, Q15 |
| Security and secrets | Q10 |
| Versioning and migration | Q11 |
| Domain: sim-to-real gap | Q13, F3 |
| Domain: commercial/licensing | F6, Q7 |
