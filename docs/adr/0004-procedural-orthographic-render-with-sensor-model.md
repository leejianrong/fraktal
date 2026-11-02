# ADR-0004: Imagery is rendered procedurally from a surface model plus a sensor model; ML refinement only if the benchmark demands it

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

Satellite imagery for the MVP is near-nadir. That means a full 3D engine is unnecessary: a digital surface model (terrain plus building and canopy heights) and a per-class reflectance model are enough to produce optical imagery with correct shadows. Generative ML (diffusion conditioned on label maps) can look more realistic but gives labels that only approximately match pixels and needs training data. Drone and off-nadir views will need a real 3D renderer, but that is a later consumer.

## Decision

The renderer takes the world model and produces: a DSM raster, per-class reflectance per band (RGB and NIR), directional sun lighting with cast shadows from the DSM, atmosphere (haze, optional clouds), seasonal variation, and sensor effects (point-spread function, noise, GSD resampling). Labels (building footprints, land cover, roads) are rasterised from the same world at the same grid, so label and pixel always agree. Learned refinement of renderer output (a GAN or diffusion "style pass") is considered only if the downstream benchmark (ADR-0006) shows an unclosed gap that randomisation cannot close.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Full 3D engine (Blender, Unreal) from the start | Heavy, GPL/licence questions for Blender linking, overkill for near-nadir imagery |
| Generative-only (diffusion on label maps) | Labels no longer exact, needs real imagery to train, harder to control |
| Hybrid from day one | Two systems before knowing if either is needed |

## Consequences

Cheap, fast and exactly labelled, and the renderer can run on GPU through PyTorch or on CPU through numba. Off-nadir (oblique) views and anything with vertical structure detail (facades, roofs at angles) are not covered, and will need a 3D renderer later (robotics/drone ideas in PLAN). Realism is bounded by how good the reflectance and randomisation models are, which is exactly what the benchmark measures.
