# ADR-0002: Worlds are procedural but conditioned on real geography, starting in tropical SE Asia

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

Purely procedural worlds have no guaranteed match to the distribution of real satellite imagery, so models trained on them transfer poorly. Real geographic data (elevation, land cover, building footprints, roads) is freely available but sparse and uneven, and cannot be varied. Tropical SE Asia (dense urban, informal settlements, plantations, paddy, mangrove, frequent cloud cover) is underserved by common benchmarks and is where the user is based.

## Decision

A world is built in layers. Real data provides the skeleton where it exists: Copernicus DEM GLO-30 for elevation, ESA WorldCover for land cover, OpenStreetMap for roads and building footprints. Procedural generation fills gaps and creates variation: re-seeded settlement patterns, building placement, vegetation, and wholly synthetic regions. Each world records which layers were real and which were procedural. The first regional focus is tropical SE Asia.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Pure procedural | Unknown domain gap, nothing to validate realism against |
| Real data only (re-rendering existing footprints) | Cannot create new scenes, no variation, and ODbL obligations apply to everything |
| Learned generative world (e.g. diffusion over layouts) | Needs training data we do not have, harder to control and debug |

## Consequences

Realism and anchoring come from real data, volume and variety from procedural fill. Provenance per layer must be tracked, because OSM-derived outputs carry ODbL share-alike obligations. Whether rendered images count as a "produced work" or a derivative database is a legal question to settle before selling datasets (ADR-0007, PLAN risks). SE Asia focus shapes the biome rules, so a temperate-climate customer would need a new biome pack.
