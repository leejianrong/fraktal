# ADR-0001: Satellite/EO data is the MVP, built on a shared world model

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

Two use cases were on the table: synthetic satellite/geospatial training data, and large-scale agent simulation of cities. The EO market is proven (Another Earth sells synthetic satellite data, Bifrost sells synthetic 3D data for physical AI) and its output is directly measurable against public benchmarks. Agent simulation is a larger research surface with no equally clear first customer. Further ideas (robotics/drone views, SAR, disaster pairs, digital twins) all need the same underlying thing: a coherent, labelled world.

## Decision

The MVP delivers synthetic optical satellite imagery with perfect labels. The thing we build underneath is a **world model**: terrain, water, land cover, roads, buildings and land use, each feature with a stable ID. Renderers and simulators are consumers of the world model and never write to it. Agent simulation, drone views, SAR and disaster scenarios are later consumers.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Agent simulation first | No objective success test, long research tail, and it still needs the world model first |
| Two separate products | Duplicates the hardest part (a consistent world) and doubles maintenance |
| Rendering-only tool, no world model | Labels would be derived from images instead of being exact, which throws away the main advantage over real data |

## Consequences

Labels are exact by construction. Every later use case reuses the world model, so its schema has to be right early: roads, buildings, land use and population-relevant attributes are in the model from day one even though only buildings and land cover are rendered at first. The cost is that the world model is design effort the MVP does not strictly need to show a demo, and slice ordering has to keep the demo loop short anyway (see SLICES V1).
