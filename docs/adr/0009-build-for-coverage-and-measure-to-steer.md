# ADR-0009: Build for coverage; measure to steer, not to gate

- Status: Accepted
- Date: 2026-11-03
- Deciders: Jian

## Context

V1 and V3 measured whether synthetic pretraining helps a real building-segmentation model (ADR-0006). The Q14 bar was missed at 10% real data (where the test could not discriminate) and at 1% real data (where it could). Realism work raised the synthetic-only score from about 0 to about 0.3 IoU on real tiles, but fine-tuned scores did not beat real data alone. The plan's open risk says that if V3 does not move the benchmark, the premise needs rethinking before V4, which would turn one benchmark result into a stop condition.

## Decision

Treat the premise as the working hypothesis of the project: with enough realism, and enough degrees of freedom in the procedural generation, a large synthetic dataset can cover a distribution wide enough to contain the real test and deployment data, and synthetic pretraining will then help. The open question is how much, not whether.

- **Keep building.** Realism, procedural world generation, new biomes and the 3D direction proceed. A missed benchmark bar is a finding to act on, not a reason to stop or to hold other slices.
- **Keep measuring.** Every slice that changes what the data looks like is measured: the realism gauge, the synthetic-only score, and the three-arm benchmark at the scarce-data fraction where it is decidable. Measurement steers what to build next. It does not gate other work.
- **Widen coverage deliberately.** Prefer changes that add independent degrees of freedom the generator can vary (appearance, sensor, scene content, layout, geography), and randomise over them, over polishing one fixed look. The target is a distribution that contains the real data, not one image that resembles it.
- **Improve until helpful.** The Q14 bar stays as the target. When it is missed, the next step is the largest measured gap, which may mean a different approach to that gap.

## Consequences

- V2 (procedural world), V4 (biome pack), the 3D direction (Q18) and the rest are not held for a premise review. Their order is set by what the measurements say is the largest gap and by what the dataset product needs.
- Ablation and benchmark runs continue as steering signals, with seeds enough to tell effects from noise. Results are recorded as they come, including negative ones.
- This does not claim the premise is proven. If measurements stay flat across many changes, that is evidence to change the approach (more diversity, learned refinement, different pretraining), not to stop.
