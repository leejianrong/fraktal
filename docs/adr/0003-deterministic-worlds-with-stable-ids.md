# ADR-0003: A world is `(seed, config, generator_version)` and is deterministic, with stable feature IDs

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

Datasets must be reproducible for debugging, for audits by customers, and for regenerating a dataset at a different resolution or sensor without changing the scene. Later consumers (simulation, drone views) must refer to the same buildings and roads as the satellite renderer, including after export and re-import.

## Decision

- A world is fully determined by `(seed, config, generator_version)`. Identical triples give byte-identical vector output.
- Randomness comes only from named sub-streams derived from the seed (for example `terrain`, `roads`, `buildings`), so adding a new feature type does not shift existing ones.
- Each feature's ID is a hash of (world ID, feature kind, local coordinates or parent ID and index). IDs do not depend on generation order and survive export and re-import.
- Generated datasets are immutable and content-addressed. Their manifest records the generator version and config hash.
- Breaking changes to world output bump the generator's major version.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Sequential integer IDs | Break when generation order or feature counts change |
| Store worlds only, regenerate nothing | Loses reproducibility, large storage |
| Single global RNG | One new feature type reshuffles the entire world |

## Consequences

Reproducibility is a testable contract (generate twice, compare hashes). Floating-point determinism across machines and across a later Python-to-Rust swap is hard and is limited here to vector output; rendered rasters are reproducible per environment, not guaranteed bit-exact across hardware. Every kernel that gets reimplemented later must pass the same golden-hash tests, or consciously bump the version.
