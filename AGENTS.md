# Fraktal: agent brief

Procedural world generation for synthetic satellite data. Python first, with numeric kernels
isolated so a Rust or C++ core can replace them later. Early stage: check the Pandan board
(key `FRK`) for what is built, and trust the code over the docs where they disagree, then
fix the docs in the same PR.

## Commands

```bash
make install           # uv sync --frozen
make check             # the pre-push gate: ruff, mypy --strict, fast tests. Run before pushing
make test-integration  # subprocesses, external services, network
make fmt               # apply ruff fixes and formatting
```

## Rules that are not obvious from the code

- **Jian is the sole author.** No co-author trailers or tool attributions in commits, PRs or files.
- **Branch per slice, PR only, rebase-merge** so each story keeps its own commit. CI must be green.
- **Determinism is a contract** ([ADR-0003](docs/adr/0003-deterministic-worlds-with-stable-ids.md)). Randomness only via
  `fraktal.ids.substream`. No global RNG, wall-clock time or unordered iteration in world generation.
- **Kernels** ([ADR-0005](docs/adr/0005-python-first-with-a-kernel-seam-for-a-later-rust-core.md)) live in
  `src/fraktal/kernels/`: arrays in, arrays out, no paths or global state, with a golden-output test.
  No pure-Python hot loops: vectorise or use numba.
- **Stages talk through files** (GeoParquet, Cloud-Optimized GeoTIFF), not in-memory objects.
- **No benchmark leakage.** Held-out test-tile footprints must never enter a world used for
  synthetic pretraining ([ADR-0006](docs/adr/0006-success-is-a-downstream-benchmark.md)).
- **Pin GitHub Actions to full release tags** (`@v10.3.3`). Floating majors may not exist.
- **Check licences** before adding a dependency or dataset ([ADR-0007](docs/adr/0007-sell-datasets-first-keep-the-engine-private.md)). Assets go in `assets/` with a manifest row, owned or CC0 only (`fraktal assets check`, enforced in CI; see `assets/README.md`).
- **Tests are layered:** `tests/unit` (temp files fine) and `tests/integration`
  (`pytestmark = pytest.mark.integration`). Fix a bug with a failing test first.
- **Long jobs report progress.** Anything that can run for minutes emits `fraktal.log.emit` lines (stderr, never stdout) so a remote run can be watched.
- **Renderer ownership** ([ADR-0012](docs/adr/0012-blender-is-the-product-renderer-godot-is-a-demo-viewer.md)): Blender is the product renderer and gets the realism work; 2.5D is maintenance; Godot is a demo viewer, not a capture path. Judge realism by looking at samples next to the real reference board (eval only, never committed).
- **Benchmarks steer, they do not gate** ([ADR-0009](docs/adr/0009-build-for-coverage-and-measure-to-steer.md)). The working premise is that enough realism and generator degrees of freedom make synthetic data helpful. Keep building, keep measuring, record negative results, and widen the distribution rather than polishing one look.
- **Train only at big milestones.** Day to day, judge synthetic data with the free checks: `fraktal compare`, the distribution plots and chip strips, and looking at samples. Spend GPU on a training benchmark only when a major milestone lands (a new world generator, a large realism step, a dataset release).
- **Secrets** come from environment variables only. Never commit `.env` or tokens.

## Docs

`docs/RUNPOD.md` (GPU runs: launch, watch, tear down), `godot/README.md` (the real-time viewer: export a tile, install Godot, run it on WSL or Windows), `docs/PLAN.md` (scope and shape), `docs/SLICES.md` (build order and acceptance tests),
`docs/adr/` (read the relevant ADR before changing what it covers), `docs/QUESTIONS.md`.
Later bets (robotics/drone views, SAR, disaster pairs, agent simulation) are in
`docs/PLAN.md` under "Later bets".
