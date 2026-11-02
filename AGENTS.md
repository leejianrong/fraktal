# Fraktal: agent brief

Procedural world generation for synthetic satellite data. Python first, with hot kernels
isolated so a Rust or C++ core can replace them later (ADR-0005).

## Build status

Trust the code over the docs where they disagree, and update the docs in the same PR.

| Area | State |
|------|-------|
| Plan, slices, ADRs | Written (PLAN.md, SLICES.md, docs/adr/) |
| Deterministic RNG streams and feature IDs (`src/fraktal/ids.py`) | Built, tested |
| CLI (`src/fraktal/cli.py`) | Skeleton: `version` works; five stage commands return exit 3. JSON envelope, exit codes 0/2/3/4 |
| World model v0 (`src/fraktal/world.py`) | Building and landcover layers, provenance, GeoParquet round-trip, `world_id` |
| Ingest, world model, renderer, benchmark | Not started (SLICES V1 onward) |

## Commands

```bash
make install           # uv sync --frozen
make check             # pre-push gate: ruff, mypy --strict, fast tests
make test              # fast layer only (no infra, no network)
make test-integration  # subprocesses, external services, network
make fmt               # apply ruff fixes and formatting
```

Run `make check` before proposing a push. CI runs the same checks as separate jobs, plus
integration, a dependency audit and a secrets scan.

## Conventions

- **Branch per slice, PR only.** Branch off fresh `main`, one branch per SLICES.md slice. No direct
  pushes to `main`. CI must be green before merge.
- **Jian is the sole author.** Never add co-author trailers or tool attributions to commits, PRs
  or files.
- **Tests are layered.** Fast tests in `tests/unit` (temp files are fine). Anything needing subprocesses, external
  services or network goes in `tests/integration` and carries `pytestmark = pytest.mark.integration`.
- **Every bug becomes a failing test first.** Assert observable behaviour, not internals.
- **Determinism is a contract (ADR-0003).** Randomness only via `fraktal.ids.substream`. Never use
  a global RNG, wall-clock time or unordered iteration in world generation.
- **Kernels (ADR-0005).** Heavy numeric code lives in `src/fraktal/kernels/`: arrays in, arrays
  out, no paths, no global state, with a golden-output test. No pure-Python hot loops: vectorise
  or use numba.
- **Stages talk through files** (GeoParquet, Cloud-Optimized GeoTIFF), not in-memory objects.
- **Secrets stay out of git.** Credentials come from environment variables only.
- **Licences matter.** Check a dependency's or dataset's licence before adding it (ADR-0007).

## Layout

- `src/fraktal/`: package. `ids.py` RNG and IDs, `cli.py` entry point, `kernels/` numeric kernels.
- `tests/unit`, `tests/integration`: the two test layers.
- `docs/adr/`: decisions. Read the relevant ADR before changing what it covers.
- `PLAN.md`, `SLICES.md`, `QUESTIONS.md`: plan, build order, decision register.

## Docs map

Scope and requirements: `PLAN.md`. What to build next and how it is accepted: `SLICES.md`.
Why a choice was made: `docs/adr/`. Later bets (robotics/drone views, SAR, disaster pairs,
agent simulation, digital twins): `PLAN.md` section "Later bets".
