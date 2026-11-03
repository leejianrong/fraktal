# Contributing

## Set up

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install   # locked dependencies
make install-train   # also PyTorch, only for the benchmark (large)
make hooks     # pre-push gate (runs `make check`)
make           # list all targets
```

## Before you push

`make check` runs lint, strict type-checking and the fast tests. CI runs the same checks as separate jobs, plus the integration layer, a dependency audit and a secrets scan. Skip the hook once with `git push --no-verify`.

## Workflow

- One branch per slice (see [docs/SLICES.md](docs/SLICES.md)), off fresh `main`. Changes land through a pull request with green CI. No direct pushes to `main`.
- Fix a bug by writing a failing test first.
- Heavy numeric code goes in `src/fraktal/kernels/` (see [ADR-0005](docs/adr/0005-python-first-with-a-kernel-seam-for-a-later-rust-core.md)). World generation must be deterministic ([ADR-0003](docs/adr/0003-deterministic-worlds-with-stable-ids.md)).
- Check a dependency's or dataset's licence before adding it.

## Tests

- `tests/unit`: fast, no network, no external services. Run by `make test`.
- `tests/integration`: subprocesses, external services, network. Mark with `pytestmark = pytest.mark.integration`. Run by `make test-integration`.
