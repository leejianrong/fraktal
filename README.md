# Fraktal

Procedural world generation for synthetic satellite and geospatial training data. Fraktal builds deterministic, labelled worlds, conditioned on real geography and filled in procedurally, and renders them as satellite imagery with pixel-exact labels.

**Status: early.** The plan is written and the foundations are in place (CLI, world model, stable IDs, data ingest, a crude renderer and the benchmark harness). Whether synthetic pretraining helps is not measured yet. The first milestone is a benchmark that shows whether training on synthetic imagery improves a real building-segmentation model.

```mermaid
flowchart LR
    R["Real data<br/>DEM, land cover, footprints"] --> W
    P["Procedural generators<br/>terrain, roads, buildings"] --> W
    W["World model<br/>stable IDs, provenance"] --> I["Imagery + exact labels"]
    I --> B["Benchmark<br/>does it help a real model?"]
    W -.-> L["Later: SAR, drone views,<br/>agent simulation"]
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone https://github.com/leejianrong/fraktal.git
cd fraktal
uv sync
uv run fraktal version
```

## Usage

Every command can print a machine-readable result with `--json`:

```console
$ uv run fraktal version --json
{"ok": true, "data": {"version": "0.0.1"}, "error": null}
```

The V1 pipeline, in order (data lands under `data/`, which git ignores):

```console
$ uv run fraktal ingest                  # SpaceNet 2 Khartoum (5 GB), DEM and land cover
$ uv run fraktal split --seed 0          # train and held-out test tiles, written as a manifest
$ uv run fraktal generate --seed 0       # skeleton world from train-tile footprints only
$ uv run fraktal synth --regions 3       # render synthetic chips, with sample PNGs
$ uv run fraktal benchmark --synthetic data/synthetic/chips   # IoU for the three arms
```

`render` renders any building layer onto a region. `benchmark` needs PyTorch (`make install-train`) and a GPU for the full recipe. `inspect <world> --id <id>` returns a feature's geometry and provenance as JSON. Exit codes: 0 ok, 2 usage error, 3 not implemented, 4 failure.

## Documentation

- [Plan](docs/PLAN.md): problem, scope, requirements, shape, and later bets
- [Slices](docs/SLICES.md): build order and acceptance tests
- [Decisions](docs/adr/): why it is built this way
- [Open questions](docs/QUESTIONS.md): the decision register

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Apache-2.0. See [LICENSE](LICENSE).
