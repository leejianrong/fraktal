# Fraktal

Procedural world generation for synthetic satellite and geospatial training data.

Fraktal builds deterministic, labelled worlds (terrain, land cover, roads, buildings), conditioned on real geography and filled in procedurally, then renders them as satellite imagery with exact labels. The same world model is meant to feed later consumers: city agent simulation, drone and robotics views, and SAR.

**Status:** planning done, implementation starting. Nothing generates data yet.

- [Plan](PLAN.md): problem, scope, requirements and shape
- [Slices](SLICES.md): build order and acceptance tests
- [Decisions](docs/adr/): why it is built this way

## Develop

Requires [uv](https://docs.astral.sh/uv/).

```bash
make install   # locked dependencies
make hooks     # pre-push gate
make check     # lint, type-check, fast tests
make           # list all targets
```

## Licence

Apache-2.0. See [LICENSE](LICENSE).
