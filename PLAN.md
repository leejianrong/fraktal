# Fraktal: Plan

Status: draft (forks closed, awaiting review) · Milestone: MVP

## Problem

Teams training Earth-observation models are short of labelled satellite imagery. Real imagery is expensive, especially for remote regions and for sensors with restricted access, and labelling it by hand is slow and error-prone. Tropical SE Asia is poorly covered by common benchmarks even though it has dense cities, informal settlements, plantations and constant cloud cover.

Procedural generation can produce unlimited scenes with perfect labels, but only if models trained on them actually transfer to real imagery. That sim-to-real gap is the risk the whole project has to answer first.

## Solution

A tool that generates deterministic, labelled synthetic satellite datasets. You pick a region and a seed, plus sensor, resolution and label classes. You get back tiles of imagery with exact masks, a catalog and a manifest that says what was real and what was procedural. The same world can be re-rendered at another resolution or sensor without changing the scene.

Under the surface is a shared world model (terrain, water, land cover, roads, buildings, land use, stable IDs). Imagery is the first consumer. City agent simulation, drone and robotics views, SAR and disaster scenarios are later consumers of the same world (see "Later bets").

## Users and actors

1. **ML engineers at EO/geospatial companies and researchers** who need labelled imagery for training. They consume datasets and read dataset cards.
2. **Us (operator):** runs generation, benchmarks and packaging through the CLI.
3. **Coding agents** that drive the CLI. They need JSON output and meaningful exit codes. If a human's and an agent's request conflict, the CLI contract wins, because both use the same entry points.

## Scope

**In this milestone**
- Optical imagery, RGB plus NIR, at 0.5 m and 3 m GSD, near-nadir
- Classes: buildings (footprints), land cover, roads
- Real-conditioned worlds (Copernicus DEM, ESA WorldCover, OSM) with procedural fill, per ADR-0002
- A tropical SE Asia biome pack: dense urban, informal settlement, plantation, paddy, mangrove
- Sensor realism: sun and shadows, haze, cloud, season, noise, PSF
- Deterministic generation, stable feature IDs, immutable content-addressed datasets
- A downstream benchmark (building segmentation) with a three-arm comparison
- Dataset packaging: STAC catalog, COCO and GeoTIFF masks, manifest, dataset card
- CLI and Python library

**Out (and why)**
- Agent/city simulation: needs the world model first, no clear first customer (Q12)
- SAR, drone/robotics views, oblique imagery: need different renderers (see Later bets)
- Custom UI: QGIS or MapLibre reads our files, and a viewer is not what answers the risk
- Public API or hosted platform: ADR-0007
- Learned/diffusion refinement: only if the benchmark demands it (ADR-0004)
- Individual-level population data of any kind (Q10)

## Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| R0 | Synthetic imagery + labels that measurably improve a real downstream model (ADR-0006) | Core goal |
| R1 | Same `(seed, config, version)` gives identical vector output; feature IDs are stable | Must-have |
| R2 | Worlds conditioned on real geography with provenance recorded per layer | Must-have |
| R3 | Sensor realism configurable: GSD, bands, sun angle, season, haze, cloud, noise | Must-have |
| R4 | SE Asia biome pack covers the five settlement/land types above | Must-have |
| R5 | Exact labels: every mask pixel agrees with the world the image was rendered from | Must-have |
| R6 | Datasets ship as STAC + COCO/GeoTIFF with manifest and dataset card | Must-have |
| R7 | Generation scales out to rented GPUs with the same CLI | Nice-to-have |
| R8 | World model schema carries roads, land use and per-building attributes now, for later consumers | Must-have |

## Shape

| Part | Mechanism | ADR |
|------|-----------|-----|
| S1 | **World model:** layered vector+raster store keyed by `(seed, config, version)`; sub-streamed RNG; hashed feature IDs; per-layer provenance | 0003 |
| S2 | **Real-data ingest:** fetch and cache Copernicus DEM tiles, WorldCover, OSM extracts, clip to region, write to local store | 0002 |
| S3 | **Procedural generators:** terrain detail, hydrology, land cover, road network, settlement/building placement, vegetation, parameterised by biome pack | 0002 |
| S4 | **Renderer:** world to DSM, reflectance per class and band, sun + shadows, haze/cloud, sensor model, resample to GSD | 0004 |
| S5 | **Label rasteriser:** same grid as the render; building, land cover and road masks from the world | 0004 |
| S6 | **Dataset packager:** tile, validate, write STAC/COCO/GeoTIFF + manifest + dataset card, content-address | 0003 |
| S7 | **Benchmark harness:** fixed U-Net and recipe; three arms; report JSON | 0006 |
| S8 | **Kernels package:** noise, hydrology, shadows, rasterisation, behind array-in/array-out interface with golden tests | 0005 |
| S9 | **CLI + Python API:** `ingest`, `generate`, `render`, `package`, `benchmark`; JSON out, exit codes | 0005 |

## Affordances

**Non-UI**

| Affordance | Kind | Wires to |
|------------|------|----------|
| `fraktal ingest --region` | CLI command | S2, writes real layers |
| `fraktal generate --seed --config` | CLI command | S1, S3, writes world |
| `fraktal render --world --sensor` | CLI command | S4, S5, writes tiles + masks |
| `fraktal package --render` | CLI command | S6, writes dataset |
| `fraktal benchmark --dataset` | CLI command | S7, writes report |
| `fraktal inspect <id>` | CLI command | reads world, dataset, or report as JSON |
| Config files (YAML, schema-checked) | config | all commands |
| Manifest + dataset card | generated files | S6 |

**UI.** None. QGIS and MapLibre read the output.

## Implementation decisions

- **Python first (ADR-0005).** NumPy, numba, PyTorch, rasterio/GDAL/pyproj/shapely/geopandas, managed by `uv`. Heavy loops live only in `kernels`, take arrays and return arrays, and have golden tests.
- **Stage contracts are files.** Stages read and write GeoParquet and Cloud-Optimized GeoTIFF, so no stage depends on another's in-memory objects. Any stage can later be reimplemented in Rust or C++.
- **World schema.** Features are `building`, `road`, `water`, `landcover_polygon`, `parcel`, each with `id`, `kind`, `geometry`, `provenance` (`real:osm`, `real:worldcover`, `procedural:<generator>`), and kind-specific attributes (building height, roof type, land use; road class, width). Terrain is a raster layer. Attributes needed by later simulation (use, capacity) are included now as optional fields.
- **Determinism (ADR-0003).** Named RNG sub-streams. Vector output is byte-identical for identical inputs. Rendered rasters are reproducible per environment but not promised bit-exact across hardware.
- **Failure behaviour.** Missing real data for a region is a hard error unless the config sets `fallback: procedural`, in which case the layer is marked `procedural` in provenance. A tile failing validation (label/image mismatch, NaN, all-empty) is dropped and listed in the manifest. Nothing is silently shipped.
- **Concurrency.** Single-writer batch jobs. Outputs are immutable and content-addressed, so concurrent runs cannot overwrite each other.
- **Runtime.** Local Linux/WSL2 on the RTX 3080 Ti for development. The same CLI runs on a RunPod pod for scale (R7); data moves in and out as tarballs/object storage.
- **Security and privacy.** Nothing sensitive in the MVP. Download credentials only via environment variables. Later agent simulation must use synthetic populations built from aggregate statistics, not individual-level data.
- **Versioning.** Every dataset manifest holds generator version and config hash. Datasets are immutable. A change that alters world output bumps the generator's major version.

## Testing approach

Test at the fewest, highest seams. Behaviour, not internals.

- **Determinism contract:** generate twice, compare hashes. This one test guards the whole world model.
- **Label/pixel agreement:** a render's masks must match an independently computed rasterisation of the world.
- **Benchmark as regression:** the three-arm IoU comparison is the product's real acceptance test and runs on every significant renderer change.
- **Kernel golden tests** (ADR-0005): fixed seed, stored output hash or tolerance, so a native replacement can be proven equivalent.
- Per-slice test plans are in SLICES.md.

## Assumed defaults

| ID | Assumed | Cost if wrong |
|----|---------|---------------|
| Q1 | Primary user is an EO ML team | Wrong customer shapes the dataset card and packaging; a day or two |
| Q3 | GeoParquet, COG, STAC, COCO | Switching formats is days; low |
| Q7 | Dependency and licence picks, Blender only as optional subprocess | A licence surprise could block selling; days to weeks |
| Q9 | Optical RGB+NIR only, 0.5 m and 3 m | Different bands/GSD means renderer changes; days |
| Q13 | Domain randomisation before learned refinement | If insufficient, adds a second system (weeks) |
| Q14 | Benchmark bar: +5 IoU over real-10%, at least 90% of real-100% | Bar is revised after the V1 baseline; low |
| Q15 | Benchmark data chosen in V1 step 1 | May not be tropical; regional transfer unproven until V4 |

## Open risks

- **Sim-to-real gap is larger than randomisation can close.** Revealed by V1 (crude render) and V3 (realism). If V3 does not move the benchmark, the whole premise needs rethinking before V4.
- **Python performance ceiling.** Revealed in V5 (scale-out) and the first agent simulation. Mitigated by the kernel seam (ADR-0005).
- **OSM/ODbL and benchmark-dataset licences** block a first sale. Resolved by the gate in V5.
- **Benchmark region mismatch:** open benchmarks may be non-tropical, so transfer to SE Asia is unmeasured until a tropical real test set is found (V4).
- **Floating-point determinism across machines and after a native swap.** Limited by ADR-0003 to vector output.

## Later bets

Recorded now because the user wants to explore them after the MVP. All reuse the world model (S1), so R8 keeps the schema ready.

1. **Robotics/drone views (high interest).** Render the same world from ground and low-altitude viewpoints. Needs a real 3D renderer and building geometry beyond footprints and heights (roofs, facades, props). Likely path: export the world to a 3D format and use Blender (optional subprocess) or a GPU renderer. This is the Bifrost-adjacent market. Needs an ADR when started.
2. **SAR simulation (high interest).** Real SAR training data is scarce and hard to label. Start with a physics-based simulation on the DSM: backscatter by class and incidence angle, layover, shadow, speckle noise. Another reuse of the DSM from ADR-0004. Needs an ADR when started.
3. **Disaster before/after pairs.** Synthetic flood, fire or earthquake scenes with change-detection labels. Mostly a world-state diff plus a second render of the same scene. Cheap once V3 exists.
4. **Agent/city simulation.** Hundreds of thousands of agents on the world's road and building graph (activity-based, mesoscopic), usable for modelling and for synthetic mobility and population datasets. Likely the point where a Rust or C++ kernel replaces Python (ADR-0005).
5. **Digital-twin scenarios.** Agent simulation plus imagery for "what if we build X here" planning. Combines 3 and 4.
6. **Learned refinement and diffusion** on top of renderer output, if the benchmark requires it.
