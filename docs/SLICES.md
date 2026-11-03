# Fraktal: Slices

Vertical increments. Each ends in something you can demonstrate. Progress is tracked on the Pandan board (key FRK), one epic per slice. Slice 1 confronts the riskiest unknown: does synthetic imagery help a real model at all.

## V1: Benchmark spine

Thin end-to-end loop on a small region with a crude renderer, plus the three-arm benchmark. Real building footprints are used as the skeleton, so this slice isolates the renderer and the sim-to-real question.

**Delivers:** R0 (first measurement), R5, R1 (partial), R8 (partial)

**Build plan**

1. ~~Pick the benchmark dataset.~~ Done: SpaceNet 2 Khartoum, see ADR-0008.
2. Project setup with `uv`, CLI skeleton (`ingest`, `generate`, `render`, `benchmark`, `inspect`) with JSON output and exit codes, `kernels` package skeleton.
3. `ingest`: fetch SpaceNet 2 Khartoum, and the AOI's DEM (Copernicus GLO-30) and WorldCover; cache and clip. Building footprints come from the SpaceNet training-split labels, not OSM, so the benchmark labels and the world agree. OSM returns in V2.
4. World model v0: building and land-cover layers with provenance, hashed IDs, named RNG sub-streams.
5. Crude renderer: flat per-class reflectance on DSM, single sun direction, hard shadows. Label rasteriser on the same grid.
6. Benchmark harness: fixed U-Net and training recipe, three arms, report JSON.
7. Pretraining worlds exclude footprints from held-out test tiles, with a check that fails if one leaks.
8. Run: real-10%, real-100%, synthetic-pretrain then real-10%. Record the baseline and confirm or revise the Q14 bar.

**Demo:** run `fraktal benchmark`; see a report with IoU for all three arms, and open a rendered tile and its mask side by side in QGIS.

**Rests on assumptions:** Q14 (benchmark bar), revised on the first baseline. The dataset is decided (ADR-0008); if the licence review rejects it, switching costs a day or two.

### Test plan

#### End-to-end

- `fraktal ingest` then `generate` then `render` for a 2 km × 2 km region produces tiles and masks, exit code 0, JSON summary on stdout.
- `fraktal benchmark` writes a report with three IoU numbers; running it twice with the same seed gives IoUs within 1 point.
- Every rendered tile's building mask equals an independent rasterisation of the world's footprints (R5).

#### Integration

- Ingest for a cached region performs no network calls on a second run.
- Rendering the same world twice gives byte-identical masks.
- Missing DEM tile gives a hard error and nonzero exit code, not a flat terrain.

#### Unit

- Shadow kernel: a single 10 m tall box at 45° sun produces a shadow of the expected length (±1 pixel).
- Feature ID hash is independent of generation order.
- RNG sub-streams: adding a stream does not change existing streams' output.

## V2: Procedural world

> Sequencing: V3 runs first. V1 showed synthetic-only IoU of about 0 on real tiles (ADR-0006), so renderer realism is the bottleneck, not world content.

Replace the real skeleton with procedural generation, so a world can exist anywhere, from just a seed.

**Delivers:** R1, R2, R8

**Build plan**

1. Terrain detail and hydrology kernels (noise, flow accumulation) with golden tests.
2. Land cover generator conditioned on DEM and WorldCover where present.
3. Road network growth (arterials, then local streets), with road classes and widths.
4. Settlement and building placement: parcels, footprints, heights, roof types.
5. `generate --fallback procedural` for regions without real data; provenance records each layer's origin.
6. `inspect` can query a feature by ID and return its geometry and provenance as JSON.

**Demo:** generate the same region at 3 seeds and open the three worlds in QGIS; show one identical when regenerated.

**Rests on assumptions:** Q13 not yet exercised. ADR-0002's hybrid approach is the design, not an assumption.

### Test plan

#### End-to-end

- `fraktal generate --seed 42` twice produces identical GeoParquet hashes (R1).
- A region with no OSM data generates a plausible world with `provenance: procedural` on all buildings and roads.
- Re-run of V1 benchmark with procedural buildings in place of real ones produces a report (the number is recorded, not gated).

#### Integration

- Every building lies on a parcel and every parcel touches a road (connectivity check).
- Roads do not cross water without a bridge feature.
- Changing the `buildings` sub-stream config does not change the road network.

#### Unit

- Flow accumulation on a known tilted plane gives the expected drainage.
- Parcel subdivision never produces zero-area or overlapping parcels.

## V3: Sensor realism

Make the imagery look and behave like real satellite data, and measure whether it helps.

**Delivers:** R3, R0 (second measurement)

**Build plan**

1. Per-class reflectance in RGB and NIR with intra-class variation (roof materials, vegetation health).
2. Sun elevation/azimuth, soft shadow handling, atmospheric haze, optional cloud layer.
3. Seasonal variation (vegetation colour, paddy flooding).
4. Sensor model: PSF, noise, GSD resampling to 0.5 m and 3 m.
5. Randomisation config: ranges sampled per tile from named sub-streams.
6. Ablation runner: benchmark with each realism component on and off.

**Demo:** side-by-side of V1 and V3 renders of the same tile, plus an ablation table showing the benchmark IoU for each realism component.

**Rests on assumptions:** Q13 (randomisation before learned refinement). If the ablation shows no gain, the premise is in question and V4 should wait.

### Test plan

#### End-to-end

- Benchmark arm 3 (synthetic pretrain + real 10%) improves over V1 arm 3 by a recorded margin, and the ablation table is produced.
- Arm 3 beats arm 1 by at least 5 IoU points (Q14 bar), or the report flags the miss and the project pauses to revisit.

#### Integration

- Same world, different sensor config (0.5 m vs 3 m) keeps masks aligned with imagery.
- Cloud cover marks affected tile pixels in a quality mask, and masks are dropped from the label where a cloud hides a building.

#### Unit

- Haze model is monotonic in optical depth.
- NIR reflectance for healthy vegetation exceeds red by the configured ratio.
- Noise model's measured variance matches its parameter.

## V4: SE Asia biome pack

Make the worlds specifically tropical SE Asian, and measure regional transfer.

**Delivers:** R4

**Build plan**

1. Biome rules for five types: dense urban, informal settlement, plantation (oil palm), paddy, mangrove/coast.
2. Informal settlement generator (irregular parcels, tin and tile roofs, tight spacing).
3. Plantation and paddy patterns (regular rows and terraced fields) with seasonal states.
4. Find or assemble a small tropical real test set (candidates: Open Cities AI, local OSM-labelled areas); record licence.
5. Benchmark per biome and on the tropical test set.

**Demo:** a sheet of tiles from all five biomes, and a benchmark table by biome on the tropical test set.

**Rests on assumptions:** Q15, because a suitable tropical test set may not exist. If not, the fallback is a small hand-labelled set (days of effort).

### Test plan

#### End-to-end

- `fraktal generate --biome <name>` produces each of the five biome types and the packaged sheet renders without errors.
- Benchmark on the tropical test set reports IoU per biome and overall.

#### Integration

- Informal-settlement buildings average a footprint area and density in the configured range.
- Plantation rows are parallel within a tolerance and aligned to the field boundary.

#### Unit

- Row generator produces equal spacing along the field's bounding axis.
- Biome config schema rejects unknown keys.

## V5: Dataset product

Turn generation into something a customer can buy, and prove it scales.

**Delivers:** R6, R7, plus the licence gate

**Build plan**

1. Packager: tile, validate (drop and list bad tiles), STAC catalog, COCO and GeoTIFF masks, manifest, content-addressed output.
2. Dataset card generator: region, biomes, sensor, GSD, provenance and licences, benchmark results.
3. Licence review of OSM-derived content, WorldCover, DEM and the chosen benchmark datasets. Write the outcome into an ADR before anything is sold.
4. Scale-out: run the same CLI on a RunPod pod for a 100 km² batch; record cost and throughput.
5. Python API parity with the CLI.

**Demo:** `fraktal package` produces a catalog that opens in QGIS and loads into a PyTorch dataset in a ten-line script; show the dataset card and the RunPod run's cost per km².

**Rests on assumptions:** Q7 (licences). If OSM obligations make sale impractical, fall back to WorldCover/DEM-only skeletons with procedural buildings.

### Test plan

#### End-to-end

- Package a dataset, load the STAC catalog with a standard reader, and run a one-epoch training from the output.
- A 100 km² RunPod run finishes and the output is identical to a local run of the same seed for vector layers.
- The dataset card lists every layer's provenance and licence.

#### Integration

- A tile with a corrupted label is dropped and appears in the manifest's dropped list.
- Re-packaging unchanged inputs yields the same content hash.

#### Unit

- COCO exporter produces valid polygon annotations for multi-polygon buildings.
- Manifest schema validates and rejects missing generator version.
