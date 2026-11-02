# ADR-0005: Start in Python, and isolate hot kernels behind an array-in/array-out seam so a Rust or C++ core can replace them later

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian (chose Python over the recommended Rust core)

## Context

The recommendation was a Rust core with Python glue, mainly for the later agent-simulation milestone (100k+ agents) and for determinism and speed. Jian preferred starting in Python and swapping a faster core in later. That is a sound call for the MVP: the biggest unknown is whether the data helps downstream models (ADR-0006), so iteration speed and the ML ecosystem matter more than raw throughput. The host has Python 3.10, an RTX 3080 Ti (16 GB), 20 cores and 31 GB RAM.

## Decision

- The whole MVP is Python. Stack: NumPy, numba for CPU kernels, PyTorch for GPU kernels and the benchmark model, rasterio/GDAL/pyproj/shapely/geopandas for geospatial IO. Project managed with `uv`.
- Anything computationally heavy (noise, flow accumulation, road network growth, shadow ray-marching, rasterisation, later the agent loop) lives in a `kernels` package. A kernel takes NumPy arrays and plain parameters and returns NumPy arrays. It holds no file paths, no global state, no objects from the rest of the system.
- Pipeline stages communicate through files (GeoParquet, Cloud-Optimized GeoTIFF), not in-process objects, so a stage can be reimplemented in any language.
- Each kernel has golden-output tests (fixed seed, stored hash or tolerance), so a Rust or C++ replacement via PyO3 can be checked against the Python one.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Rust core from day one (the recommendation) | Slower iteration on the riskiest question, higher up-front cost, ML code is Python anyway |
| Go | Weaker numerics and GPU story |
| Python with no seam | A later rewrite would have to untangle the whole codebase |

## Consequences

Fastest route to a measured result. The risk is that Python hits a ceiling at agent-simulation scale and in the Python-to-native swap, which is why the seam and golden tests are paid for now. Pure-Python loops in kernels are forbidden: if it cannot be vectorised or numba-compiled, it is a candidate for early native rewrite. Revisit when a kernel dominates runtime or when the agent simulation milestone starts.
