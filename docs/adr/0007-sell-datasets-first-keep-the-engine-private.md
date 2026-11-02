# ADR-0007: Sell datasets and custom data projects first; the engine stays private

- Status: Accepted
- Date: 2026-11-02
- Deciders: Jian

## Context

Comparable companies (Another Earth: data-as-a-service plus project-based custom data; Bifrost AI: enterprise subscription to a platform) show two models. Customers in this market buy labelled training data to solve a bottleneck, and are not generally looking to run a world generator. Selling the engine as a product requires an API, documentation and support, which is heavy before the data is proven. Data licensing also constrains what can be sold: OSM (ODbL, share-alike on derived databases), ESA WorldCover (CC-BY 4.0), Copernicus DEM (attribution), benchmark datasets (various, some non-commercial).

## Decision

Revenue model for the MVP: (a) catalog datasets (immutable, versioned, with dataset cards) and (b) custom datasets built to a customer spec (region, sensor, GSD, label classes). The generator and Python API are not public. Every dataset ships with a provenance manifest listing real and procedural layers and their licences. Before any dataset is sold, a licence review covers OSM-derived content and whether rendered outputs are produced works or derivative databases.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Platform/API subscription first | Support burden, needs proven data quality first |
| Open-source engine, sell services | Premature; the engine is the moat and not yet stable |

## Consequences

Fastest path to revenue with the least product surface. The cost is that revenue scales with services and data volume instead of software margins. Keeping the engine private is reversible later; open-sourcing it is the one-way direction, so private is the cheaper default. The OSM licence question is a hard gate before first sale.
