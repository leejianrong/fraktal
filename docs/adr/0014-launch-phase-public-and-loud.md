# ADR-0014: Launch phase: public and loud first, closed engine later

- Status: Accepted (2026-11-17, Jian); amends ADR-0007 and ADR-0011 for the launch phase
- Date: 2026-11-17
- Deciders: Jian

## Context

ADR-0007 said to sell catalog and custom datasets and keep the engine private, and ADR-0011 kept that
"until a studio exists". The repository is in fact public under Apache-2.0, there is no benchmark result,
no customer and no audience yet, and the landing page was a placeholder. A closed engine protects a
product that people already want. At this stage the scarce thing is attention: nobody pays for data from
an unknown project with no evidence, and ML engineers trust what they can read and run.

## Decision

For the launch phase the project is public and as loud as it can honestly be:

1. **The repository stays public** (Apache-2.0) and the landing page links to it.
2. **The primary call to action is a free sample**: the shipped tile as GeoTIFFs plus a manifest,
   downloadable without sign-up. Custom requests go to an email address, not a form or a price list.
3. **The page is honest about evidence.** It says the benchmark question (ADR-0006, ADR-0009) is open,
   lists what is unfinished, and promises to publish the first benchmark including a negative one. No
   invented customers, testimonials or numbers.
4. **Hard limits that do not change:** real street photos never appear on a public page or in a public
   artefact (ADR-0006, evaluation only); every asset is owned or CC0 (ADR-0010, `fraktal assets`);
   secrets stay out of the repo.
5. **The landing page lives in `site/`** in this repository, self-contained, so it can move to its own
   repo later with its history.

ADR-0007's closed-engine and sell-first model is not rejected. It is the intended shape once the product
is bigger and better, and this ADR is the place to revisit it. Revisit when there is a benchmark result,
a first paying request, or a feature worth protecting.

## Alternatives considered

| Option | Why not now |
|--------|-------------|
| Keep the engine private and sell only datasets (ADR-0007) | Needs an audience and evidence first; a private engine gives nothing to look at |
| Gate the sample behind an email | Adds a backend and friction before anyone cares; downloads are the signal we want |
| Say nothing until the benchmark is done | The benchmark is a milestone, not a launch date; the work is already worth looking at |

## Consequences

- Anyone can read, run and fork the engine. Making it private later is possible for new work but cannot
  recall what has been published, so the code in this repo should be treated as already public.
- Selling datasets still needs the licence review in ADR-0007 (OSM-derived content, produced works or
  derivative databases) before the first sale. Giving away a sample with attribution is a lighter case.
- The page makes promises (publish the benchmark) that the project has to keep.
