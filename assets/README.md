# Assets

Materials, meshes and textures the 3D renderer may use. Only assets we own, that are CC0-1.0, or that a model with permissive output terms generated for us (`generated-permissive`) go
here (ADR-0010 decision 5). `manifest.json` lists every file; `fraktal assets check` (and CI) fails
on a file without a row, a licence that is not `owned` or `CC0-1.0`, a row not marked `verified`, a
missing file, or a hash that no longer matches.

## Add an asset

1. Read the source's own licence page and confirm CC0 (or that you made it). Marketplace and
   scraped assets, CC-BY and CC-NC are out.
2. Put the file under `assets/<kind>/`.
3. Add a row with `id`, `file` (relative to `assets/`), `kind`, `source`, `licence`, `author`, `url`
   (the page you checked), `sha256` (`sha256sum file`) and `"verified": true` once you have read the page.
4. Run `fraktal assets check`.

## Generated textures

A texture made by a model (FRK-73 and FRK-76: SDXL base 1.0) uses licence `generated-permissive` and a
`generator` object with `model`, `version`, `licence` (the model's output terms, read from its licence
file), `prompt`, `seed` and `date`; `fraktal assets` fails without them. Say in the row's `source` how it
was made. Generated images are probably not copyrightable (docs/research/texture-model-spike.md), so we
do not claim to own them; we can ship them, and so can anyone else.

## Meshes

A row with `kind: mesh` always needs a `generator` object, whether a model or code made it (FRK-125,
ADR-0013 decision 5): `model`, `version`, `seed`, `date` and `components`, a list of every part of the
pipeline (`name`, `version`, `licence` each), for example the image model, the background remover and the 3D
model, or `fraktal.propgen` for a code-written prop. A mesh from a model also takes licence
`generated-permissive` and the texture generator keys above. `fraktal assets` (and CI) fails on a mesh
without it, and on any component the ADR excludes (TripoSG, TRELLIS, DINOv3, RMBG, Hunyuan3D, Stable Fast
3D, Cube 3D, FLUX.1 dev, hosted services such as Meshy or Rodin). The list is `EXCLUDED_COMPONENTS` in
`src/fraktal/assets.py`; change it when the ADR changes.
