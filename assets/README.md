# Assets

Materials, meshes and textures the 3D renderer may use. Only assets we own or that are CC0-1.0 go
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
