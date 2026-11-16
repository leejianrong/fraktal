# Fraktal landing page

A single static page for fraktal.abangai.dev: `index.html` (inline CSS and JS, no external fonts or
trackers), `img/` and `media/`. It imports nothing from `src/`, so it can move to its own repo with
`git filter-repo --subdirectory-filter site`.

- **Deploy:** `.github/workflows/site.yml` publishes `site/` to GitHub Pages on a push to `main` that
  touches `site/`. Setup: repo Settings → Pages → source "GitHub Actions", then point the DNS CNAME for
  `fraktal.abangai.dev` at `leejianrong.github.io` (the same as the Abang page).
- **Preview:** `python3 -m http.server -d site 8000`.
- **Assets:** `img/sample-rgb.jpg` and `img/sample-class.png` come from the shipped sample tile
  (`data/dist/sharing/sample/`); `media/fraktal-sample.zip` is that tile with a cleaned `sample.json`
  (no local paths). Rebuild them when the sample changes; keep `site/` under a few MB.
- **Never** put real street photos here (ADR-0006, evaluation only).
- Decision record: [ADR-0014](../docs/adr/0014-launch-phase-public-and-loud.md).
