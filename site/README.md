# Fraktal landing page

A single static page for fraktal.abangai.dev: `index.html` (inline CSS and JS, no external fonts or
trackers), `img/` and `media/`. It imports nothing from `src/`, so it can move to its own repo with
`git filter-repo --subdirectory-filter site`.

- **Logo:** `favicon.svg` is `design/brand/svg/fraktal-icon.svg` from the abang-landing-page repo; the header mark is the same icon inline.
- **Deploy:** `.github/workflows/site.yml` publishes `site/` to GitHub Pages on a push to `main` that
  touches `site/`. Setup: repo Settings → Pages → source "GitHub Actions", then replace the Cloudflare Worker
  that serves the placeholder (`deploy/cloudflare/fraktal-placeholder-worker.js` in abang-landing-page, route
  `fraktal.abangai.dev/*`) with a DNS CNAME to `leejianrong.github.io`, or have the Worker proxy to Pages.
- **Preview:** `python3 -m http.server -d site 8000`.
- **Assets:** `img/sample-rgb.jpg` and `img/sample-class.png` come from the shipped sample tile
  (`data/dist/sharing/sample/`); `media/fraktal-sample.zip` is that tile with a cleaned `sample.json`
  (no local paths). Rebuild them when the sample changes; keep `site/` under a few MB.
- **Never** put real street photos here (ADR-0006, evaluation only).
- Decision record: [ADR-0014](../docs/adr/0014-launch-phase-public-and-loud.md).
