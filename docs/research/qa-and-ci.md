# QA and CI: research spike and what to apply

A spike, not a decision. Evidence is mostly docs and blog posts; nothing below was measured on our
suite except where stated. Re-check each item against a real CI run before adopting it.

## What the repo does today

- `make check` is ruff, `mypy --strict` and `pytest -m "not integration"`. `.githooks/pre-push`
  runs it, so the local gate and the CI `unit` job run the same 644 tests.
- CI (`.github/workflows/ci.yml`) has six parallel jobs (lint, typecheck, unit, integration,
  audit, secrets), cancel-in-progress, read-only permissions, pinned actions and a uv cache.
- Missing: `timeout-minutes`, a durations report, test parallelism, path filters, a `slow`
  marker, property tests, and Dependabot auto-merge. Six of the nine numba kernels use
  `cache=True`.
- Cost: the `unit` job takes about 8.5 min on GitHub; locally 15 min to 2 h 25 min on the 8 GB
  machine, where swap and other sessions dominate. In a three-file sample the slowest test was
  7 s (`test_markings` city export) and most were under 0.5 s, so the time is probably a few
  heavy end-to-end tests, numba compile on a cold cache and memory pressure. Confirm with
  `--durations=30` on one CI run before tuning.

## Recommendations, by payoff against effort

1. **Make the pre-push hook cheap.** Point it at ruff and mypy only; CI is the test gate (matches
   AGENTS.md, PR #163). About 10 minutes.
2. **Report slow tests.** Add `--durations=30` to the unit step. One line; it tells us where to aim.
3. **A `slow` marker.** Tag the slowest few tests, add `make test-fast`. After step 2.
4. **`timeout-minutes: 20` on every job**, and one `ci-ok` job that needs all the others, as the
   only required check. A hung runner then costs 20 minutes, not six hours.
5. **Numba cache in CI.** `cache=True` on the three kernels without it, plus `actions/cache` keyed on
   `hashFiles('src/fraktal/kernels/**')`. No primary source found; measure the unit time before and after.
6. **pytest-xdist in CI only** (`-n auto --dist loadfile`). Expect 2x to 3x on four vCPUs, unverified;
   check for shared temp files first. Not locally: 8 GB.
7. **Dependabot auto-merge for patch and minor bumps** once CI is trustworthy (`gh pr merge --auto
   --rebase`, gated on the actor). Needs the repo auto-merge setting. Ruleset behaviour unconfirmed.
8. **Hypothesis for a few invariants** in kernels and geometry: same seed gives the same output,
   polygons stay in bounds, no NaNs. `deadline=None` because of numba compile. One to two hours.

## Not adopting

- Test-impact selection (testmon): a false negative would hide a break in the determinism contract.
- Merge queue: sole author, no concurrent PR conflicts.
- A coverage gate: conflicts with "benchmarks steer, they do not gate"; run `--cov` for information at most.
- Image-diff tests on Blender output: drivers differ across machines and realism is judged by eye. Keep
  golden tests on kernel arrays.
- Automatic flaky-test retries: determinism is a contract, a flaky test is a bug.
- Larger paid runners before caching and sharding.
- Blender and network integration on every PR: gate by path or run nightly.

## Open points

- `requires-python` is `>=3.10` in `pyproject.toml` while the brief says 3.11+. Decide and align.
- Path filters on `paths-ignore: docs/**` stop required checks from reporting; use a paths-filter
  step inside the jobs instead.

## Sources

pytest-xdist distribution modes: https://pytest-xdist.readthedocs.io/en/stable/distribution.html ·
GitHub, automating Dependabot with Actions:
https://docs.github.com/en/code-security/tutorials/secure-your-dependencies/automate-dependabot-with-actions ·
setup-uv caching: https://github.com/astral-sh/setup-uv · pytest-split matrix example:
https://github.com/jerry-git/pytest-split-gh-actions-demo
