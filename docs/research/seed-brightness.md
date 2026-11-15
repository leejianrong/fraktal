# Seed-to-seed brightness (FRK-171)

Measured 2026-11-16 on commit 5ae1077, CPU Blender, 12 samples, eye-level strips (`city_strips.py --seed N`), 8 seeds per
city, variety on. Brightness is the harness number (mean of the brightest channel). Photo band: Manila-Quezon 0.486 ± 0.09;
Nairobi centre 0.487, and the same ± 0.09 is used as its band (its own deviation was not re-measured here).

## Cause

Two things, about equally. With the variety off and one factor swapped at a time (Manila-Quezon, `seed_brightness.py relight`):

| Swap | Brightness |
|---|---|
| layout of seed 13, light of 13 / 2 / 8 (fill 1.47 / 1.83 / 1.86) | 0.291 / 0.342 / 0.357 |
| layout of seed 8, light of 13 / 8 | 0.353 / 0.435 |
| light of seed 13, layout of 13 / 8 / 39 (sky share 0.36 / 0.48 / 0.55) | 0.291 / 0.353 / 0.396 |

- **Light:** the sky fill (1.3 to 1.9) moves brightness by 0.07 to 0.08 between its ends. The sun strength, dust and the
  FRK-168 variety are small next to it (variety moved seeds 13 and 61 by +0.015).
- **Layout and camera:** how much sky the strip sees (sky share 0.34 to 0.55) moves it by about 0.10. This is the route and the
  trees and buildings that frame it, not a bug.

## Fix

`metering.apply` (`render3d`, eye-level strips, `--no-meter` to skip): one gamma for the whole strip pulls the mean `METER_STRENGTH`
(0.7) of the way to `METER_TARGET` (0.46), as a camera's auto-exposure would. The fill range, the sun offset and the clouds are
untouched. Labels and depth are not changed.

| | before | after |
|---|---|---|
| Manila-Quezon mean, std, range | 0.372, 0.053, 0.311 to 0.460 | 0.433, 0.016, 0.415 to 0.460 |
| Manila-Quezon seeds inside 0.396 to 0.576 | 3 of 8 | 8 of 8 |
| Nairobi centre mean, std, range | 0.430, 0.043, 0.374 to 0.488 | 0.451, 0.013, 0.434 to 0.468 |
| Nairobi centre seeds inside the band | 5 of 8 | 8 of 8 |

Contrast is not changed by it (std across seeds 0.030 to 0.026 for Manila, 0.024 to 0.025 for Nairobi).

## What it costs, and what is left

- Brightness variety across seeds drops from a std of about 0.05 to about 0.015. The photos spread 0.09 on their own, so the
  synthetic set is now narrower than the photos on this axis. The fill range was not narrowed, so the spread can be widened again
  by lowering `METER_STRENGTH` (0 turns it off). Light colour, shadow length and cloud shadows still vary per seed.
- Manila-Quezon's mean is still 0.05 under the photos' (0.433 against 0.486). A target of 0.46 at strength 0.7 does not close it,
  and a per-city target would be fitting to photos. The pilot cycle (FRK-149) decides whether the light should come up instead.
- The metering is a gamma on display values, so it does not recover highlights that are already clipped, and it adds nothing to
  contrast. Khartoum (0.627 against 0.540) has not been run with it yet.
