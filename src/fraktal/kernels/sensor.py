"""Sensor effects shared by the 2.5D and 3D renderers: haze, tone, contrast, blur and noise.

Arrays and plain parameters in, arrays out (ADR-0005). The 2.5D renderer calls ``grade`` on its
float radiance before rounding; the 3D path calls ``sensor_chain`` on a finished 8-bit render, so
both look through the same lens. Randomness is the caller's: noise takes a generator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fraktal.kernels.filters import gaussian_blur

HAZE_RGB = (200.0, 205.0, 215.0)  # colour of scattered skylight that haze adds


def transmittance(optical_depth: float, sun_elevation_deg: float) -> float:
    """Share of surface light that survives the haze: falls as optical depth grows."""
    if optical_depth < 0:
        raise ValueError("optical depth must not be negative")
    return math.exp(-optical_depth / math.sin(math.radians(sun_elevation_deg)))


def apply_haze(radiance: Any, optical_depth: float, sun_elevation_deg: float) -> Any:
    """Blend (H, W, 3) radiance toward the haze colour: ``r * t + haze * (1 - t)``."""
    t = transmittance(optical_depth, sun_elevation_deg)
    return radiance * t + np.array(HAZE_RGB, dtype="float32") * (1.0 - t)


def grade(
    radiance: Any,
    haze: tuple[float, float] = (0.0, 50.0),
    tone: tuple[float, float, float] = (1.0, 1.0, 1.0),
    contrast: float = 1.0,
) -> Any:
    """Haze, per-channel tone and contrast on (H, W, 3) float radiance; no rounding yet.

    ``haze`` is (optical depth, sun elevation in degrees). Contrast stretches about each
    channel's own mean, so dark channels (blue) survive.
    """
    out = apply_haze(radiance, haze[0], haze[1]) if haze[0] > 0 else radiance
    graded = out * np.array(tone, dtype="float32")
    if contrast != 1.0:
        pivot = graded.reshape(-1, 3).mean(axis=0)
        graded = (graded - pivot) * contrast + pivot
    return graded


def apply_psf(rgb: Any, sigma_px: float) -> Any:
    """Gaussian point-spread blur of a (3, H, W) uint8 image, one band at a time."""
    if sigma_px <= 0:
        return rgb
    out = np.empty_like(rgb)
    for band in range(rgb.shape[0]):
        blurred = gaussian_blur(rgb[band].astype("float32"), sigma_px)
        out[band] = np.clip(np.rint(blurred), 0, 255).astype("uint8")
    return out


def add_gaussian_noise(tile: Any, sigma: float, rng: Any) -> Any:
    """Add zero-mean Gaussian noise (std ``sigma`` in digital numbers) to a (3, H, W) uint8 tile."""
    if sigma <= 0:
        return tile
    noise = rng.standard_normal(tile.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(np.rint(tile.astype("float32") + noise), 0, 255).astype("uint8")


def _radius(h: int, w: int) -> Any:
    """Distance from the centre, 1 at the corners, as an (H, W) array."""
    y, x = np.indices((h, w), dtype="float32")
    cy, cx = (h - 1) / 2, (w - 1) / 2
    return np.hypot(y - cy, x - cx) / max(float(np.hypot(cy, cx)), 1e-6)


def _scaled(channel: Any, factor: float) -> Any:
    """``channel`` resampled so the picture is ``factor`` times larger about the centre."""
    h, w = channel.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    y, x = np.indices((h, w), dtype="float32")
    sy, sx = np.clip(cy + (y - cy) / factor, 0, h - 1), np.clip(cx + (x - cx) / factor, 0, w - 1)
    y0, x0 = np.floor(sy).astype(int), np.floor(sx).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = sy - y0, sx - x0
    top = channel[y0, x0] * (1 - fx) + channel[y0, x1] * fx
    bottom = channel[y1, x0] * (1 - fx) + channel[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


_DCT = np.array(
    [[np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8) * np.cos((2 * x + 1) * u * np.pi / 16)
      for x in range(8)] for u in range(8)], dtype="float32"
)  # fmt: skip
_QUANT = np.array(  # the standard JPEG luminance table
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype="float32",
)  # fmt: skip


def _block_compress(rgb: Any, quality: int) -> Any:
    """JPEG's loss without the file: 8 x 8 DCT blocks of luma and chroma, quantised and undone.

    Plain numpy, so the result is the same on every machine. ``quality`` is 1 to 95 (JPEG's scale).
    """
    q = min(max(quality, 1), 95)
    scale = 5000 / q if q < 50 else 200 - 2 * q
    table = np.maximum(np.floor((_QUANT * scale + 50) / 100), 1.0)
    x = np.moveaxis(rgb.astype("float32"), 0, -1)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    ycc = np.stack(
        [0.299 * r + 0.587 * g + 0.114 * b,
         128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
         128 + 0.5 * r - 0.418688 * g - 0.081312 * b], axis=-1
    ) - 128  # fmt: skip
    h, w = ycc.shape[:2]
    ph, pw = -h % 8, -w % 8
    padded = np.pad(ycc, ((0, ph), (0, pw), (0, 0)), mode="edge")
    blocks = padded.reshape((h + ph) // 8, 8, (w + pw) // 8, 8, 3).transpose(0, 2, 4, 1, 3)
    coef = _DCT @ blocks @ _DCT.T
    coef = np.rint(coef / table) * table
    blocks = _DCT.T @ coef @ _DCT
    padded = blocks.transpose(0, 3, 1, 4, 2).reshape(h + ph, w + pw, 3)[:h, :w] + 128
    y, cb, cr = padded[..., 0], padded[..., 1] - 128, padded[..., 2] - 128
    back = np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], axis=-1)
    return np.moveaxis(np.clip(np.rint(back), 0, 255).astype("uint8"), -1, 0)


def lens_effects(rgb: Any, vignette: float = 0.0, ca_px: float = 0.0, jpeg_quality: int = 0) -> Any:
    """Drone-lens effects on a (3, H, W) uint8 image; each one is off at 0.

    ``ca_px`` is the red-blue fringe at the corner, in pixels (red is magnified, blue shrunk);
    ``vignette`` darkens the corners by that share; ``jpeg_quality`` (1 to 95) is block loss.
    """
    out = rgb
    if ca_px > 0:
        half_diag = float(np.hypot(*out.shape[1:])) / 2
        k = ca_px / max(half_diag, 1.0)
        planes = out.astype("float32")
        planes[0], planes[2] = _scaled(planes[0], 1 + k), _scaled(planes[2], 1 - k)
        out = np.clip(np.rint(planes), 0, 255).astype("uint8")
    if vignette > 0:
        falloff = 1.0 - vignette * _radius(*out.shape[1:]) ** 2
        out = np.clip(np.rint(out.astype("float32") * falloff), 0, 255).astype("uint8")
    if jpeg_quality > 0:
        out = _block_compress(out, int(jpeg_quality))
    return out


def sensor_chain(
    rgb: Any,
    *,
    haze: tuple[float, float] = (0.0, 50.0),
    tone: tuple[float, float, float] = (1.0, 1.0, 1.0),
    contrast: float = 1.0,
    psf_sigma_px: float = 0.0,
) -> Any:
    """Grade then blur a finished (3, H, W) uint8 render: the 3D path's sensor, noise aside."""
    radiance = np.moveaxis(rgb.astype("float32"), 0, -1)
    graded = np.clip(np.rint(grade(radiance, haze, tone, contrast)), 0, 255).astype("uint8")
    return apply_psf(np.moveaxis(graded, -1, 0), psf_sigma_px)


def expose(rgb: Any, ev: float, shoulder: float) -> Any:
    """Exposure in stops and a soft highlight shoulder on a (3, H, W) uint8 image.

    ``ev`` scales linear light by 2**ev. ``shoulder`` (0 to 1) rolls highlights off like a phone's
    tone curve: 0 is a hard clip, 1 is a full Reinhard curve. Both are neutral at 0.
    """
    if ev == 0.0 and shoulder <= 0.0:
        return rgb
    lin = (rgb.astype("float32") / 255.0) ** 2.2 * np.float32(2.0**ev)
    if shoulder > 0:
        lin = (1 - shoulder) * np.minimum(lin, 1.0) + shoulder * lin / (1.0 + lin)
        lin = lin / ((1 - shoulder) + shoulder / 2.0)  # white stays white
    return np.clip(np.rint(255.0 * np.clip(lin, 0, 1) ** (1 / 2.2)), 0, 255).astype("uint8")


def desaturate(rgb: Any, factor: float) -> Any:
    """Blend a (3, H, W) uint8 image toward its luma: 1 keeps colour, 0 is grey."""
    if factor >= 1.0:
        return rgb
    x = rgb.astype("float32")
    luma = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    return np.clip(np.rint(luma + (x - luma) * factor), 0, 255).astype("uint8")


def add_sensor_noise(rgb: Any, read_sigma: float, shot_gain: float, rng: Any) -> Any:
    """Read noise plus shot noise: std is ``sqrt(read**2 + shot_gain * value)`` in digital numbers.

    Brighter pixels are noisier in absolute terms and dark ones carry the read floor, as in a real
    sensor. A luma-shared component gives the grain a little colour correlation.
    """
    if read_sigma <= 0 and shot_gain <= 0:
        return rgb
    value = rgb.astype("float32")
    sigma = np.sqrt(read_sigma**2 + shot_gain * value)
    shape = value.shape
    grain = rng.standard_normal(shape, dtype=np.float32)
    shared = rng.standard_normal(shape[1:], dtype=np.float32)
    noise = (0.7 * grain + 0.7 * shared[None]) * sigma
    return np.clip(np.rint(value + noise), 0, 255).astype("uint8")


def bonnet(
    h: int, w: int, height_share: float, curve: float, tone: tuple[float, float, float]
) -> tuple[Any, Any]:
    """A dashcam's view of the car bonnet: ``(rgb (3, H, W) uint8, mask (H, W) bool)``.

    The bonnet fills the bottom ``height_share`` of the frame with a gently arched top edge
    (``curve`` is the arch height as a share of the bonnet) and a reflection of the sky that
    fades with distance. ``mask`` is True where the bonnet hides the scene.
    """
    rows, cols = np.indices((h, w), dtype="float32")
    top = h * (1.0 - height_share)
    x = (cols - (w - 1) / 2) / (w / 2)
    edge = top - curve * h * height_share * (1.0 - x**2)
    mask = rows >= edge
    depth = np.clip((rows - edge) / max(h * height_share, 1.0), 0, 1)
    shade = 0.25 + 0.55 * (1.0 - depth) ** 2  # brighter reflection near the far edge
    rgb = np.stack([shade * 255.0 * t for t in tone]).astype("float32")
    return np.clip(np.rint(rgb), 0, 255).astype("uint8"), np.asarray(mask)


__all__ = [
    "HAZE_RGB",
    "add_sensor_noise",
    "bonnet",
    "desaturate",
    "expose",
    "add_gaussian_noise",
    "apply_haze",
    "apply_psf",
    "grade",
    "lens_effects",
    "sensor_chain",
    "transmittance",
]
