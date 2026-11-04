"""Real evaluation chips from Open Cities AI (FRK-43 follow-up). Evaluation only.

Open Cities AI is drone imagery of African cities with community-mapped building footprints,
licensed ODbL-1.0 (https://source.coop/open-cities/ai-challenge). Jian's decision, 2026-11-04:
private evaluation only. These chips are never training data, never shipped and never mixed into
a synthetic set; ``chips.json`` says so (``eval_only``) and has an empty train split, so the
benchmark's train path cannot pick them up by accident.

Scenes are cloud-optimised GeoTIFFs, so a window is read at the target pixel size straight from
the overviews, with area averaging, which is also the low-pass that turns 4 to 8 cm drone pixels
into something like a 0.5 m satellite pixel. Nothing larger than a chip is downloaded, except
the scene's label file (1 to 20 MB).
"""

from __future__ import annotations

import json
import tempfile
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import Window

from fraktal.ids import substream
from fraktal.log import emit

BASE_URL = "https://data.source.coop/open-cities/ai-challenge"
LICENCE = "ODbL-1.0"
CITATION = (
    'GFDRR Labs (2020). "Open Cities AI Challenge Dataset", Version 1.0, Radiant MLHub. '
    "https://doi.org/10.34911/rdnt.f94cxb"
)
SCHEMA = 1


class EvalSetError(ValueError):
    pass


@dataclass(frozen=True)
class Scene:
    city: str  # the folder under train_tier_1, e.g. "dar"
    id: str  # the scene id, e.g. "a017f9"

    @property
    def name(self) -> str:
        return f"{self.city}-{self.id}"


def parse_scene(text: str) -> Scene:
    city, _, scene = text.partition("/")
    if not city or not scene or "/" in scene:
        raise EvalSetError(f"scene must look like 'dar/a017f9', got {text!r}")
    return Scene(city, scene)


def _image_path(base: str, scene: Scene) -> str:
    path = f"{base}/train_tier_1/{scene.city}/{scene.id}/{scene.id}.tif"
    return f"/vsicurl/{path}" if base.startswith("http") else path


def _label_path(base: str, scene: Scene) -> str:
    return f"{base}/train_tier_1/{scene.city}/{scene.id}-labels/{scene.id}.geojson"


def _fetch_labels(base: str, scene: Scene, crs: Any, work: Path) -> gpd.GeoDataFrame:
    source = _label_path(base, scene)
    if base.startswith("http"):
        local = work / f"{scene.name}.geojson"
        with urllib.request.urlopen(source, timeout=120) as response:  # noqa: S310 - fixed https base
            local.write_bytes(response.read())
        source = str(local)
    frame = gpd.read_file(source)
    return frame.to_crs(crs) if frame.crs is not None else frame.set_crs(crs)


def scene_chips(
    scene: Scene,
    base: str,
    gsd_m: float,
    chip_px: int,
    max_chips: int,
    seed: int,
    work: Path,
    empty_share: float = 0.1,
) -> list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
    """Up to ``max_chips`` (id, RGB uint8 (3, N, N), mask) chips from one scene.

    Chips with more than 5% of their pixels outside the imagery (alpha 0) are skipped. Chips
    without a building are kept only up to ``empty_share`` of the result.
    """
    chips: list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []
    with rasterio.open(_image_path(base, scene)) as src:
        if src.crs is None or src.count < 3:
            raise EvalSetError(f"{scene.name}: need a georeferenced RGB(A) image")
        res = float(src.res[0])
        side = chip_px * gsd_m / res  # a chip's side in source pixels
        if side < chip_px:
            raise EvalSetError(f"{scene.name}: already coarser than {gsd_m} m ({res:.3f} m)")
        labels = _fetch_labels(base, scene, src.crs, work)
        geoms = [g for g in labels.geometry if g is not None and not g.is_empty]
        cols = int(src.width // side)
        rows = int(src.height // side)
        order = substream(seed, f"evalset:{scene.name}").permutation(rows * cols)
        empties = 0
        for k in order:
            if len(chips) >= max_chips:
                break
            r, c = divmod(int(k), cols)
            window = Window(c * side, r * side, side, side)
            data = src.read(
                window=window, out_shape=(src.count, chip_px, chip_px),
                resampling=Resampling.average, boundless=False,
            )  # fmt: skip
            if src.count >= 4 and float((data[3] == 0).mean()) > 0.05:
                continue
            transform = src.window_transform(window) @ Affine.scale(side / chip_px)
            shapes = [(g, 1) for g in geoms if g.intersects(_bounds_box(transform, chip_px))]
            mask = np.zeros((chip_px, chip_px), dtype=np.uint8)
            if shapes:
                mask = rasterize(shapes, (chip_px, chip_px), transform=transform, dtype="uint8")
            if mask.sum() == 0:
                if empties >= max(1, int(empty_share * max_chips)):
                    continue
                empties += 1
            chips.append((f"{scene.name}-r{r:03d}c{c:03d}", data[:3].astype(np.uint8), mask))
    return chips


def _bounds_box(transform: Affine, n: int) -> Any:
    from shapely.geometry import box

    x0, y0 = transform @ (0, n)
    x1, y1 = transform @ (n, 0)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def build_eval_set(
    scenes: Sequence[Scene],
    out_dir: Path,
    *,
    base: str = BASE_URL,
    gsd_m: float = 0.5,
    chip_px: int = 500,
    max_chips_per_scene: int = 20,
    seed: int = 0,
) -> Path:
    """Write an evaluation chip directory (``<id>_img.npy``, ``<id>_mask.npy``, ``chips.json``)."""
    if not scenes:
        raise EvalSetError("give at least one scene")
    out_dir.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    per_scene: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for scene in scenes:
            emit("evalset_scene", scene=scene.name)
            found = scene_chips(scene, base, gsd_m, chip_px, max_chips_per_scene, seed, Path(tmp))
            for chip_id, image, mask in found:
                np.save(out_dir / f"{chip_id}_img.npy", image)
                np.save(out_dir / f"{chip_id}_mask.npy", mask)
                ids.append(chip_id)
            per_scene[scene.name] = len(found)
    if not ids:
        raise EvalSetError("no usable chips: every window was outside the imagery or empty")
    meta = {
        "schema": SCHEMA,
        "kind": "evaluation",
        "eval_only": True,
        "use": "private evaluation only: never train on, redistribute or ship these chips",
        "source": "open-cities-ai-challenge",
        "licence": LICENCE,
        "citation": CITATION,
        "gsd_m": gsd_m,
        "chip_px": chip_px,
        "seed": seed,
        "scenes": per_scene,
        "train": [],  # empty on purpose
        "test": sorted(ids),
        "tiles": sorted(ids),
    }
    (out_dir / "chips.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "EVAL_ONLY.txt").write_text(meta["use"] + f"\nLicence: {LICENCE}\n{CITATION}\n")
    return out_dir


__all__ = [
    "BASE_URL",
    "EvalSetError",
    "Scene",
    "build_eval_set",
    "parse_scene",
    "scene_chips",
]
