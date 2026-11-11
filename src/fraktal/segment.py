"""Class shares of street images from a pretrained segmenter, for evaluation only (FRK-145).

The model is OneFormer with the COCO Swin-L weights (MIT; the licence read is in
docs/research/eval-segmentation.md). It runs on the reference photos and on our frames alike, so
its own errors affect both sides. Results are internal numbers: nothing here is trained on,
conditioned on or shipped (ADR-0012).

``transformers`` is an optional dependency (``make install-seg``) and is imported only when the
model is first used. Everything else here takes any ``Segmenter``, so it is tested without it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray[Any, Any]
Segmenter = Callable[[Array], Array]  # (H, W, 3) uint8 -> (H, W) of class names, as a '<U..' array

MODEL = "shi-labs/oneformer_coco_swin_large"
GROUPS: dict[str, tuple[str, ...]] = {
    "building": (
        "building-other-merged", "house", "roof", "wall-brick", "wall-other-merged", "wall-stone",
        "wall-tile", "wall-wood", "window-other", "window-blind", "door-stuff", "tent",
    ),
    "road": ("road", "railroad", "gravel", "dirt-merged", "sand"),
    "pavement": ("pavement-merged", "platform", "floor-other-merged", "stairs"),
    "sky": ("sky-other-merged",),
    "vegetation": ("tree-merged", "grass-merged", "potted plant", "flower"),
    "vehicle": ("car", "truck", "bus", "motorcycle", "bicycle", "train"),
    "person": ("person",),
    "fence": ("fence-merged", "net"),
    "pole_sign": ("traffic light", "stop sign", "light", "banner"),
}  # fmt: skip
OTHER = "other"
NAMES = (*GROUPS, OTHER)
_GROUP_OF = {name: group for group, names in GROUPS.items() for name in names}


def shares(label_names: Array) -> dict[str, float]:
    """The share of the image in each of ``NAMES``, from a map of COCO class names."""
    out = dict.fromkeys(NAMES, 0.0)
    values, counts = np.unique(label_names, return_counts=True)
    for value, count in zip(values, counts, strict=True):
        out[_GROUP_OF.get(str(value), OTHER)] += float(count) / label_names.size
    return {k: round(v, 5) for k, v in out.items()}


def load_model() -> Segmenter:
    """The OneFormer segmenter. Needs ``transformers`` and ``torch`` (``make install-seg``)."""
    try:
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
    except ImportError as err:
        raise ValueError("segmentation needs `make install-seg` (transformers, torch)") from err
    processor: Any = OneFormerProcessor.from_pretrained(MODEL)
    model: Any = OneFormerForUniversalSegmentation.from_pretrained(MODEL)
    model.eval()
    names = np.array([model.config.id2label[i] for i in range(len(model.config.id2label))])

    def run(image: Array) -> Array:
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        ids = processor.post_process_semantic_segmentation(out, target_sizes=[image.shape[:2]])[0]
        labelled: Array = names[ids.numpy()]
        return labelled

    return run


def _key(image: Array) -> str:
    return hashlib.sha256(image.tobytes() + str(image.shape).encode()).hexdigest()[:20]


def cached_shares(
    images: Sequence[Array], make_segmenter: Callable[[], Segmenter], cache: Path
) -> list[dict[str, float]]:
    """``shares`` of each image, kept in the JSON file ``cache`` (keyed by the pixels and the model)
    so a city is segmented once. ``make_segmenter`` is only called when something is not cached."""
    store: dict[str, Any] = json.loads(cache.read_text()) if cache.exists() else {}
    keys = [f"{MODEL}:{_key(img)}" for img in images]
    segment: Segmenter | None = None
    for img, key in zip(images, keys, strict=True):
        if key not in store:
            segment = segment or make_segmenter()
            store[key] = shares(segment(img))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(store) + "\n")
    return [dict(store[k]) for k in keys]


def as_stats(per_image: Mapping[str, float]) -> dict[str, float]:
    """Class shares named as statistics for the street report (``seg_building`` and so on)."""
    return {f"seg_{k}": v for k, v in per_image.items()}


__all__ = [
    "GROUPS", "NAMES", "Segmenter", "as_stats", "cached_shares", "load_model", "shares",
]  # fmt: skip
