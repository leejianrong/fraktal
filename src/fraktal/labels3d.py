"""Label images from the 3D renderer (FRK-61).

Blender writes ``labels.png`` (or ``labels/frame_NNNN.png``) next to the beauty render, from the
same camera, with flat colours and no anti-aliasing: red is the class, green and blue are a 16-bit
instance number (the index into ``labels.json["buildings"]`` or the tree list, plus one; zero means
none). So a label is exact for any camera, oblique ones included, and hidden surfaces never leak.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CLASSES = {
    "none": 0,
    "ground": 1,
    "building": 2,
    "tree": 3,
    "vehicle": 4,
    "pole": 5,
    "person": 6,
    "drone": 7,
    "yard_object": 9,  # 8 is reserved for ``wall`` (FRK-106); classes are append-only
}
MAX_INSTANCES = 65534


class LabelError(ValueError):
    pass


def decode(rgb: Any) -> tuple[Any, Any]:
    """``(class, instance)`` arrays from an (H, W, 3) or (3, H, W) uint8 label image."""
    image = np.asarray(rgb)
    if image.ndim != 3:
        raise LabelError(f"expected a 3-channel image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise LabelError(f"expected a 3-channel image, got shape {image.shape}")
    semantic = image[..., 0].astype("uint8")
    instance = (image[..., 1].astype("int32") << 8) | image[..., 2].astype("int32")
    return semantic, instance


def encode(semantic: Any, instance: Any) -> Any:
    """The inverse of ``decode``: an (H, W, 3) uint8 image."""
    inst = np.asarray(instance, dtype="int32")
    if inst.max(initial=0) > MAX_INSTANCES:
        raise LabelError(f"more than {MAX_INSTANCES} instances in one tile")
    high, low = (inst >> 8).astype("uint8"), (inst & 255).astype("uint8")
    return np.stack([np.asarray(semantic, dtype="uint8"), high, low], axis=-1)


def meta(
    building_ids: list[str],
    trees: int,
    vehicles: int = 0,
    poles: int = 0,
    people: int = 0,
    drones: int = 0,
    props: int = 0,
) -> dict[str, Any]:
    """The ``labels`` block of ``render.json``."""
    return {
        "classes": CLASSES,
        "encoding": "R = class, G and B = instance + 1 as a 16-bit number (0 = none)",
        "buildings": building_ids,
        "trees": trees,
        "vehicles": vehicles,
        "poles": poles,
        "people": people,
        "drones": drones,
        "props": props,
    }


__all__ = ["CLASSES", "LabelError", "decode", "encode", "meta"]
