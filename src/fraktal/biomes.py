"""Biome presets and their config schema (V4).

A biome is a named, complete :class:`~fraktal.procedural.ProceduralConfig`. ``resolve`` turns a
name plus optional overrides into one config, rejecting unknown biomes and unknown keys loudly so a
typo never silently falls back to a default. Overrides are nested like the config itself, for
example ``{"aridity": 0.8, "roads": {"local_spacing_m": 60}}``. Later biome stories register their
presets in ``BIOMES``; each preset is just data that the existing generators already read.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from fraktal.procedural import DEFAULT_BIOME, ProceduralConfig


class BiomeError(ValueError):
    pass


@dataclass(frozen=True)
class Biome:
    name: str
    description: str
    config: ProceduralConfig


BIOMES: dict[str, Biome] = {
    DEFAULT_BIOME: Biome(
        DEFAULT_BIOME,
        "Dry, low-rise Sahelian city: dusty ground, sparse trees (the SpaceNet Khartoum look).",
        ProceduralConfig(biome=DEFAULT_BIOME),
    ),
}


def names() -> list[str]:
    return sorted(BIOMES)


def _coerce(path: str, current: Any, value: Any) -> Any:
    """``value`` as the type of ``current``, or a BiomeError naming the key."""
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise BiomeError(f"{path} must be true or false, got {value!r}")
        return value
    if isinstance(current, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BiomeError(f"{path} must be a number, got {value!r}")
        return float(value) if isinstance(current, float) else int(value)
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(current):
            raise BiomeError(f"{path} must be a list of {len(current)} numbers, got {value!r}")
        pairs = zip(current, value, strict=True)
        return tuple(_coerce(f"{path}[{i}]", c, v) for i, (c, v) in enumerate(pairs))
    if isinstance(current, str):
        if not isinstance(value, str):
            raise BiomeError(f"{path} must be text, got {value!r}")
        return value
    raise BiomeError(f"{path} cannot be overridden")


def _apply(obj: Any, overrides: dict[str, Any], path: str = "") -> Any:
    fields = {f.name for f in dataclasses.fields(obj)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        where = f"{path}{key}"
        if key not in fields or (not path and key == "biome"):
            allowed = sorted(fields - {"biome"})
            raise BiomeError(f"unknown biome setting {where!r}; allowed: {allowed}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise BiomeError(f"{where} must be an object of settings")
            changes[key] = _apply(current, value, f"{where}.")
        else:
            changes[key] = _coerce(where, current, value)
    return dataclasses.replace(obj, **changes)


def resolve(name: str, overrides: dict[str, Any] | None = None) -> ProceduralConfig:
    """The config for biome ``name`` with ``overrides`` applied on top."""
    if name not in BIOMES:
        raise BiomeError(f"unknown biome {name!r}; choose from {', '.join(names())}")
    config = BIOMES[name].config
    return _apply(config, overrides) if overrides else config
