import numpy as np
import pytest

from fraktal.propgen import GENERATORS, MATERIALS, Part

# (min, max) of the bounding box sides: length x, width y, height z, in metres
LIMITS = {
    "container": ((5.9, 12.4), (2.3, 2.6), (2.5, 2.7)),
    "water_tank": ((0.9, 1.4), (0.9, 1.4), (1.4, 2.5)),
    "market_stall": ((1.8, 3.2), (1.2, 2.2), (2.0, 2.6)),
    "kerb_stall": ((1.8, 3.2), (1.2, 2.2), (2.0, 2.6)),
    "jeepney": ((6.2, 7.3), (1.9, 2.15), (2.1, 2.3)),
    "car": ((4.4, 5.1), (1.7, 2.0), (1.2, 1.5)),
    "signboard": ((1.2, 2.6), (0.05, 0.3), (1.7, 2.7)),
    "tuk_tuk": ((2.2, 3.1), (1.2, 1.7), (1.6, 1.9)),
    "silo": ((2.8, 5.2), (2.8, 5.2), (6.5, 11.0)),
    "motorbike": ((1.8, 2.1), (0.6, 0.85), (0.95, 1.2)),
    "minibus": ((4.8, 5.6), (1.8, 2.0), (2.1, 2.6)),
    "dumpster": ((1.8, 2.5), (1.2, 1.6), (1.2, 1.5)),
    "barrel_stack": ((0.55, 1.9), (0.55, 1.3), (0.85, 0.95)),
    "rubbish_pile": ((0.3, 4.5), (0.3, 4.5), (0.08, 1.0)),
    "pedestrian": ((0.2, 0.5), (0.35, 0.75), (1.5, 1.95)),
}


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_deterministic_and_seeds_differ(kind: str) -> None:
    a, b = GENERATORS[kind](7), GENERATORS[kind](7)
    assert len(a.parts) == len(b.parts)
    for p, q in zip(a.parts, b.parts, strict=True):
        assert p.material == q.material
        assert np.array_equal(p.verts, q.verts) and np.array_equal(p.faces, q.faces)
    # a container has two parameters, so one other seed can match; some seed must differ
    others = [GENERATORS[kind](k) for k in range(8, 20)]
    assert any(
        o.size() != a.size() or [p.material for p in o.parts] != [p.material for p in a.parts]
        for o in others
    )


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_size_stands_on_ground_and_mesh_is_valid(kind: str) -> None:
    for seed in range(12):
        prop = GENERATORS[kind](seed)
        lo, _ = prop.bounds()
        assert lo[2] == pytest.approx(0.0, abs=0.02), (kind, seed, lo[2])
        for got, (low, high) in zip(prop.size(), LIMITS[kind], strict=True):
            assert low <= got <= high, (kind, seed, prop.size())
        for part in prop.parts:
            assert part.material in MATERIALS
            assert np.isfinite(part.verts).all()
            assert part.faces.min() >= 0 and part.faces.max() < len(part.verts)


def _signed_volume(part: Part) -> float:
    v, f = part.verts, part.faces
    return float(np.einsum("ij,ij->", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])) / 6)


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_parts_face_outwards(kind: str) -> None:
    # Blender recalculates normals, but a viewer that culls back faces (the Godot demo) does not
    for part in GENERATORS[kind](3).parts:
        assert _signed_volume(part) > 0, (kind, part.material)
