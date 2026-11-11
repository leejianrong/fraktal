"""The prop mesh QA (scripts/props/meshqa.py, FRK-122).

Skipped without trimesh, which is a pod-only dependency of the props trial.
"""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("fast_simplification")


def _qa() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "props" / "meshqa.py"
    spec = importlib.util.spec_from_file_location("meshqa", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_prop_is_scaled_to_metres_and_stands_on_the_ground() -> None:
    import trimesh

    qa = _qa()
    body = trimesh.creation.box(extents=(1.0, 0.4, 0.5))
    wheel = trimesh.creation.cylinder(radius=0.25, height=0.1, sections=48)
    wheel.apply_translation([0.4, -0.25, -0.2])
    mesh = trimesh.util.concatenate([body, wheel])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # y up
    final, report = qa.process(mesh, 1.9, (1.9, 0.8, 1.1))
    assert max(report["extents_m"]) == pytest.approx(1.9, abs=0.01)
    assert final.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert final.bounds[:, :2].mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-6)


def test_tiny_floaters_go_but_separate_parts_stay() -> None:
    import trimesh

    qa = _qa()
    parts = [trimesh.creation.icosphere(radius=0.5, subdivisions=3) for _ in range(2)]
    parts[1].apply_translation([2.0, 0, 0])
    dust = trimesh.creation.icosphere(radius=0.01, subdivisions=0)  # 20 faces: under 2%
    dust.apply_translation([5.0, 0, 0])
    kept, pieces, kept_n = qa.clean(trimesh.util.concatenate([*parts, dust]))
    assert (pieces, kept_n) == (3, 2) and len(kept.faces) == 2 * 1280


def test_a_cube_is_not_a_container() -> None:
    qa = _qa()
    assert qa.proportions_off([6.1, 2.4, 2.6], (6.1, 2.4, 2.6)) is None
    assert qa.proportions_off([6.1, 5.9, 5.9], (6.1, 2.4, 2.6)) is not None
    # the expected box is scaled to the mesh's longest side
    assert qa.proportions_off([3.0, 1.2, 1.3], (6.0, 2.4, 2.6)) is None


def test_a_glb_is_written_y_up_and_comes_back_z_up() -> None:
    import trimesh

    qa = _qa()
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))  # long in x, thin in z
    box.apply_translation((0.0, 0.0, 0.25))  # standing on z = 0
    gltf = qa.to_gltf(box)
    np.testing.assert_allclose(gltf.extents, (2.0, 0.5, 1.0), atol=1e-9)  # height is now y
    np.testing.assert_allclose(gltf.bounds[0][1], 0.0, atol=1e-9)  # still on the floor
    back = qa.to_z_up(gltf)
    np.testing.assert_allclose(back.bounds, box.bounds, atol=1e-9)
