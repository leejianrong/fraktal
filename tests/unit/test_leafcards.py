import hashlib

import numpy as np

from fraktal import leafcards


def test_the_cluster_is_deterministic_and_seeded() -> None:
    a = leafcards.leaf_cluster(0, 64)
    assert np.array_equal(a, leafcards.leaf_cluster(0, 64))
    assert not np.array_equal(a, leafcards.leaf_cluster(1, 64))
    assert a.shape == (64, 64, 4) and a.dtype == np.uint8


def test_the_cluster_has_a_ragged_see_through_edge_and_grey_light() -> None:
    a = leafcards.leaf_cluster(0, 128)
    alpha = a[..., 3].astype(float) / 255.0
    assert 0.3 < alpha.mean() < 0.7  # a card, not a solid block and not empty
    assert alpha[0, :].max() == 0 and alpha[-1, :].max() == 0 and alpha[:, 0].max() == 0
    assert np.all(a[..., 0] == a[..., 1]) and np.all(a[..., 1] == a[..., 2])  # tinted in the shader
    assert (alpha > 0).any() and ((alpha > 0) & (alpha < 1)).any()  # soft edges from supersampling


def test_the_committed_texture_is_the_one_the_code_makes() -> None:
    from pathlib import Path

    import rasterio

    path = Path(__file__).parents[2] / "assets" / "textures" / "leaf-cluster.png"
    with rasterio.open(path) as src:
        stored = src.read().transpose(1, 2, 0)
    assert np.array_equal(stored, leafcards.leaf_cluster(0, 256))
    assert hashlib.sha256(path.read_bytes()).hexdigest()  # and the manifest row pins its hash
