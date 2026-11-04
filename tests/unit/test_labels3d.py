import numpy as np
import pytest

from fraktal import labels3d


def test_encode_decode_round_trip_for_large_instance_numbers() -> None:
    semantic = np.array([[0, 1], [2, 3]], dtype="uint8")
    instance = np.array([[0, 0], [300, 65534]], dtype="int32")
    image = labels3d.encode(semantic, instance)
    assert image.shape == (2, 2, 3) and image.dtype == np.uint8
    out_class, out_inst = labels3d.decode(image)
    assert (out_class == semantic).all() and (out_inst == instance).all()
    channels_first = np.moveaxis(image, -1, 0)  # rasterio order
    assert (labels3d.decode(channels_first)[1] == instance).all()


def test_too_many_instances_and_bad_shapes_are_refused() -> None:
    with pytest.raises(labels3d.LabelError):
        labels3d.encode(np.zeros((1, 1)), np.full((1, 1), 70000))
    with pytest.raises(labels3d.LabelError):
        labels3d.decode(np.zeros((4, 4)))
