import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

from fraktal.labels import independent_building_mask


def test_a_rasterised_label_lies_between_the_strict_and_touching_masks() -> None:
    # Footprints whose edges fall exactly on pixel centres (1 m pixels, edges at x.5), as
    # snapped dense_urban buildings do. The rasteriser picks some edge pixels; the independent
    # check must accept that, or the whole tile is dropped.
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    geoms = [box(2.5, 2.5, 6.5, 7.5), box(0.5, 0.5, 1.5, 1.5)]
    label = rasterize([(g, 1) for g in geoms], (10, 10), transform=transform, dtype="uint8")
    inner = independent_building_mask(geoms, transform, (10, 10))
    outer = independent_building_mask(geoms, transform, (10, 10), touching=True)
    assert not ((inner > 0) & (label == 0)).any()
    assert not ((outer == 0) & (label > 0)).any()
    assert (outer > inner).any()  # the case really has edge pixels


def test_without_edge_pixels_the_two_masks_coincide() -> None:
    transform = from_origin(0.0, 10.0, 1.0, 1.0)
    geoms = [box(2.2, 2.2, 6.8, 7.8)]
    inner = independent_building_mask(geoms, transform, (10, 10))
    outer = independent_building_mask(geoms, transform, (10, 10), touching=True)
    assert np.array_equal(inner, outer)  # no ties: the two masks coincide
