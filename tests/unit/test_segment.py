from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fraktal import segment, streetview


def _labels() -> np.ndarray[Any, Any]:
    names = np.full((10, 10), "sky-other-merged", dtype="<U24")
    names[5:] = "road"
    names[5:, :3] = "car"
    names[8:, 7:] = "something-unlisted"
    return names


def test_shares_group_coco_classes_and_sum_to_one() -> None:
    s = segment.shares(_labels())
    assert s["sky"] == pytest.approx(0.5) and s["vehicle"] == pytest.approx(0.15)
    assert s["other"] == pytest.approx(0.06)  # an unlisted class falls in other
    assert s["road"] == pytest.approx(0.29)
    assert sum(s.values()) == pytest.approx(1.0) and set(s) == set(segment.NAMES)


def test_results_are_cached_by_pixels_and_the_model_is_built_only_when_needed(
    tmp_path: Path,
) -> None:
    calls = {"built": 0, "ran": 0}

    def make() -> segment.Segmenter:
        calls["built"] += 1

        def run(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            calls["ran"] += 1
            return _labels()

        return run

    a = np.zeros((4, 4, 3), dtype="uint8")
    b = np.ones((4, 4, 3), dtype="uint8")
    cache = tmp_path / "seg" / "cache.json"
    first = segment.cached_shares([a, b], make, cache)
    assert calls == {"built": 1, "ran": 2} and first[0] == first[1]
    again = segment.cached_shares([b, a, a], make, cache)
    assert calls == {"built": 1, "ran": 2} and again == [first[1], first[0], first[0]]  # no model
    assert segment.cached_shares([a + 2], make, cache) and calls["ran"] == 3


def test_compare_takes_class_shares_as_extra_statistics(tmp_path: Path) -> None:
    img = np.full((20, 20, 3), 100, dtype="uint8")
    photos = [streetview.Photo(tmp_path, "commons", "x", img)] * 3
    frames = [streetview.Photo(tmp_path, "synthetic", "x", img)] * 3

    def extra(items: Any) -> list[dict[str, float]]:
        # photos are leafy, frames have none: a vegetation gap the colour statistics would not name
        veg = 0.3 if items[0].source == "commons" else 0.0
        return [segment.as_stats({"vegetation": veg, "road": 0.2}) for _ in items]

    report = streetview.compare(photos, frames, extra)
    assert "seg_vegetation" in report["stats"] and "sky_share" in report["stats"]
    assert report["stats"]["seg_vegetation"]["flag"] and not report["stats"]["seg_road"]["flag"]


def test_load_model_explains_a_missing_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "transformers", None)  # an import of it now fails
    with pytest.raises(ValueError, match="install-seg"):
        segment.load_model()
