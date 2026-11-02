from fraktal.ids import feature_id, substream


def test_substream_is_deterministic() -> None:
    a = substream(42, "terrain").random(5)
    b = substream(42, "terrain").random(5)
    assert (a == b).all()


def test_substreams_differ_by_name_and_seed() -> None:
    assert (substream(42, "terrain").random(5) != substream(42, "roads").random(5)).any()
    assert (substream(42, "terrain").random(5) != substream(43, "terrain").random(5)).any()


def test_adding_a_stream_does_not_change_existing_ones() -> None:
    before = substream(7, "buildings").random(5)
    substream(7, "new-feature-type").random(100)
    after = substream(7, "buildings").random(5)
    assert (before == after).all()


def test_feature_id_is_stable_and_order_independent() -> None:
    first = feature_id("w1", "building", 10, 20)
    feature_id("w1", "building", 99, 99)
    assert feature_id("w1", "building", 10, 20) == first


def test_feature_id_depends_on_world_kind_and_parts() -> None:
    base = feature_id("w1", "building", 1)
    assert base != feature_id("w2", "building", 1)
    assert base != feature_id("w1", "road", 1)
    assert base != feature_id("w1", "building", 2)
