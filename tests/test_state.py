from stepbook.state.merge import count_deletions, diff, merge


def test_deep_merge_touches_only_named_keys():
    base = {"inventory": {"shelf_1": "item_7", "shelf_42": "item_12"}, "x": 1}
    out = merge(base, {"inventory": {"shelf_42": None}})
    assert out == {"inventory": {"shelf_1": "item_7"}, "x": 1}
    assert base["inventory"]["shelf_42"] == "item_12"  # no mutation


def test_null_deletes_and_omission_is_noop():
    assert merge({"a": 1, "b": 2}, {"a": None}) == {"b": 2}
    assert merge({"a": 1, "b": 2}, {}) == {"a": 1, "b": 2}


def test_scalars_and_lists_replace():
    assert merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}
    assert merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}


def test_diff_and_deletion_count():
    assert diff({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 3}, "d": 0}) == ["b.c", "d"]
    assert count_deletions({"a": None, "b": {"c": None, "d": 1}}) == 2
