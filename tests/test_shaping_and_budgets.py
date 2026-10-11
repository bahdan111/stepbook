import pytest

from stepbook.runtime.observations import ObservationShaper
from stepbook.state.validator import PatchError, Validator


def test_short_observation_untouched(tmp_path):
    sh = ObservationShaper(max_chars=100, archive_dir=tmp_path)
    text, info = sh.shape("hello", 0)
    assert text == "hello" and info == {"truncated": False}


def test_long_observation_truncated_and_archived(tmp_path):
    sh = ObservationShaper(max_chars=100, head_chars=40, tail_chars=20, archive_dir=tmp_path)
    big = "A" * 40 + "B" * 500 + "C" * 20
    text, info = sh.shape(big, 7)
    assert text.startswith("A" * 40) and text.endswith("C" * 20)
    assert "500 characters omitted" in text and "00007.txt" in text
    assert info["truncated"] and (tmp_path / "00007.txt").read_text() == big


def test_field_budget_enforced():
    schema = {
        "type": "object",
        "properties": {"notes": {"type": "object", "x-budget": 40, "additionalProperties": {"type": "string"}}},
        "additionalProperties": False,
    }
    v = Validator(schema)
    v.apply({}, {"notes": {"a": "short"}})
    with pytest.raises(PatchError, match="over its budget"):
        v.apply({}, {"notes": {"a": "x" * 100}})


def test_truncation_marker_tells_the_model_how_to_read_the_rest(tmp_path):
    """Truncation markers explain how to read archived output."""
    from stepbook.runtime.observations import ObservationShaper

    sh = ObservationShaper(max_chars=100, head_chars=40, tail_chars=20, archive_dir=tmp_path)
    text, info = sh.shape("z" * 600, 9)
    assert info["truncated"] and "run `cat " in text and str(tmp_path / "00009.txt") in text


def test_default_observation_cap_is_twelve_thousand():
    from stepbook.runtime.observations import ObservationShaper

    sh = ObservationShaper()
    assert (sh.max_chars, sh.head_chars, sh.tail_chars) == (12000, 8000, 3000)
    # The defaults must be the numbers shaping actually applies, not merely the fields it stores.
    assert sh.shape("a" * 12000, 0) == ("a" * 12000, {"truncated": False})
    body = "a" * 8000 + "b" * 2000 + "c" * 3000
    text, info = sh.shape(body, 4)
    assert text == "a" * 8000 + "\n[... 2000 characters omitted ...]\n" + "c" * 3000
    assert info == {"truncated": True, "archived": None, "original_chars": 13000}
