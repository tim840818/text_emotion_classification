import text_preprocess
import pytest

@pytest.mark.parametrize("text, expected", [
    ("This is a test.", "this is a test"),
    ("This is a testing?", "this is a testing"),
    ("I'm a test.", "i'm a test"),
])
def test_clean_text(text, expected):
    assert text_preprocess.clean_text(text) == expected


@pytest.mark.parametrize("text, do_stem, expected", [
    ("This is a testing", False, "This testing"),
    ("This is a testing", True, "test"),
    ("This is a testing!", True, "testing!"),
    ("This IS a testing", True, "test"),
    ("I'm a testing", True, "test"),
])
def test_remove_stopwords(text, do_stem, expected):
    assert text_preprocess.remove_stopwords(text, do_stem) == expected


@pytest.mark.parametrize("text, do_stem, expected", [
    ("This is a testing", True, "test"),
    ("This is a testing!", True, "test"),
    ("This IS a testing", True, "test"),
    ("I'm a testing", True, "test"),
])
def test_preprocess_text(text, do_stem, expected):
    assert text_preprocess.preprocess_text(text, do_stem) == expected
