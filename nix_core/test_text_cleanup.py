from text_cleanup import clean_response_text


def test_cleanup_collapses_spaces_and_trailing_blanks_without_merging_lines():
    assert clean_response_text("Hello    \tthere   !  \nNext line  \n\nDone  ") == (
        "Hello there!\nNext line\n\nDone"
    )


def test_cleanup_preserves_fenced_code_interior_and_line_breaks():
    text = "Text    here\n```python  \nvalue    =  2  \n```  \nAfter   code  "
    assert clean_response_text(text) == (
        "Text here\n```python\nvalue    =  2  \n```\nAfter code"
    )


def test_cleanup_normalizes_nonbreaking_spaces():
    assert clean_response_text("Hello\u00a0\u00a0there") == "Hello there"


def test_luna_v7_reasoning_markup_is_removed_only_for_v7():
    from luna_runtime import (
        LUNA_V6_MODEL_ID,
        LUNA_V7_MODEL_ID,
        strip_v7_reasoning_markup,
    )

    answer = "<think>private\nanalysis</think>Hello    "
    assert strip_v7_reasoning_markup(LUNA_V7_MODEL_ID, answer) == "Hello    "
    assert strip_v7_reasoning_markup(LUNA_V6_MODEL_ID, answer) == answer
