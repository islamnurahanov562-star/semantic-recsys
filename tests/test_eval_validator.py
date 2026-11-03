"""Tests for evaluation-query validation and relevance ID parsing."""

import pandas as pd

from scripts.build_eval_queries import EvalQueryValidator, _build_output


def _items() -> pd.DataFrame:
    """Return a small in-memory catalog with source descriptions."""
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "description": [
                "A quiet ocean journey under a sky full of stars.",
                "Friends solve a mystery in their mountain town.",
                "A chef opens a cafe beside the harbor.",
            ],
        }
    )


def _template_row(
    query: str = "Find a calm sea story",
    item_id: int = 1,
    extra_relevant_ids: str = "",
) -> pd.DataFrame:
    """Build one template row without reading external files."""
    return pd.DataFrame(
        {
            "item_id": [item_id],
            "query": [query],
            "extra_relevant_ids": [extra_relevant_ids],
        }
    )


def test_valid_query_and_relevant_ids_are_sorted_unique() -> None:
    """The source item and valid extra IDs form sorted unique output IDs."""
    template = _template_row(extra_relevant_ids="3|2|1| 2 |")
    items = _items()

    assert EvalQueryValidator().validate(template, items) == []
    output = _build_output(template)

    assert output.to_dict("records") == [
        {"query": "Find a calm sea story", "relevant_ids": "1|2|3"}
    ]


def test_missing_item_id_is_reported() -> None:
    """A source item ID absent from items is reported with its row."""
    issues = EvalQueryValidator().validate(_template_row(item_id=99), _items())

    assert any("Source item_id 99 does not exist" in issue for issue in issues)


def test_non_integer_extra_id_is_reported() -> None:
    """Non-integer extra relevance tokens are reported."""
    issues = EvalQueryValidator().validate(
        _template_row(extra_relevant_ids="2|not-an-id"), _items()
    )

    assert any("not-an-id' is not an integer" in issue for issue in issues)


def test_extra_id_absent_from_items_is_reported() -> None:
    """An integer extra ID must also exist in the item catalog."""
    issues = EvalQueryValidator().validate(
        _template_row(extra_relevant_ids="99"), _items()
    )

    assert any("Extra relevant ID 99 does not exist" in issue for issue in issues)


def test_query_shorter_than_three_words_is_reported() -> None:
    """Queries need at least three punctuation-insensitive word tokens."""
    issues = EvalQueryValidator().validate(_template_row(query="Sea! Story"), _items())

    assert any("at least three words" in issue for issue in issues)


def test_duplicate_queries_are_detected_without_case_sensitivity() -> None:
    """Case-only query differences are reported on both duplicate rows."""
    template = pd.DataFrame(
        {
            "item_id": [1, 2],
            "query": ["Find a calm sea story", "find a calm sea story"],
            "extra_relevant_ids": ["", ""],
        }
    )

    issues = EvalQueryValidator().validate(template, _items())

    duplicate_issues = [issue for issue in issues if "duplicated" in issue]
    assert len(duplicate_issues) == 2


def test_five_consecutive_description_words_are_reported() -> None:
    """A copied phrase of five consecutive normalized words is rejected."""
    template = _template_row(query="quiet, ocean journey under a")

    issues = EvalQueryValidator().validate(template, _items())

    assert any("five consecutive words" in issue for issue in issues)


def test_validator_collects_multiple_problems_in_one_pass() -> None:
    """The validator reports malformed IDs and description leakage together."""
    template = _template_row(
        query="quiet ocean journey under a",
        item_id=1,
        extra_relevant_ids="bad-id|88",
    )

    issues = EvalQueryValidator().validate(template, _items())

    assert len(issues) >= 3
    assert any("not an integer" in issue for issue in issues)
    assert any("Extra relevant ID 88 does not exist" in issue for issue in issues)
    assert any("five consecutive words" in issue for issue in issues)


def test_missing_columns_do_not_hide_other_detectable_problems() -> None:
    """Structural issues are returned alongside checks supported by available data."""
    template = pd.DataFrame({"item_id": [1], "query": ["Only two"]})
    items = pd.DataFrame({"item_id": [1]})

    issues = EvalQueryValidator().validate(template, items)

    assert any("extra_relevant_ids" in issue for issue in issues)
    assert any("description" in issue for issue in issues)
    assert any("at least three words" in issue for issue in issues)
