"""Tests for text cleaning and item preprocessing."""

from pathlib import Path

import pandas as pd

from src.preprocessor import TextPreprocessor


def test_html_tags_and_entities_are_cleaned() -> None:
    """HTML tags are removed and entities are unescaped."""
    dataframe = pd.DataFrame(
        {
            "title": ["<b>Hi</b> &amp; bye"],
            "description": ["A sufficiently long description"],
        }
    )

    result = TextPreprocessor(["title", "description"], min_chars=1).transform(
        dataframe
    )

    assert result.loc[0, "title"] == "Hi & bye"


def test_multiple_spaces_and_newlines_are_collapsed() -> None:
    """Repeated whitespace is normalized to single spaces."""
    dataframe = pd.DataFrame(
        {
            "title": ["  A   title\nwith\tspaces  "],
            "description": ["A sufficiently long description"],
        }
    )

    result = TextPreprocessor(["title", "description"], min_chars=1).transform(
        dataframe
    )

    assert result.loc[0, "title"] == "A title with spaces"


def test_empty_and_missing_category_become_unknown() -> None:
    """Empty category values and absent category columns become unknown."""
    with_empty_category = pd.DataFrame(
        {
            "title": ["A title"],
            "category": ["  "],
            "description": ["A sufficiently long description"],
        }
    )
    without_category = pd.DataFrame(
        {
            "title": ["Another title"],
            "description": ["Another sufficiently long description"],
        }
    )
    preprocessor = TextPreprocessor(["title", "category", "description"], min_chars=1)

    empty_result = preprocessor.transform(with_empty_category)
    missing_result = preprocessor.transform(without_category)

    assert empty_result.loc[0, "category"] == "unknown"
    assert missing_result.loc[0, "category"] == "unknown"
    assert "unknown" in missing_result.loc[0, "text"]


def test_text_joins_nonempty_parts_in_order_with_full_genres() -> None:
    """Text keeps all genres and joins populated fields in configured order."""
    dataframe = pd.DataFrame(
        {
            "title": ["A title"],
            "type": ["Movie"],
            "category": ["Dramas, International Movies"],
            "description": ["A plot"],
        }
    )

    result = TextPreprocessor(
        ["title", "type", "category", "description"], min_chars=1
    ).transform(dataframe)

    assert result.loc[0, "text"] == (
        "A title. Movie. Dramas, International Movies. A plot"
    )


def test_category_is_reduced_to_first_genre_after_text_creation() -> None:
    """The output category keeps only its first genre."""
    dataframe = pd.DataFrame(
        {
            "title": ["A title"],
            "category": ["Dramas, International Movies"],
            "description": ["A plot"],
        }
    )

    result = TextPreprocessor(
        ["title", "category", "description"], min_chars=1
    ).transform(dataframe)

    assert result.loc[0, "category"] == "Dramas"
    assert "Dramas, International Movies" in result.loc[0, "text"]


def test_short_rows_are_removed_and_reported() -> None:
    """Rows below min_chars are removed and counted in report_."""
    dataframe = pd.DataFrame(
        {
            "title": ["x", "A sufficiently long title"],
            "description": ["y", "A sufficiently long description"],
        }
    )
    preprocessor = TextPreprocessor(["title", "description"], min_chars=20)

    result = preprocessor.transform(dataframe)

    assert len(result) == 1
    assert result.loc[0, "title"] == "A sufficiently long title"
    assert preprocessor.report_["dropped_short"] == 1


def test_case_insensitive_duplicate_pairs_are_removed_and_reported() -> None:
    """Duplicate title-description pairs ignore letter case."""
    dataframe = pd.DataFrame(
        {
            "title": ["A Unique Title", "a unique title"],
            "description": ["A unique description", "a UNIQUE description"],
        }
    )
    preprocessor = TextPreprocessor(["title", "description"], min_chars=1)

    result = preprocessor.transform(dataframe)

    assert len(result) == 1
    assert result.loc[0, "title"] == "A Unique Title"
    assert preprocessor.report_["dropped_duplicates"] == 1


def test_text_truncation_respects_word_boundaries_and_is_reported() -> None:
    """Long text is cut at a word boundary and counted as truncated."""
    dataframe = pd.DataFrame({"title": ["alpha beta gamma"]})
    preprocessor = TextPreprocessor(["title"], max_chars=9, min_chars=1)

    result = preprocessor.transform(dataframe)

    assert result.loc[0, "text"] == "alpha"
    assert len(result.loc[0, "text"]) <= 9
    assert preprocessor.report_["truncated"] == 1


def test_item_ids_match_output_positions_and_are_first_column() -> None:
    """Output IDs are sequential after filtering and appear first."""
    dataframe = pd.DataFrame(
        {
            "title": ["First item title", "Second item title"],
            "description": ["First item description", "Second item description"],
        },
        index=[7, 3],
    )

    result = TextPreprocessor(["title", "description"], min_chars=1).transform(
        dataframe
    )

    assert result.columns[0] == "item_id"
    assert result["item_id"].tolist() == [0, 1]
    assert result.index.tolist() == [0, 1]


def test_transform_does_not_modify_input_dataframe() -> None:
    """Cleaning and derived columns do not mutate the source DataFrame."""
    dataframe = pd.DataFrame(
        {
            "title": ["<b>Original</b> title"],
            "category": ["Dramas, International Movies"],
            "description": ["Original description text"],
        }
    )
    original = dataframe.copy(deep=True)

    TextPreprocessor(["title", "category", "description"], min_chars=1).transform(
        dataframe
    )

    pd.testing.assert_frame_equal(dataframe, original)


def test_save_creates_parent_directories_and_round_trips_csv(
    tmp_path: Path,
) -> None:
    """CSV output creates missing directories and preserves cell values."""
    dataframe = pd.DataFrame(
        {
            "item_id": [0],
            "title": ["Café"],
            "description": ["A story"],
            "category": ["Dramas"],
            "text": ["Café. A story"],
        }
    )
    csv_path = tmp_path / "new" / "nested" / "items.csv"

    TextPreprocessor.save(dataframe, csv_path)

    assert csv_path.exists()
    saved = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(saved, dataframe, check_dtype=False)
