"""Tests for CSV loading and validation."""

from pathlib import Path

import pytest

from src.data_loader import DataLoader


def test_loads_valid_csv_with_expected_shape_and_columns(tmp_path: Path) -> None:
    """A valid CSV is loaded with its rows and columns intact."""
    csv_path = tmp_path / "valid.csv"
    csv_path.write_text("show_id,title\ns1,Alpha\ns2,Beta\n", encoding="utf-8")

    dataframe = DataLoader(csv_path, required_columns=["show_id", "title"]).load()

    assert dataframe.shape == (2, 2)
    assert list(dataframe.columns) == ["show_id", "title"]


def test_column_map_renames_source_columns(tmp_path: Path) -> None:
    """Source headers are renamed to the required canonical names."""
    csv_path = tmp_path / "mapped.csv"
    csv_path.write_text("raw_title,summary\nAlpha,Story\n", encoding="utf-8")

    dataframe = DataLoader(
        csv_path,
        required_columns=["title", "description"],
        column_map={"raw_title": "title", "summary": "description"},
    ).load()

    assert list(dataframe.columns) == ["title", "description"]
    assert dataframe.loc[0, "title"] == "Alpha"


def test_column_map_with_missing_source_column_raises_value_error(
    tmp_path: Path,
) -> None:
    """A mapping key absent from the CSV is reported as invalid."""
    csv_path = tmp_path / "missing_source.csv"
    csv_path.write_text("title\nAlpha\n", encoding="utf-8")

    with pytest.raises(ValueError, match="raw_title"):
        DataLoader(
            csv_path,
            required_columns=["title"],
            column_map={"raw_title": "title"},
        ).load()


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    """A nonexistent source path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        DataLoader(tmp_path / "absent.csv", required_columns=["title"]).load()


def test_csv_with_header_only_raises_value_error(tmp_path: Path) -> None:
    """A CSV with headers but no data rows is rejected."""
    csv_path = tmp_path / "header_only.csv"
    csv_path.write_text("title,description\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        DataLoader(csv_path, required_columns=["title"]).load()


def test_missing_required_column_raises_value_error_with_column_name(
    tmp_path: Path,
) -> None:
    """The validation error names a required column absent from the CSV."""
    csv_path = tmp_path / "missing_required.csv"
    csv_path.write_text("title\nAlpha\n", encoding="utf-8")

    with pytest.raises(ValueError, match="description"):
        DataLoader(csv_path, required_columns=["title", "description"]).load()


def test_fully_empty_required_column_raises_value_error(tmp_path: Path) -> None:
    """A required column containing only empty cells is rejected."""
    csv_path = tmp_path / "empty_column.csv"
    csv_path.write_text("title,description\n,Story one\n,Story two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="title"):
        DataLoader(csv_path, required_columns=["title"]).load()


def test_latin1_csv_with_non_ascii_text_loads_successfully(tmp_path: Path) -> None:
    """A latin-1 CSV containing non-ASCII text is decoded successfully."""
    csv_path = tmp_path / "latin1.csv"
    csv_path.write_bytes("title\nCafé\n".encode("latin-1"))

    dataframe = DataLoader(csv_path, required_columns=["title"]).load()

    assert dataframe.loc[0, "title"] == "Café"
