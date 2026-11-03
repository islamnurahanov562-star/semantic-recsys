"""Process the raw Netflix CSV into the canonical items dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import DataLoader
from src.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

_CONFIG_KEYS = {
    "raw_path",
    "column_map",
    "text_columns",
    "output_path",
    "max_chars",
    "min_chars",
}
_CONTRACT_COLUMNS = (
    "item_id",
    "title",
    "description",
    "category",
    "year",
    "text",
)
_REQUIRED_OUTPUT_COLUMNS = ("item_id", "title", "description", "category", "text")
_MINIMUM_ROWS = 1000


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the JSON processing configuration.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        The validated configuration dictionary.

    Raises:
        ValueError: If the JSON is invalid or required keys are missing.
    """
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The JSON configuration must contain an object.")

    missing_keys = _CONFIG_KEYS.difference(config)
    if missing_keys:
        raise ValueError(f"Configuration is missing keys: {sorted(missing_keys)}")
    if not isinstance(config["column_map"], dict):
        raise ValueError("Configuration key 'column_map' must be an object.")
    if not isinstance(config["text_columns"], list) or not all(
        isinstance(column, str) for column in config["text_columns"]
    ):
        raise ValueError("Configuration key 'text_columns' must be a list of strings.")
    return config


def sanity_check(items: pd.DataFrame) -> None:
    """Validate the processed items against the output data contract.

    Args:
        items: Processed item records to check.

    Raises:
        ValueError: If IDs, required values, or output column order are invalid.
    """
    missing_columns = [
        column for column in _REQUIRED_OUTPUT_COLUMNS if column not in items.columns
    ]
    if missing_columns:
        raise ValueError(f"Processed items are missing columns: {missing_columns}")

    expected_columns = [
        column for column in _CONTRACT_COLUMNS if column in items.columns
    ]
    if items.columns.tolist() != expected_columns:
        raise ValueError(
            "Processed columns must follow the data contract order: "
            f"{expected_columns}."
        )

    expected_ids = list(range(len(items)))
    if items["item_id"].tolist() != expected_ids:
        raise ValueError("item_id values must match row positions starting from zero.")
    if "year" in items.columns and str(items["year"].dtype) != "Int64":
        raise ValueError("The optional year column must use nullable Int64 dtype.")

    checked_columns = ["title", "description", "category", "text"]
    null_columns = (
        items[checked_columns].columns[items[checked_columns].isna().any()].tolist()
    )
    if null_columns:
        raise ValueError(f"Processed items contain null values in: {null_columns}")

    logger.info("Processed items passed the data contract sanity check.")


def process_data(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, preprocess, validate, and save the configured dataset.

    Args:
        config: Validated processing settings.

    Returns:
        The processed DataFrame and a copy of the preprocessing report.
    """
    loader = DataLoader(
        path=config["raw_path"],
        required_columns=["title", "description"],
        column_map=config["column_map"],
    )
    raw_items = loader.load()
    preprocessor = TextPreprocessor(
        text_columns=config["text_columns"],
        max_chars=config["max_chars"],
        min_chars=config["min_chars"],
    )
    processed_items = preprocessor.transform(raw_items)
    sanity_check(processed_items)
    TextPreprocessor.save(processed_items, config["output_path"])
    return processed_items, dict(preprocessor.report_)


def print_summary(
    items: pd.DataFrame,
    report: dict[str, int],
    output_path: str | Path,
) -> None:
    """Print processing counts and descriptive statistics.

    Args:
        items: Processed item records.
        report: Preprocessing counters from TextPreprocessor.
        output_path: Destination CSV path.
    """
    word_lengths = items["text"].astype("string").str.split().str.len()
    median_words = float(word_lengths.median()) if not word_lengths.empty else math.nan
    median_display = "N/A" if math.isnan(median_words) else f"{median_words:.2f}"
    genre_count = int(items["category"].nunique())

    print("Сводка обработки данных")
    print(f"Строк на входе: {report['rows_in']}")
    print(f"Удалено коротких строк: {report['dropped_short']}")
    print(f"Удалено дубликатов: {report['dropped_duplicates']}")
    print(f"Строк на выходе: {report['rows_out']}")
    print(f"Обрезано текстов: {report['truncated']}")
    print(f"Уникальных жанров: {genre_count}")
    print(f"Медианная длина text в словах: {median_display}")
    print(f"Файл: {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured data-processing pipeline.

    Args:
        argv: Optional command-line arguments, used for testing.

    Returns:
        Process exit status, nonzero if processing fails or too few rows remain.
    """
    parser = argparse.ArgumentParser(description="Process the raw Netflix titles CSV.")
    parser.add_argument(
        "--config",
        default="configs/dataset.json",
        help="Path to the JSON processing config (default: configs/dataset.json).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        config = load_config(args.config)
        processed_items, report = process_data(config)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("Data processing failed: %s", error)
        return 1

    print_summary(processed_items, report, config["output_path"])
    if len(processed_items) < _MINIMUM_ROWS:
        logger.error(
            "Only %d rows remain after preprocessing; at least %d are required.",
            len(processed_items),
            _MINIMUM_ROWS,
        )
        return 1

    logger.info("Data processing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
