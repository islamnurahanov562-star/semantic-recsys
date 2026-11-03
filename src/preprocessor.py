"""Clean catalog text and prepare items for embedding and search."""

import html
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Clean item text, derive categories, and assign stable row identifiers."""

    def __init__(
        self,
        text_columns: Sequence[str],
        max_chars: int = 1000,
        min_chars: int = 20,
    ) -> None:
        """Initialize preprocessing settings.

        Args:
            text_columns: Columns concatenated to create the embedding text.
            max_chars: Maximum text length after truncation.
            min_chars: Minimum text length retained before deduplication.
        """
        self._text_columns = tuple(text_columns)
        self._max_chars = max_chars
        self._min_chars = min_chars
        self.report_: dict[str, int] = {
            "rows_in": 0,
            "dropped_short": 0,
            "dropped_duplicates": 0,
            "rows_out": 0,
            "truncated": 0,
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned item DataFrame without modifying the input.

        Args:
            df: Source records containing the configured text columns.

        Returns:
            A DataFrame with item_id, title, description, category, year, and
            text columns when available.

        Raises:
            ValueError: If a configured text column other than category is
                missing from the source DataFrame.
        """
        dataframe = df.copy()
        self.report_ = {
            "rows_in": len(dataframe),
            "dropped_short": 0,
            "dropped_duplicates": 0,
            "rows_out": 0,
            "truncated": 0,
        }

        self._clean_text_columns(dataframe)
        self._ensure_category(dataframe)
        dataframe["text"] = self._build_text(dataframe)
        dataframe, dropped_short = self._drop_short_rows(dataframe)
        dataframe, dropped_duplicates = self._drop_duplicates(dataframe)
        dataframe, truncated = self._truncate_text(dataframe)
        dataframe["category"] = self._first_genre(dataframe["category"])

        dataframe = dataframe.reset_index(drop=True)
        dataframe["item_id"] = range(len(dataframe))
        output_columns = [
            "item_id",
            "title",
            "description",
            "category",
            "year",
            "text",
        ]
        output_columns = [
            column for column in output_columns if column in dataframe.columns
        ]
        dataframe = dataframe.loc[:, output_columns]

        self.report_.update(
            {
                "dropped_short": dropped_short,
                "dropped_duplicates": dropped_duplicates,
                "rows_out": len(dataframe),
                "truncated": truncated,
            }
        )
        logger.info("Text preprocessing completed: %s", self.report_)
        return dataframe

    def _clean_text_columns(self, dataframe: pd.DataFrame) -> None:
        """Clean configured text columns in place on the working copy.

        Args:
            dataframe: Working DataFrame copied by transform.

        Raises:
            ValueError: If a configured column other than category is absent.
        """
        missing_columns = [
            column
            for column in self._text_columns
            if column != "category" and column not in dataframe.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing text columns: {missing_columns}")

        for column in self._text_columns:
            if column in dataframe.columns:
                dataframe[column] = self._clean_series(dataframe[column])

    @staticmethod
    def _clean_series(values: pd.Series) -> pd.Series:
        """Unescape and normalize text values using pandas string operations.

        Args:
            values: Text values to clean.

        Returns:
            Cleaned pandas string Series with missing values represented as
            empty strings.
        """
        cleaned = values.fillna("").astype("string").map(html.unescape)
        cleaned = cleaned.astype("string")
        cleaned = cleaned.str.replace(r"<[^>]*>", " ", regex=True)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
        return cleaned

    @staticmethod
    def _ensure_category(dataframe: pd.DataFrame) -> None:
        """Create category if needed and replace empty values with unknown.

        Args:
            dataframe: Working DataFrame to update.
        """
        if "category" not in dataframe.columns:
            dataframe["category"] = "unknown"
            return

        category = dataframe["category"].fillna("").astype("string").str.strip()
        dataframe["category"] = category.mask(category.eq(""), "unknown")

    def _build_text(self, dataframe: pd.DataFrame) -> pd.Series:
        """Join non-empty configured columns in their declared order.

        Args:
            dataframe: Working DataFrame containing cleaned text columns.

        Returns:
            A string Series joined with period-space separators.
        """
        text = pd.Series("", index=dataframe.index, dtype="string")
        for column in self._text_columns:
            if column not in dataframe.columns:
                continue
            part = dataframe[column].fillna("").astype("string").str.strip()
            has_part = part.ne("")
            separator = pd.Series(". ", index=dataframe.index, dtype="string")
            separator = separator.where(text.ne("") & has_part, "")
            text = text + separator + part.where(has_part, "")
        return text

    def _drop_short_rows(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Remove rows whose assembled text is shorter than min_chars.

        Args:
            dataframe: Working DataFrame with its assembled text column.

        Returns:
            The retained rows and number of removed rows.
        """
        short_mask = dataframe["text"].str.len().lt(self._min_chars)
        retained = dataframe.loc[~short_mask].copy()
        return retained, int(short_mask.sum())

    @staticmethod
    def _drop_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop repeated lowercase title-description pairs, keeping the first.

        Args:
            dataframe: Working DataFrame after short rows were removed.

        Returns:
            The deduplicated DataFrame and number of removed duplicate rows.
        """
        key_columns = {"title", "description"}
        if not key_columns.issubset(dataframe.columns):
            return dataframe, 0

        duplicate_keys = pd.DataFrame(
            {
                "title": dataframe["title"].fillna("").astype("string").str.lower(),
                "description": (
                    dataframe["description"].fillna("").astype("string").str.lower()
                ),
            },
            index=dataframe.index,
        )
        duplicate_mask = duplicate_keys.duplicated(keep="first")
        retained = dataframe.loc[~duplicate_mask].copy()
        return retained, int(duplicate_mask.sum())

    def _truncate_text(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Truncate text at the last complete word within max_chars.

        Args:
            dataframe: Working DataFrame after filtering and deduplication.

        Returns:
            A copied DataFrame with truncated text and the number of affected
            rows.
        """
        text = dataframe["text"]
        over_limit = text.str.len().gt(self._max_chars)
        if not over_limit.any():
            return dataframe, 0

        prefix = text.str.slice(0, self._max_chars)
        last_complete_word = prefix.str.rsplit(" ", n=1).str[0].str.rstrip()
        boundary_after_prefix = text.str.slice(self._max_chars, self._max_chars + 1).eq(
            " "
        )
        has_boundary = prefix.str.contains(" ", regex=False) | boundary_after_prefix
        truncated_text = last_complete_word.where(
            ~boundary_after_prefix, prefix.str.rstrip()
        ).where(has_boundary, "")

        result = dataframe.copy()
        result["text"] = text.where(~over_limit, truncated_text)
        return result, int(over_limit.sum())

    @staticmethod
    def _first_genre(categories: pd.Series) -> pd.Series:
        """Keep the first comma-separated genre and fill empty values.

        Args:
            categories: Full category values, possibly containing genres.

        Returns:
            A string Series containing the first genre or unknown.
        """
        first_genres = (
            categories.fillna("")
            .astype("string")
            .str.split(",", n=1)
            .str[0]
            .str.strip()
        )
        return first_genres.mask(first_genres.eq(""), "unknown")

    @staticmethod
    def save(df: pd.DataFrame, path: str | Path) -> None:
        """Save a DataFrame as UTF-8 CSV without its index.

        Args:
            df: DataFrame to save.
            path: Destination path for the CSV file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, encoding="utf-8", index=False)
        logger.info("Saved processed data to %s", output_path)
