"""Load CSV datasets and validate their required columns."""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Read a CSV file, rename columns, and validate required data fields."""

    def __init__(
        self,
        path: str | Path,
        required_columns: Sequence[str],
        column_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize the loader configuration.

        Args:
            path: Path to the source CSV file.
            required_columns: Canonical column names required after renaming.
            column_map: Optional mapping from source names to canonical names.
        """
        self._path = Path(path)
        self._required_columns = tuple(required_columns)
        self._column_map = dict(column_map) if column_map is not None else None

    def load(self) -> pd.DataFrame:
        """Read, optionally rename, and validate the CSV data.

        Returns:
            The loaded and validated DataFrame.

        Raises:
            FileNotFoundError: If the configured CSV path does not exist.
            ValueError: If mapped source columns or required columns are invalid.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        dataframe = self._read_csv()
        if self._column_map is not None:
            missing_source_columns = [
                source_column
                for source_column in self._column_map
                if source_column not in dataframe.columns
            ]
            if missing_source_columns:
                raise ValueError(
                    "Columns listed in column_map are missing from the CSV: "
                    f"{missing_source_columns}"
                )
            dataframe = dataframe.rename(columns=self._column_map)

        self.validate(dataframe)
        return dataframe

    def validate(self, df: pd.DataFrame) -> None:
        """Validate required columns and log their missing-value percentages.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If the DataFrame is empty, columns are missing, or a
                required column contains no non-missing values.
        """
        if df.empty:
            raise ValueError("DataFrame is empty.")

        missing_columns = [
            column for column in self._required_columns if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        empty_columns = [
            column
            for column in self._required_columns
            if self._missing_mask(df[column]).all()
        ]
        if empty_columns:
            raise ValueError(f"Required columns are entirely empty: {empty_columns}")

        for column in self._required_columns:
            missing_percent = self._missing_mask(df[column]).mean() * 100
            logger.warning(
                "Column '%s' has %.2f%% missing values.", column, missing_percent
            )

    def _read_csv(self) -> pd.DataFrame:
        """Read the source CSV as UTF-8, falling back to latin-1 if needed.

        Returns:
            The parsed CSV data.
        """
        try:
            return pd.read_csv(self._path, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decoding failed for %s; retrying with latin-1 encoding.",
                self._path,
            )
            return pd.read_csv(self._path, encoding="latin-1")

    @staticmethod
    def _missing_mask(values: pd.Series) -> pd.Series:
        """Return a boolean mask for null, empty, and whitespace-only values.

        Args:
            values: Series to inspect.

        Returns:
            A boolean Series marking missing or blank values.
        """
        blank_values = values.astype("string").str.strip().eq("").fillna(True)
        return values.isna() | blank_values
