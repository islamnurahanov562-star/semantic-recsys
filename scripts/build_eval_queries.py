"""Validate evaluation-template rows and build the canonical query dataset."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = Path("data/processed/eval_template.csv")
_DEFAULT_ITEMS = Path("data/processed/items.csv")
_DEFAULT_OUTPUT = Path("data/processed/eval_queries.csv")
_TEMPLATE_COLUMNS = {"item_id", "extra_relevant_ids", "query"}
_ITEM_COLUMNS = {"item_id", "description"}
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_LEAKAGE_WINDOW = 5


def _tokenize(text: object) -> list[str]:
    """Return case-insensitive word tokens with punctuation removed.

    Args:
        text: Value to tokenize.

    Returns:
        Unicode word tokens in their original order.
    """
    if pd.isna(text):
        return []
    return _WORD_PATTERN.findall(str(text).casefold())


def _parse_id(value: object) -> int | None:
    """Parse an integer item ID without accepting fractional values.

    Args:
        value: Source cell value.

    Returns:
        Parsed integer ID, or None when the value is not an integer.
    """
    if pd.isna(value):
        return None
    token = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", token):
        return None
    return int(token)


def _contains_description_phrase(query_tokens: list[str], description: object) -> bool:
    """Check whether query and description share a consecutive five-word phrase.

    Args:
        query_tokens: Normalized query tokens.
        description: Source description for the query's item.

    Returns:
        True when at least five consecutive query words also occur consecutively
        in the description.
    """
    if len(query_tokens) < _LEAKAGE_WINDOW:
        return False
    description_tokens = _tokenize(description)
    description_windows = {
        tuple(description_tokens[index : index + _LEAKAGE_WINDOW])
        for index in range(len(description_tokens) - _LEAKAGE_WINDOW + 1)
    }
    return any(
        tuple(query_tokens[index : index + _LEAKAGE_WINDOW]) in description_windows
        for index in range(len(query_tokens) - _LEAKAGE_WINDOW + 1)
    )


class EvalQueryValidator:
    """Collect structural, relevance-ID, duplicate, and leakage problems."""

    def validate(self, df: pd.DataFrame, items: pd.DataFrame) -> list[str]:
        """Return every validation issue as a row/problem pair.

        Args:
            df: Non-empty evaluation-template rows to validate.
            items: Processed items used to verify IDs and source descriptions.

        Returns:
            Issue strings formatted as ``row<TAB>problem``. An em dash is used
            for file-level issues that do not belong to one row.
        """
        issues: list[str] = []
        missing_template_columns = _TEMPLATE_COLUMNS.difference(df.columns)
        missing_item_columns = _ITEM_COLUMNS.difference(items.columns)
        for column in sorted(missing_template_columns):
            issues.append(f"—\tTemplate is missing required column '{column}'.")
        for column in sorted(missing_item_columns):
            issues.append(f"—\tItems file is missing required column '{column}'.")

        item_ids: set[int] = set()
        descriptions_by_id: dict[int, object] = {}
        if "item_id" in items.columns:
            item_columns = ["item_id"]
            if "description" in items.columns:
                item_columns.append("description")
            for item_values in items[item_columns].itertuples(index=False, name=None):
                item_id = _parse_id(item_values[0])
                if item_id is not None:
                    item_ids.add(item_id)
                    if len(item_values) > 1:
                        descriptions_by_id[item_id] = item_values[1]

        query_rows: dict[str, list[int]] = defaultdict(list)
        for position, (index, row) in enumerate(df.iterrows()):
            row_number = int(index) + 2 if isinstance(index, Integral) else position + 2
            if "query" not in df.columns:
                continue

            query_value = row["query"]
            query_tokens = _tokenize(query_value)
            if not query_tokens:
                continue

            normalized_query = str(query_value).strip().casefold()
            query_rows[normalized_query].append(row_number)
            if len(query_tokens) < 3:
                issues.append(f"{row_number}\tQuery must contain at least three words.")

            base_item_id: int | None = None
            if "item_id" in df.columns:
                base_item_id = _parse_id(row["item_id"])
                if base_item_id is None:
                    issues.append(f"{row_number}\tSource item_id must be an integer.")
                elif base_item_id not in item_ids:
                    issues.append(
                        f"{row_number}\tSource item_id {base_item_id} does not exist."
                    )

            if "extra_relevant_ids" in df.columns:
                extra_value = row["extra_relevant_ids"]
                extra_tokens = (
                    []
                    if pd.isna(extra_value)
                    else [token.strip() for token in str(extra_value).split("|")]
                )
                for token in extra_tokens:
                    if not token:
                        continue
                    extra_item_id = _parse_id(token)
                    if extra_item_id is None:
                        issues.append(
                            f"{row_number}\tExtra relevant ID '{token}' "
                            "is not an integer."
                        )
                    elif extra_item_id not in item_ids:
                        issues.append(
                            f"{row_number}\tExtra relevant ID {extra_item_id} "
                            "does not exist."
                        )

            if (
                base_item_id is not None
                and base_item_id in descriptions_by_id
                and _contains_description_phrase(
                    query_tokens, descriptions_by_id[base_item_id]
                )
            ):
                issues.append(
                    f"{row_number}\tQuery copies at least five consecutive words "
                    "from its source description."
                )

        for duplicate_rows in query_rows.values():
            if len(duplicate_rows) > 1:
                for row_number in duplicate_rows:
                    issues.append(
                        f"{row_number}\tQuery is duplicated without regard to case."
                    )

        if "query" in df.columns and not query_rows:
            issues.append("—\tNo non-empty queries were found in the template.")
        return issues


def _build_output(df: pd.DataFrame) -> pd.DataFrame:
    """Combine source and extra relevant IDs into sorted unique ID strings.

    Args:
        df: Validated non-empty template rows.

    Returns:
        DataFrame with the canonical query and relevant_ids columns.
    """
    output_rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        source_item_id = _parse_id(row["item_id"])
        relevant_ids: set[int] = set()
        if source_item_id is not None:
            relevant_ids.add(source_item_id)
        extra_value = row["extra_relevant_ids"]
        if not pd.isna(extra_value):
            for token in str(extra_value).split("|"):
                if not token.strip():
                    continue
                extra_item_id = _parse_id(token.strip())
                if extra_item_id is not None:
                    relevant_ids.add(extra_item_id)
        output_rows.append(
            {
                "query": str(row["query"]).strip(),
                "relevant_ids": "|".join(
                    str(item_id) for item_id in sorted(relevant_ids)
                ),
            }
        )
    return pd.DataFrame(output_rows, columns=["query", "relevant_ids"])


def _format_issues(issues: list[str]) -> str:
    """Render validator messages as a two-column text table.

    Args:
        issues: Row/problem messages from EvalQueryValidator.

    Returns:
        A formatted table with row and problem headings.
    """
    rows = [issue.split("\t", maxsplit=1) for issue in issues]
    table = pd.DataFrame(rows, columns=["строка", "проблема"])
    return table.to_string(index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a filled evaluation template and write canonical query data.

    Args:
        argv: Optional command-line arguments, used for testing.

    Returns:
        Zero on success, or one when validation or file operations fail.
    """
    parser = argparse.ArgumentParser(
        description="Validate an evaluation template and build eval_queries.csv."
    )
    parser.add_argument("--template", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument("--items", type=Path, default=_DEFAULT_ITEMS)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        template = pd.read_csv(args.template)
        items = pd.read_csv(args.items)
    except (OSError, ValueError) as error:
        logger.error("Could not read evaluation inputs: %s", error)
        return 1

    if "query" in template.columns:
        queries = template["query"].fillna("").astype("string").str.strip()
        non_empty_mask = queries.ne("")
        template = template.loc[non_empty_mask].copy()
        template["query"] = queries.loc[non_empty_mask]

    issues = EvalQueryValidator().validate(template, items)
    if issues:
        print(_format_issues(issues))
        logger.error("Evaluation template has %d validation problems.", len(issues))
        return 1

    eval_queries = _build_output(template)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        eval_queries.to_csv(args.out, index=False, encoding="utf-8")
    except OSError as error:
        logger.error("Could not write evaluation queries: %s", error)
        return 1

    relevant_counts = eval_queries["relevant_ids"].str.split("|").str.len()
    mean_relevant_count = f"{relevant_counts.mean():.2f}"
    multiple_relevant_count = int(relevant_counts.gt(1).sum())
    print(f"Число запросов: {len(eval_queries)}")
    print("Среднее relevant id:", mean_relevant_count)
    print("Запросов с >1 relevant id:", multiple_relevant_count)
    logger.info("Saved evaluation queries to %s.", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
