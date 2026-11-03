"""Create a stratified, unlabeled evaluation-query template."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ITEMS_PATH = Path("data/processed/items.csv")
_DEFAULT_OUTPUT = Path("data/processed/eval_template.csv")
_OUTPUT_COLUMNS = [
    "item_id",
    "title",
    "category",
    "description_preview",
    "query",
    "extra_relevant_ids",
]
_REQUIRED_ITEM_COLUMNS = {"item_id", "title", "category", "description"}


def _apportion(counts: pd.Series, seats: int, categories: list[str]) -> dict[str, int]:
    """Distribute integer seats proportionally using largest remainders.

    Args:
        counts: Population size for each category.
        seats: Number of samples to allocate.
        categories: Categories in stable source order.

    Returns:
        Integer allocations summing to seats.
    """
    total = int(counts.sum())
    quotas = {
        category: seats * int(counts[category]) / total for category in categories
    }
    allocations = {category: int(np.floor(quotas[category])) for category in categories}
    remaining = seats - sum(allocations.values())
    order = sorted(
        categories,
        key=lambda category: (
            -(quotas[category] - allocations[category]),
            categories.index(category),
        ),
    )
    for category in order[:remaining]:
        allocations[category] += 1
    return allocations


def _allocate_counts(counts: pd.Series, sample_size: int) -> dict[str, int]:
    """Allocate a stratified sample, giving each category one item when possible.

    Args:
        counts: Population size for each category in source order.
        sample_size: Total number of rows to select.

    Returns:
        Number of rows to sample from each category.
    """
    categories = [str(category) for category in counts.index]
    if sample_size < len(categories):
        return _apportion(counts, sample_size, categories)

    allocations = {category: 0 for category in categories}
    remaining_categories = categories.copy()
    remaining_seats = sample_size

    while remaining_categories:
        total_population = int(counts.loc[remaining_categories].sum())
        quotas = {
            category: remaining_seats * int(counts[category]) / total_population
            for category in remaining_categories
        }
        undersized = [
            category for category in remaining_categories if quotas[category] < 1
        ]
        if not undersized:
            allocations.update(
                _apportion(
                    counts.loc[remaining_categories],
                    remaining_seats,
                    remaining_categories,
                )
            )
            break

        for category in undersized:
            allocations[category] = 1
        remaining_seats -= len(undersized)
        remaining_categories = [
            category for category in remaining_categories if category not in undersized
        ]

    return allocations


def _normalize_categories(categories: pd.Series) -> pd.Series:
    """Normalize missing and whitespace-only categories to unknown.

    Args:
        categories: Category values from the items table.

    Returns:
        Normalized string category values.
    """
    normalized = categories.astype("string").fillna("unknown").str.strip()
    return normalized.mask(normalized.eq(""), "unknown")


def make_template(items: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Select a reproducible category-stratified set of items.

    Args:
        items: Processed item records.
        n: Number of rows to select.
        seed: Random seed for reproducible category and row selection.

    Returns:
        An unlabeled evaluation template with the required columns.

    Raises:
        ValueError: If required input columns are missing or n is out of range.
    """
    missing_columns = _REQUIRED_ITEM_COLUMNS.difference(items.columns)
    if missing_columns:
        raise ValueError(f"Items file is missing columns: {sorted(missing_columns)}")
    if n < 1:
        raise ValueError("--n must be at least one.")
    if n > len(items):
        raise ValueError(
            f"Requested {n} rows, but only {len(items)} items are available."
        )

    normalized_items = items.copy()
    normalized_items["category"] = _normalize_categories(normalized_items["category"])
    category_counts = normalized_items["category"].value_counts(sort=False)
    allocations = _allocate_counts(category_counts, n)
    random_generator = np.random.default_rng(seed)
    category_values = normalized_items["category"].to_numpy(dtype=str)
    selected_positions: list[int] = []

    for category, count in allocations.items():
        if count == 0:
            continue
        positions = np.flatnonzero(category_values == category)
        selected_positions.extend(
            random_generator.choice(positions, size=count, replace=False).tolist()
        )

    random_generator.shuffle(selected_positions)
    sampled_items = normalized_items.iloc[selected_positions]
    template = sampled_items.loc[:, ["item_id", "title", "category"]].copy()
    template["description_preview"] = (
        sampled_items["description"].fillna("").astype("string").str.slice(0, 300)
    )
    template["query"] = ""
    template["extra_relevant_ids"] = ""
    return template.loc[:, _OUTPUT_COLUMNS].reset_index(drop=True)


def write_template(
    template: pd.DataFrame, output_path: str | Path, force: bool
) -> None:
    """Write the CSV without overwriting an existing file unless forced.

    Args:
        template: Unlabeled evaluation rows to write.
        output_path: Destination CSV path.
        force: Whether to replace an existing destination.

    Raises:
        FileExistsError: If the destination exists and force is false.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if force:
        template.to_csv(path, index=False, encoding="utf-8")
        return

    with path.open("x", encoding="utf-8", newline="") as output_file:
        template.to_csv(output_file, index=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Load items, create a stratified template, and report category counts.

    Args:
        argv: Optional command-line arguments, used for testing.

    Returns:
        Zero on success, or one if input or output validation fails.
    """
    parser = argparse.ArgumentParser(
        description="Create a stratified evaluation-query template."
    )
    parser.add_argument("--n", type=int, default=60, help="Number of items to select.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output CSV path (default: data/processed/eval_template.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        items = pd.read_csv(_ITEMS_PATH)
        template = make_template(items, args.n, args.seed)
        write_template(template, args.out, args.force)
    except (FileExistsError, OSError, ValueError) as error:
        logger.error("Could not create evaluation template: %s", error)
        return 1

    print("Выбрано записей по категориям:")
    categories = _normalize_categories(items["category"]).drop_duplicates()
    for category in categories:
        count = int(template["category"].eq(category).sum())
        print(f"{category}: {count}")
    logger.info("Saved %d evaluation rows to %s.", len(template), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
