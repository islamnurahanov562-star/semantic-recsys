"""Generate a human-review draft for the lowest-recall evaluation queries."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.factory import build_engine

if TYPE_CHECKING:
    from src.engine import RecommendationEngine

logger = logging.getLogger(__name__)

_INDEX_TYPES = ("numpy", "faiss-flat", "faiss-ivf")
_DEFAULT_INDEX = "faiss-flat"
_DEFAULT_WORST = 10
_DEFAULT_PER_QUERY = Path(f"results/per_query_{_DEFAULT_INDEX}.csv")
_DEFAULT_METRICS = Path(f"results/metrics_{_DEFAULT_INDEX}.json")
_DEFAULT_EVAL = Path("data/processed/eval_queries.csv")
_DEFAULT_ITEMS = Path("data/processed/items.csv")
_DEFAULT_OUTPUT = Path("docs/ERROR_ANALYSIS.md")


def _clean_markdown(value: object) -> str:
    """Make a scalar safe to place in a Markdown table or list.

    Args:
        value: Value to render.

    Returns:
        Text with newlines and Markdown table separators escaped.
    """
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _select_worst(per_query: pd.DataFrame, worst: int) -> pd.DataFrame:
    """Select lowest-recall queries, prioritizing those with no hit on ties.

    Args:
        per_query: Per-query metrics containing recall@10 and first_hit_rank.
        worst: Maximum number of rows to return.

    Returns:
        Stable-ranked subset ordered by recall ascending, then no-hit first.
    """
    required_columns = {"query", "recall@10", "first_hit_rank"}
    missing_columns = required_columns.difference(per_query.columns)
    if missing_columns:
        raise ValueError(
            f"Per-query file is missing columns: {sorted(missing_columns)}"
        )
    ordered = per_query.assign(_no_hit=per_query["first_hit_rank"].isna())
    return ordered.sort_values(
        ["recall@10", "_no_hit"],
        ascending=[True, False],
        kind="stable",
    ).head(worst)


def _relevant_items(
    relevant_ids: str,
    items_by_id: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Resolve pipe-separated item IDs to title and category pairs.

    Args:
        relevant_ids: Pipe-separated item identifiers.
        items_by_id: Catalog indexed by integer item_id.

    Returns:
        Relevant titles and categories in the listed ID order.
    """
    records: list[tuple[str, str]] = []
    for token in relevant_ids.split("|"):
        item_id = int(token.strip())
        if item_id not in items_by_id.index:
            raise ValueError(f"Relevant item_id {item_id} does not exist in items.csv.")
        item = items_by_id.loc[item_id]
        records.append((str(item["title"]), str(item["category"])))
    return records


def _append_query_report(
    lines: list[str],
    row: pd.Series,
    relevant_ids: str,
    items_by_id: pd.DataFrame,
    engine: RecommendationEngine,
) -> None:
    """Append one query's relevance, recommendations, and manual label slot.

    Args:
        lines: Markdown output lines to extend.
        row: Per-query metrics for the selected query.
        relevant_ids: Source and additional relevant IDs.
        items_by_id: Catalog indexed by item_id.
        engine: Built recommendation engine used for top-three results.
    """
    query = str(row["query"])
    recommendations = engine.recommend(query, k=3)
    first_hit_rank = row["first_hit_rank"]
    rank_display = "not found" if pd.isna(first_hit_rank) else str(int(first_hit_rank))

    lines.extend(
        [
            f"### {_clean_markdown(query)}",
            "",
            f"- Recall@10: {float(row['recall@10']):.4f}",
            f"- First relevant rank: {rank_display}",
            "- Relevant titles:",
        ]
    )
    for title, category in _relevant_items(relevant_ids, items_by_id):
        lines.append(f"  - {_clean_markdown(title)} ({_clean_markdown(category)})")

    lines.append("- Top-3 recommendations:")
    if recommendations.empty:
        lines.append("  - No results returned.")
    else:
        for result in recommendations.itertuples(index=False):
            lines.append(
                f"  - {int(result.rank)}. {_clean_markdown(result.title)} "
                f"({_clean_markdown(result.category)}), score={float(result.score):.4f}"
            )
    lines.extend(
        [
            "",
            "Failure type: TODO (data issue | ambiguous query | "
            "model limitation | other)",
            "",
        ]
    )


def _render_report(
    index: str,
    metrics_payload: dict[str, object],
    worst_rows: pd.DataFrame,
    eval_df: pd.DataFrame,
    items: pd.DataFrame,
    engine: RecommendationEngine,
) -> str:
    """Render aggregate metrics and selected query examples as Markdown.

    Args:
        index: Index name used for the evaluation.
        metrics_payload: Aggregate metrics and metadata JSON payload.
        worst_rows: Selected per-query result rows.
        eval_df: Evaluation queries and relevance IDs.
        items: Catalog item data.
        engine: Built recommendation engine.

    Returns:
        Markdown draft requiring manual failure classification.
    """
    metrics = metrics_payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Metrics JSON must contain a 'metrics' object.")
    relevant_by_query = dict(zip(eval_df["query"], eval_df["relevant_ids"]))
    items_by_id = items.set_index("item_id")
    lines = [
        "# Error Analysis",
        "",
        f"- Model: {metrics_payload['model']}",
        f"- Index: {index}",
        f"- Items: {metrics_payload['n_items']}",
        f"- Queries: {metrics_payload['n_queries']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric_name, value in metrics.items():
        lines.append(f"| {_clean_markdown(metric_name)} | {value} |")
    lines.extend(["", "## Lowest-Recall Queries", ""])

    for _, row in worst_rows.iterrows():
        query = str(row["query"])
        if query not in relevant_by_query:
            raise ValueError(f"Query is missing from eval file: {query}")
        _append_query_report(
            lines,
            row,
            str(relevant_by_query[query]),
            items_by_id,
            engine,
        )

    lines.extend(
        [
            "## Manual Summary",
            "",
            "- Data issue: TODO",
            "- Ambiguous query: TODO",
            "- Model limitation: TODO",
            "- Other: TODO",
            "",
            "## Proposed Fixes",
            "",
            "1. TODO",
            "2. TODO",
            "3. TODO",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Create a human-review error-analysis report from saved R4-5 artifacts.

    Args:
        argv: Optional command-line arguments, used by tests.

    Returns:
        Zero on success, or one if input data or evaluation fails.
    """
    parser = argparse.ArgumentParser(description="Draft retrieval error analysis.")
    parser.add_argument("--index", choices=_INDEX_TYPES, default=_DEFAULT_INDEX)
    parser.add_argument("--worst", type=int, default=_DEFAULT_WORST)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.worst < 1:
        parser.error("--worst must be at least one")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    per_query_path = Path(f"results/per_query_{args.index}.csv")
    metrics_path = Path(f"results/metrics_{args.index}.json")

    try:
        per_query = pd.read_csv(per_query_path)
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics_payload = json.load(metrics_file)
        eval_df = pd.read_csv(
            _DEFAULT_EVAL,
            dtype={"query": "string", "relevant_ids": "string"},
        )
        items = pd.read_csv(_DEFAULT_ITEMS)
        items["item_id"] = pd.to_numeric(items["item_id"], errors="raise").astype(int)
        selected = _select_worst(per_query, args.worst)
        engine = build_engine(
            items_path=_DEFAULT_ITEMS,
            model_name=str(metrics_payload["model"]),
            index_type=args.index,
        )
        report = _render_report(
            args.index,
            metrics_payload,
            selected,
            eval_df,
            items,
            engine,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.error("Error analysis failed: %s", error)
        return 1

    logger.info("Saved error-analysis draft to %s.", args.out)
    logger.info("Selected %d queries for manual review.", len(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
