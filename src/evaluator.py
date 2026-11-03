"""Evaluate recommendation relevance and latency."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.engine import RecommendationEngine

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute ranking metrics and response latency for an evaluation set."""

    def __init__(
        self,
        engine: RecommendationEngine,
        eval_df: pd.DataFrame,
    ) -> None:
        """Initialize the evaluator with queries and relevant item identifiers.

        Args:
            engine: Recommendation engine used to rank catalog items.
            eval_df: DataFrame with query and pipe-separated relevant_ids columns.

        Raises:
            ValueError: If eval_df is empty, required columns are absent, or
                any query has no valid relevant item identifiers.
        """
        required_columns = {"query", "relevant_ids"}
        missing_columns = required_columns.difference(eval_df.columns)
        if missing_columns:
            raise ValueError(f"Missing evaluation columns: {sorted(missing_columns)}")
        if eval_df.empty:
            raise ValueError("Evaluation DataFrame must not be empty.")

        self._engine = engine
        self._queries = eval_df["query"].tolist()
        self._relevant_ids = [
            self._parse_relevant_ids(value, row_number)
            for row_number, value in enumerate(eval_df["relevant_ids"], start=1)
        ]
        self._results_cache: dict[int, list[tuple[list[int], float]]] = {}
        logger.info("Evaluator initialized with %d queries.", len(self._queries))

    @staticmethod
    def _parse_relevant_ids(value: object, row_number: int) -> set[int]:
        """Parse one pipe-separated relevant identifier cell.

        Args:
            value: Cell value containing integer IDs separated by pipes.
            row_number: One-based row number for error messages.

        Returns:
            A set of parsed item identifiers.

        Raises:
            ValueError: If the cell has no identifiers or contains invalid IDs.
        """
        if pd.isna(value):
            tokens: list[str] = []
        else:
            tokens = [token.strip() for token in str(value).split("|")]
            tokens = [token for token in tokens if token]

        if not tokens:
            raise ValueError(f"Evaluation row {row_number} has no relevant IDs.")

        try:
            return {int(token) for token in tokens}
        except ValueError as error:
            raise ValueError(
                f"Evaluation row {row_number} contains a non-integer relevant ID."
            ) from error

    @staticmethod
    def _validate_k(k: int) -> None:
        """Raise ValueError unless k is a positive integer."""
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
            raise ValueError("k must be a positive integer.")
        if k < 1:
            raise ValueError("k must be at least 1.")

    def _run(self, max_k: int) -> list[tuple[list[int], float]]:
        """Run and cache one ranking and elapsed time per query and depth.

        Args:
            max_k: Maximum number of recommendations requested per query.

        Returns:
            Per-query tuples of ranked item IDs and elapsed seconds.
        """
        self._validate_k(max_k)
        if max_k in self._results_cache:
            return self._results_cache[max_k]

        query_results: list[tuple[list[int], float]] = []
        for query in self._queries:
            start_time = time.perf_counter()
            recommendations = self._engine.recommend(query, k=max_k)
            elapsed_seconds = time.perf_counter() - start_time
            ranked_ids = [int(item_id) for item_id in recommendations["item_id"]]
            query_results.append((ranked_ids, elapsed_seconds))

        self._results_cache[max_k] = query_results
        logger.info("Evaluated %d queries at k=%d.", len(query_results), max_k)
        return query_results

    def recall_at_k(self, k: int) -> float:
        """Return mean Recall@k: mean(|relevant ∩ top_k| / |relevant|)."""
        self._validate_k(k)
        results = self._run(k)
        recalls = [
            len(relevant.intersection(ranked_ids[:k])) / len(relevant)
            for relevant, (ranked_ids, _) in zip(self._relevant_ids, results)
        ]
        return float(np.mean(recalls))

    def mrr(self) -> float:
        """Return MRR: mean(1 / first relevant rank in top-10), or 0 if absent."""
        results = self._run(10)
        reciprocal_ranks: list[float] = []
        for relevant, (ranked_ids, _) in zip(self._relevant_ids, results):
            first_hit_rank = next(
                (
                    rank
                    for rank, item_id in enumerate(ranked_ids[:10], start=1)
                    if item_id in relevant
                ),
                None,
            )
            reciprocal_ranks.append(
                0.0 if first_hit_rank is None else 1.0 / first_hit_rank
            )
        return float(np.mean(reciprocal_ranks))

    def hit_rate_at_k(self, k: int) -> float:
        """Return HitRate@k: mean(1 if top_k intersects relevant, otherwise 0)."""
        self._validate_k(k)
        results = self._run(k)
        hits = [
            bool(relevant.intersection(ranked_ids[:k]))
            for relevant, (ranked_ids, _) in zip(self._relevant_ids, results)
        ]
        return float(np.mean(hits))

    def avg_latency_ms(self) -> float:
        """Return mean query latency: 1000 * mean(per-query elapsed seconds)."""
        results = self._run(10)
        return float(np.mean([elapsed for _, elapsed in results]) * 1000)

    def run_all(self, ks: Sequence[int] = (1, 3, 10)) -> dict[str, int | float]:
        """Return rounded aggregate metrics for the requested cutoff values.

        Recall@k is mean(|relevant ∩ top_k| / |relevant|); HitRate@k is the
        fraction of queries whose top-k contains a relevant item; MRR is the
        mean reciprocal first-hit rank in top-10, or zero when absent;
        latency is 1000 times the mean per-query elapsed seconds.

        Args:
            ks: Positive ranking cutoffs for Recall@k and HitRate@k.

        Returns:
            A dictionary with query count, relevance metrics, MRR, and latency.
        """
        metrics: dict[str, int | float] = {"n_queries": len(self._queries)}
        for k in ks:
            self._validate_k(k)
            metrics[f"recall@{k}"] = round(self.recall_at_k(k), 4)
            metrics[f"hit_rate@{k}"] = round(self.hit_rate_at_k(k), 4)
        metrics["mrr"] = round(self.mrr(), 4)
        metrics["avg_latency_ms"] = round(self.avg_latency_ms(), 4)
        return metrics

    def per_query(self, k: int = 10) -> pd.DataFrame:
        """Return per-query hits, Recall@cutoff, and latency details.

        Per-query Recall@j is |relevant ∩ top_j| / |relevant|; first_hit_rank
        is the rank of the first relevant ID in top-k or NaN if absent; latency
        is 1000 times the measured elapsed seconds.

        Args:
            k: Cutoff used to identify first_hit_rank and include recall@k.

        Returns:
            A DataFrame with query-level metric columns.
        """
        self._validate_k(k)
        max_k = max(k, 10)
        results = self._run(max_k)
        records: list[dict[str, object]] = []

        for query, relevant, (ranked_ids, elapsed_seconds) in zip(
            self._queries, self._relevant_ids, results
        ):
            top_k_ids = ranked_ids[:k]
            first_hit_rank = next(
                (
                    rank
                    for rank, item_id in enumerate(top_k_ids, start=1)
                    if item_id in relevant
                ),
                np.nan,
            )
            record: dict[str, object] = {
                "query": query,
                "n_relevant": len(relevant),
                "first_hit_rank": first_hit_rank,
                "latency_ms": elapsed_seconds * 1000,
            }
            for cutoff in dict.fromkeys((1, 3, 10, k)):
                record[f"recall@{cutoff}"] = len(
                    relevant.intersection(ranked_ids[:cutoff])
                ) / len(relevant)
            records.append(record)

        columns = [
            "query",
            "n_relevant",
            "first_hit_rank",
            *[f"recall@{cutoff}" for cutoff in dict.fromkeys((1, 3, 10, k))],
            "latency_ms",
        ]
        return pd.DataFrame.from_records(records, columns=columns)
