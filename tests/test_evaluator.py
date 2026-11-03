"""Tests for recommendation evaluation metrics."""

import pandas as pd
import pytest

from src.evaluator import Evaluator


class FakeEngine:
    """Return fixed ranked item IDs and record calls by query."""

    def __init__(self, rankings: dict[str, list[int]]) -> None:
        """Store ranked IDs and initialize per-query call counters."""
        self._rankings = rankings
        self.calls_by_query = dict.fromkeys(rankings, 0)

    def recommend(self, query: str, k: int) -> pd.DataFrame:
        """Return the configured ranking truncated to k items."""
        self.calls_by_query[query] += 1
        return pd.DataFrame({"item_id": self._rankings[query][:k]})


def _make_evaluator() -> tuple[FakeEngine, Evaluator]:
    """Build the three-query example with manually verifiable rankings."""
    engine = FakeEngine(
        {
            "q1": [1, 2, 3],
            "q2": [4, 6, 5],
            "q3": [7, 8, 9],
        }
    )
    eval_df = pd.DataFrame(
        {
            "query": ["q1", "q2", "q3"],
            "relevant_ids": ["1 | 3", "5", "10"],
        }
    )
    return engine, Evaluator(engine, eval_df)


def test_recall_and_mrr_match_manual_calculations() -> None:
    """Recall and MRR match hand-calculated values for three queries."""
    _, evaluator = _make_evaluator()

    # Recall@1: ((1/2) + 0 + 0) / 3 = 1/6.
    assert evaluator.recall_at_k(1) == pytest.approx(0.1666666667)
    # Recall@3: (2/2 + 1/1 + 0) / 3 = 2/3.
    assert evaluator.recall_at_k(3) == pytest.approx(0.6666666667)
    # MRR: (1/1 + 1/3 + 0) / 3 = 4/9.
    assert evaluator.mrr() == pytest.approx(0.4444444444)


def test_hit_rate_matches_manual_calculations() -> None:
    """Hit rate counts queries with at least one relevant result."""
    _, evaluator = _make_evaluator()

    # HitRate@1: q1 hits; q2 and q3 miss, so 1/3.
    assert evaluator.hit_rate_at_k(1) == pytest.approx(0.3333333333)
    # HitRate@3: q1 and q2 hit; q3 misses, so 2/3.
    assert evaluator.hit_rate_at_k(3) == pytest.approx(0.6666666667)


def test_unfound_relevant_items_have_zero_mrr_contribution_and_nan_rank() -> None:
    """A query with no hit contributes zero MRR and has no first-hit rank."""
    _, evaluator = _make_evaluator()

    per_query = evaluator.per_query(k=3)

    # q3 has no relevant item in [7, 8, 9], so its reciprocal-rank contribution is 0.
    assert evaluator.mrr() == pytest.approx(0.4444444444)
    assert pd.isna(per_query.loc[2, "first_hit_rank"])


def test_multiple_relevant_ids_are_counted_in_recall() -> None:
    """Recall divides hits by all relevant IDs, not only the first one."""
    _, evaluator = _make_evaluator()

    # q1 has relevant IDs {1, 3}; top-1 finds one (1/2), top-3 finds both (2/2).
    assert evaluator.recall_at_k(1) == pytest.approx(0.1666666667)
    per_query = evaluator.per_query(k=3)
    assert per_query.loc[0, "recall@1"] == pytest.approx(0.5)
    assert per_query.loc[0, "recall@3"] == pytest.approx(1.0)


def test_invalid_evaluation_data_and_zero_k_raise_value_error() -> None:
    """Required columns, relevant IDs, and positive cutoffs are validated."""
    engine = FakeEngine({"q1": [1, 2, 3]})

    with pytest.raises(ValueError, match="query"):
        Evaluator(engine, pd.DataFrame({"relevant_ids": ["1"]}))
    with pytest.raises(ValueError, match="relevant_ids"):
        Evaluator(engine, pd.DataFrame({"query": ["q1"]}))
    with pytest.raises(ValueError, match="empty"):
        Evaluator(
            engine,
            pd.DataFrame(columns=["query", "relevant_ids"]),
        )
    with pytest.raises(ValueError, match="no relevant IDs"):
        Evaluator(
            engine,
            pd.DataFrame({"query": ["q1"], "relevant_ids": [" |  | "]}),
        )

    _, evaluator = _make_evaluator()
    with pytest.raises(ValueError, match="at least 1"):
        evaluator.recall_at_k(0)


def test_same_depth_metrics_call_engine_once_per_query() -> None:
    """Metrics sharing max_k reuse the cached rankings for each query."""
    engine, evaluator = _make_evaluator()

    evaluator.recall_at_k(3)
    evaluator.hit_rate_at_k(3)
    evaluator.recall_at_k(3)

    assert engine.calls_by_query == {"q1": 1, "q2": 1, "q3": 1}


def test_run_all_returns_expected_keys_and_nonnegative_latency() -> None:
    """The aggregate report includes all default metrics and a valid latency."""
    _, evaluator = _make_evaluator()

    metrics = evaluator.run_all()

    assert set(metrics) == {
        "n_queries",
        "recall@1",
        "hit_rate@1",
        "recall@3",
        "hit_rate@3",
        "recall@10",
        "hit_rate@10",
        "mrr",
        "avg_latency_ms",
    }
    assert metrics["n_queries"] == 3
    assert metrics["recall@1"] == 0.1667
    assert metrics["recall@3"] == 0.6667
    assert metrics["mrr"] == 0.4444
    assert metrics["avg_latency_ms"] >= 0
