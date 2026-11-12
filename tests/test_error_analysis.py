"""Tests for selecting and rendering error-analysis drafts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import error_analysis


class FakeEngine:
    """Return fixture recommendations and record analyzed queries."""

    def __init__(self) -> None:
        """Initialize an empty query call log."""
        self.queries: list[str] = []

    def recommend(self, query: str, k: int = 3) -> pd.DataFrame:
        """Return up to three deterministic recommendations."""
        self.queries.append(query)
        return pd.DataFrame(
            {
                "rank": [1],
                "item_id": [2],
                "title": ["Suggested title"],
                "category": ["Drama"],
                "score": [0.75],
                "description": ["Suggested description"],
            }
        ).head(k)


def test_select_worst_prioritizes_no_hit_for_equal_recall() -> None:
    """Ties in recall@10 put queries with NaN first_hit_rank first."""
    per_query = pd.DataFrame(
        {
            "query": ["hit", "no hit", "lower recall"],
            "recall@10": [0.2, 0.2, 0.0],
            "first_hit_rank": [3.0, float("nan"), float("nan")],
        }
    )

    selected = error_analysis._select_worst(per_query, 2)

    assert selected["query"].tolist() == ["lower recall", "no hit"]


def test_main_writes_manual_review_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The report includes metrics, relevant/found titles, and TODO labels."""
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "query": ["query one", "query two"],
            "n_relevant": [1, 1],
            "first_hit_rank": [float("nan"), 2.0],
            "recall@10": [0.0, 0.5],
        }
    ).to_csv(results_dir / "per_query_faiss-flat.csv", index=False)
    (results_dir / "metrics_faiss-flat.json").write_text(
        json.dumps(
            {
                "model": "model-id",
                "n_items": 3,
                "n_queries": 2,
                "metrics": {"recall@10": 0.25, "mrr": 0.25},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "query": ["query one", "query two"],
            "relevant_ids": ["0", "1"],
        }
    ).to_csv("data/processed/eval_queries.csv", index=False)
    items_path = processed_dir / "items.csv"
    pd.DataFrame(
        {
            "item_id": [0, 1, 2],
            "title": ["Relevant one", "Relevant two", "Suggested title"],
            "category": ["Comedy", "Action", "Drama"],
            "description": ["Description one", "Description two", "Suggestion"],
            "text": ["one", "two", "three"],
        }
    ).to_csv(items_path, index=False)
    engine = FakeEngine()
    monkeypatch.setattr(error_analysis, "build_engine", lambda **kwargs: engine)

    exit_code = error_analysis.main(["--worst", "1"])

    report = Path("docs/ERROR_ANALYSIS.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert engine.queries == ["query one"]
    assert "Recall@10: 0.0000" in report
    assert "First relevant rank: not found" in report
    assert "Relevant one (Comedy)" in report
    assert "Suggested title (Drama), score=0.7500" in report
    assert "Failure type: TODO" in report
    assert "Data issue: TODO" in report
