"""Tests for the evaluation command-line runner."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import evaluate_engine


class FakeEngine:
    """Return deterministic rankings for evaluation-runner tests."""

    def __len__(self) -> int:
        """Return the number of fixture items."""
        return 3

    def recommend(self, query: str, k: int) -> pd.DataFrame:
        """Return ranked fixture IDs for a known query."""
        rankings = {"query one": [1, 0, 2], "query two": [2, 1, 0]}
        return pd.DataFrame({"item_id": rankings[query][:k]})


def test_main_writes_per_query_csv_and_summary_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI saves measured query rows and aggregate metrics."""
    query_path = tmp_path / "eval_queries.csv"
    pd.DataFrame(
        {
            "query": ["query one", "query two"],
            "relevant_ids": ["1|2", "0"],
        }
    ).to_csv(query_path, index=False)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(evaluate_engine, "build_engine", lambda **kwargs: FakeEngine())

    exit_code = evaluate_engine.main(
        [
            "--queries",
            str(query_path),
            "--index-type",
            "numpy",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert exit_code == 0
    per_query = pd.read_csv(results_dir / "retrieval_quality.csv")
    summary = json.loads((results_dir / "eval_summary.json").read_text())
    assert per_query.shape[0] == 2
    assert per_query["index_type"].tolist() == ["numpy", "numpy"]
    assert summary["runs"]["numpy"]["n_items"] == 3
    assert summary["runs"]["numpy"]["metrics"]["n_queries"] == 2
    assert summary["runs"]["numpy"]["metrics"]["recall@1"] == 0.25


def test_main_evaluates_all_supported_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default all mode records one result set per supported index."""
    query_path = tmp_path / "eval_queries.csv"
    pd.DataFrame({"query": ["query one"], "relevant_ids": ["1"]}).to_csv(
        query_path, index=False
    )
    results_dir = tmp_path / "results"
    built_indexes: list[str] = []

    def fake_build_engine(**kwargs: object) -> FakeEngine:
        built_indexes.append(str(kwargs["index_type"]))
        return FakeEngine()

    monkeypatch.setattr(evaluate_engine, "build_engine", fake_build_engine)
    exit_code = evaluate_engine.main(
        ["--queries", str(query_path), "--results-dir", str(results_dir)]
    )

    summary = json.loads((results_dir / "eval_summary.json").read_text())
    per_query = pd.read_csv(results_dir / "retrieval_quality.csv")
    assert exit_code == 0
    assert built_indexes == ["numpy", "faiss-flat", "faiss-ivf"]
    assert set(summary["runs"]) == set(built_indexes)
    assert per_query["index_type"].nunique() == 3
