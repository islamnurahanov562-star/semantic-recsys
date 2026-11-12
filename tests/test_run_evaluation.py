"""Tests for the real-data evaluation command."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_evaluation


class FakeEngine:
    """Return fixed rankings for the evaluation CLI test."""

    def __len__(self) -> int:
        """Return the number of fixture items."""
        return 3

    def recommend(self, query: str, k: int = 3) -> pd.DataFrame:
        """Return the configured ranked IDs for a fixture query."""
        rankings = {"query one": [1, 2, 0], "query two": [0, 2, 1]}
        return pd.DataFrame({"item_id": rankings[query][:k]})


def test_main_saves_metrics_metadata_and_per_query_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI writes the expected files and includes run metadata."""
    monkeypatch.chdir(tmp_path)
    eval_path = tmp_path / "eval_queries.csv"
    pd.DataFrame(
        {
            "query": ["query one", "query two"],
            "relevant_ids": ["1|2", "0"],
        }
    ).to_csv(eval_path, index=False)
    builds: list[dict[str, object]] = []

    def fake_build_engine(**kwargs: object) -> FakeEngine:
        builds.append(kwargs)
        return FakeEngine()

    monkeypatch.setattr(run_evaluation, "build_engine", fake_build_engine)

    exit_code = run_evaluation.main(
        [
            "--index",
            "numpy",
            "--model",
            "local-model",
            "--eval",
            str(eval_path),
            "--items",
            "items.csv",
            "--ks",
            "1",
            "3",
        ]
    )

    metrics_path = Path("results/metrics_numpy.json")
    per_query_path = Path("results/per_query_numpy.csv")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_query = pd.read_csv(per_query_path)

    assert exit_code == 0
    assert builds == [
        {
            "items_path": Path("items.csv"),
            "model_name": "local-model",
            "index_type": "numpy",
        }
    ]
    assert result["model"] == "local-model"
    assert result["index"] == "numpy"
    assert result["n_items"] == 3
    assert result["n_queries"] == 2
    assert result["latency_warmup_queries"] == 1
    assert set(result["versions"]) == {
        "python",
        "numpy",
        "pandas",
        "sentence_transformers",
        "faiss",
    }
    assert result["date"]
    assert result["metrics"]["recall@1"] == 0.75
    assert per_query.shape[0] == 2
