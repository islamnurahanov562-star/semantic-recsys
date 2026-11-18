"""Tests for embedding-model comparison output and cache timing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import compare_models


class FakeEngine:
    """Return fixed rankings for a one-query model comparison test."""

    def __len__(self) -> int:
        """Return the number of fixture items."""
        return 3

    def recommend(self, query: str, k: int = 3) -> pd.DataFrame:
        """Return deterministic ranked fixture IDs."""
        return pd.DataFrame({"item_id": [0, 1, 2][:k]})


def test_main_records_nan_encoding_time_for_cached_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing cache is reused and its encoding time is NaN."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "item_id": [0, 1, 2],
            "title": ["First", "Second", "Third"],
            "description": ["one", "two", "three"],
            "category": ["Drama"] * 3,
            "text": ["first text", "second text", "third text"],
        }
    ).to_csv(data_dir / "items.csv", index=False)
    pd.DataFrame({"query": ["fixture query"], "relevant_ids": ["0"]}).to_csv(
        data_dir / "eval_queries.csv", index=False
    )
    cache_path = data_dir / "embeddings_local-model.npy"
    np.save(cache_path, np.eye(3, dtype=np.float32))
    build_calls: list[dict[str, object]] = []

    def fake_build_engine(**kwargs: object) -> FakeEngine:
        build_calls.append(kwargs)
        return FakeEngine()

    monkeypatch.setattr(compare_models, "build_engine", fake_build_engine)
    output_path = tmp_path / "results" / "comparison.csv"

    exit_code = compare_models.main(
        [
            "--models",
            "local-model",
            "--index",
            "faiss-flat",
            "--out",
            str(output_path),
        ]
    )

    result = pd.read_csv(output_path)
    assert exit_code == 0
    assert build_calls[0]["model_name"] == "local-model"
    assert build_calls[0]["index_type"] == "faiss-flat"
    assert result.loc[0, "dim"] == 3
    assert pd.isna(result.loc[0, "embed_time_s"])
    assert result.loc[0, "recall@1"] == 1.0


@pytest.mark.slow
def test_uncached_model_encoding_uses_temporary_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uncached encoding is timed and temporary vectors are passed to factory."""
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    items = pd.DataFrame({"text": ["first", "second"]})
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    encoded: list[list[str]] = []
    built_with: list[dict[str, object]] = []

    def fake_encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        encoded.append(texts)
        return embeddings

    def fake_build_engine(**kwargs: object) -> FakeEngine:
        built_with.append(kwargs)
        assert Path(str(kwargs["embeddings_path"])).is_file()
        return FakeEngine()

    monkeypatch.setattr(compare_models.EmbeddingEncoder, "encode", fake_encode)
    monkeypatch.setattr(compare_models, "build_engine", fake_build_engine)
    query_path = data_dir / "eval_queries.csv"
    pd.DataFrame({"query": ["fixture"], "relevant_ids": ["0"]}).to_csv(
        query_path, index=False
    )

    row = compare_models._evaluate_one_model(
        "uncached-model",
        items,
        pd.read_csv(query_path, dtype=str),
        "faiss-flat",
    )

    assert encoded == [["first", "second"]]
    assert row["dim"] == 2
    assert row["embed_time_s"] >= 0
    assert built_with
