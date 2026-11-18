"""Tests for recommendation-engine construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.encoder import EmbeddingEncoder
from src.factory import build_engine
from src.index import FaissIndex, NumpyIndex


def _write_items(path: Path) -> None:
    """Write a small catalog with valid sequential item IDs."""
    pd.DataFrame(
        {
            "item_id": [0, 1],
            "title": ["First", "Second"],
            "description": ["First description", "Second description"],
            "category": ["Drama", "Comedy"],
            "text": ["First item text", "Second item text"],
        }
    ).to_csv(path, index=False)


@pytest.fixture
def encoder_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid device discovery or model loading in factory tests."""
    monkeypatch.setattr(EmbeddingEncoder, "_select_device", staticmethod(lambda: "cpu"))


def test_build_engine_loads_cached_embeddings_and_numpy_index(
    tmp_path: Path,
    encoder_on_cpu: None,
) -> None:
    """An existing embedding cache is used to build the requested index."""
    items_path = tmp_path / "items.csv"
    embeddings_path = tmp_path / "cached.npy"
    _write_items(items_path)
    expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(embeddings_path, expected)

    engine = build_engine(
        items_path=items_path,
        embeddings_path=embeddings_path,
        index_type="numpy",
    )

    assert isinstance(engine._index, NumpyIndex)
    assert engine._index.size == 2
    np.testing.assert_array_equal(engine._index._vectors, expected)


def test_build_engine_encodes_and_caches_missing_embeddings(
    tmp_path: Path,
    encoder_on_cpu: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing cache is generated and saved beside the item CSV."""
    items_path = tmp_path / "items.csv"
    _write_items(items_path)
    expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    encoded_texts: list[list[str]] = []

    def fake_encode(
        self: EmbeddingEncoder,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        encoded_texts.append(texts)
        return expected

    monkeypatch.setattr(EmbeddingEncoder, "encode", fake_encode)

    engine = build_engine(items_path=items_path, index_type="numpy")
    cache_path = tmp_path / "embeddings_all-MiniLM-L6-v2.npy"

    assert cache_path.is_file()
    assert encoded_texts == [["First item text", "Second item text"]]
    assert engine._index.size == 2
    np.testing.assert_array_equal(np.load(cache_path), expected)


def test_build_engine_supports_faiss_flat(
    tmp_path: Path,
    encoder_on_cpu: None,
) -> None:
    """The factory maps faiss-flat to a Flat FAISS index."""
    items_path = tmp_path / "items.csv"
    embeddings_path = tmp_path / "embeddings.npy"
    _write_items(items_path)
    np.save(embeddings_path, np.eye(2, dtype=np.float32))

    engine = build_engine(
        items_path=items_path,
        embeddings_path=embeddings_path,
        index_type="faiss-flat",
    )

    assert isinstance(engine._index, FaissIndex)
    assert engine._index.size == 2


def test_build_engine_rejects_unknown_index_type() -> None:
    """Unsupported index types fail before loading external files."""
    with pytest.raises(ValueError, match="index_type"):
        build_engine(index_type="unknown")
