"""Tests for the abstract vector index interface."""

from pathlib import Path

import numpy as np
import pytest

from src.index import BaseIndex


def test_base_index_cannot_be_instantiated() -> None:
    """The abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseIndex()


def test_partial_index_implementation_cannot_be_instantiated() -> None:
    """A subclass must implement every abstract member."""

    class PartialIndex(BaseIndex):
        def build(self, vectors: np.ndarray) -> None:
            self._vectors = vectors

    with pytest.raises(TypeError):
        PartialIndex()


def test_complete_index_implementation_can_be_used(tmp_path: Path) -> None:
    """A complete subclass can be instantiated and its interface called."""

    class MinimalIndex(BaseIndex):
        def __init__(self) -> None:
            self._vectors = np.empty((0, 2), dtype=np.float32)
            self._saved_path: Path | None = None

        def build(self, vectors: np.ndarray) -> None:
            self._vectors = vectors

        def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
            scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
            indices = np.full((queries.shape[0], k), -1, dtype=np.int64)
            return scores, indices

        def save(self, path: str | Path) -> None:
            self._saved_path = Path(path)

        def load(self, path: str | Path) -> None:
            self._saved_path = Path(path)

        @property
        def size(self) -> int:
            return self._vectors.shape[0]

    index = MinimalIndex()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    queries = np.array([[0.0, 1.0]], dtype=np.float32)
    path = tmp_path / "index.faiss"

    index.build(vectors)
    scores, indices = index.search(queries, k=2)
    index.save(path)
    index.load(path)

    assert index.size == 1
    assert scores.shape == (1, 2)
    assert scores.dtype == np.float32
    assert indices.shape == (1, 2)
    assert indices.dtype == np.int64
    assert np.all(scores == -np.inf)
    assert np.all(indices == -1)
    assert index._saved_path == path
