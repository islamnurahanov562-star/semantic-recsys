"""Tests for the exact NumPy vector index."""

from pathlib import Path

import numpy as np
import pytest

from src.index import NumpyIndex


def test_build_stores_contiguous_float32_vectors_and_warns_on_norms(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Build converts vectors and warns when their norms are not unit length."""
    index = NumpyIndex()
    vectors = np.array([[3.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    index.build(vectors)

    assert index._vectors is not None
    assert index._vectors.dtype == np.float32
    assert index._vectors.flags.c_contiguous
    assert index.size == 2
    assert "not L2-normalized" in caplog.text


def test_build_rejects_non_2d_and_empty_vectors() -> None:
    """Build requires a two-dimensional array with at least one row."""
    index = NumpyIndex()

    with pytest.raises(ValueError, match="2D"):
        index.build(np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="at least one row"):
        index.build(np.empty((0, 2), dtype=np.float32))


def test_search_returns_sorted_top_k_and_pads_missing_results() -> None:
    """Search sorts scores descending and pads when k exceeds index size."""
    index = NumpyIndex()
    index.build(np.eye(3, dtype=np.float32))

    scores, indices = index.search(np.array([[0.8, 0.2, 0.1]]), k=5)

    assert scores.shape == (1, 5)
    assert scores.dtype == np.float32
    assert indices.shape == (1, 5)
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(indices, [[0, 1, 2, -1, -1]])
    np.testing.assert_allclose(scores[0, :3], [0.8, 0.2, 0.1])
    assert np.isneginf(scores[0, 3:]).all()


def test_search_rejects_unbuilt_index_invalid_queries_and_k() -> None:
    """Search validates index state, query dimensions, and result count."""
    index = NumpyIndex()

    with pytest.raises(RuntimeError, match="built"):
        index.search(np.ones((1, 2)), k=1)

    index.build(np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        index.search(np.ones(2), k=1)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones((1, 3)), k=1)
    with pytest.raises(ValueError, match="at least 1"):
        index.search(np.ones((1, 2)), k=0)


def test_save_and_load_round_trip_creates_parent_directory(tmp_path: Path) -> None:
    """Save creates missing folders and load restores the vector matrix."""
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = NumpyIndex()
    index.build(vectors)
    path = tmp_path / "nested" / "index.npy"

    index.save(path)
    loaded_index = NumpyIndex()
    loaded_index.load(path)

    assert loaded_index.size == 2
    np.testing.assert_array_equal(loaded_index._vectors, vectors)
