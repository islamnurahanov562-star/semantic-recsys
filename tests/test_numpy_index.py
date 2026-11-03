"""Tests for the exact NumPy vector index."""

from pathlib import Path

import numpy as np
import pytest

from src.index import NumpyIndex


def _normalized_random_vectors(rows: int, dimensions: int) -> np.ndarray:
    """Create reproducible float32 vectors with unit L2 norms."""
    generator = np.random.default_rng(42)
    vectors = generator.normal(size=(rows, dimensions)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


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
    vectors = _normalized_random_vectors(3, 8)
    index.build(vectors)

    scores, indices = index.search(_normalized_random_vectors(1, 8), k=5)

    assert scores.shape == (1, 5)
    assert scores.dtype == np.float32
    assert indices.shape == (1, 5)
    assert indices.dtype == np.int64
    assert set(indices[0, :3]) == {0, 1, 2}
    assert np.all(np.diff(scores[0, :3]) <= 0)
    assert np.isneginf(scores[0, 3:]).all()
    assert np.all(indices[0, 3:] == -1)


def test_query_equal_to_indexed_vector_returns_it_first() -> None:
    """An indexed vector retrieves itself with cosine similarity one."""
    vectors = _normalized_random_vectors(12, 16)
    index = NumpyIndex()
    index.build(vectors)

    scores, indices = index.search(vectors[[7]], k=5)

    assert indices[0, 0] == 7
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_scores_are_descending_for_each_query_row() -> None:
    """Every query's scores are ordered from highest to lowest."""
    index = NumpyIndex()
    index.build(_normalized_random_vectors(40, 16))

    scores, _ = index.search(_normalized_random_vectors(10, 16), k=12)

    assert np.all(np.diff(scores, axis=1) <= 0)


def test_top_five_matches_full_argsort_reference() -> None:
    """Top-five indices and scores match exhaustive matrix multiplication."""
    vectors = _normalized_random_vectors(37, 16)
    queries = _normalized_random_vectors(10, 16)
    index = NumpyIndex()
    index.build(vectors)

    actual_scores, actual_indices = index.search(queries, k=5)
    all_scores = queries @ vectors.T
    expected_indices = np.argsort(-all_scores, axis=1)[:, :5]
    expected_scores = np.take_along_axis(all_scores, expected_indices, axis=1)

    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_allclose(actual_scores, expected_scores)


def test_batched_search_returns_one_result_row_per_query() -> None:
    """Searching ten queries returns the requested (10, k) shapes."""
    index = NumpyIndex()
    index.build(_normalized_random_vectors(30, 16))

    scores, indices = index.search(_normalized_random_vectors(10, 16), k=6)

    assert scores.shape == (10, 6)
    assert indices.shape == (10, 6)


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
    """Saving and loading preserves search results and creates parent folders."""
    vectors = _normalized_random_vectors(20, 16)
    queries = _normalized_random_vectors(4, 16)
    index = NumpyIndex()
    index.build(vectors)
    path = tmp_path / "nested" / "index.npy"
    expected_scores, expected_indices = index.search(queries, k=7)

    index.save(path)
    loaded_index = NumpyIndex()
    loaded_index.load(path)
    actual_scores, actual_indices = loaded_index.search(queries, k=7)

    assert loaded_index.size == 20
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_scores, expected_scores)


def test_size_is_zero_before_build_and_vector_count_after_build() -> None:
    """Size reflects whether the index is built and its number of vectors."""
    index = NumpyIndex()
    assert index.size == 0

    index.build(_normalized_random_vectors(9, 16))

    assert index.size == 9
