"""Tests for the FAISS Flat vector index."""

from pathlib import Path

import numpy as np
import pytest

from src.index import FaissIndex
from src.index import NumpyIndex

faiss = pytest.importorskip("faiss")


def _normalized_vectors(rows: int, dimensions: int, seed: int = 42) -> np.ndarray:
    """Create deterministic random float32 vectors with unit L2 norms."""
    generator = np.random.default_rng(seed)
    vectors = generator.normal(size=(rows, dimensions)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _clustered_normalized_vectors(
    cluster_count: int,
    vectors_per_cluster: int,
    dimensions: int,
    seed: int = 42,
) -> np.ndarray:
    """Create deterministic random unit vectors around random unit centers."""
    generator = np.random.default_rng(seed)
    centers = generator.normal(size=(cluster_count, dimensions)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    noise = generator.normal(
        scale=0.05,
        size=(cluster_count * vectors_per_cluster, dimensions),
    ).astype(np.float32)
    vectors = np.repeat(centers, vectors_per_cluster, axis=0) + noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors, dtype=np.float32)


def test_flat_search_returns_sorted_float32_scores_and_pads() -> None:
    """Flat IP returns exact scores and sentinel padding for k above size."""
    index = FaissIndex()
    index.build(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))

    scores, indices = index.search(np.array([[1.0, 0.0]]), k=5)

    assert index.size == 3
    assert scores.shape == (1, 5)
    assert scores.dtype == np.float32
    assert indices.shape == (1, 5)
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(indices, [[0, 1, 2, -1, -1]])
    np.testing.assert_allclose(scores[0, :3], [1.0, 0.0, -1.0])
    assert np.isneginf(scores[0, 3:]).all()


def test_search_validates_build_state_dimensions_and_k() -> None:
    """Search rejects an unbuilt index, mismatched dimensions, and invalid k."""
    index = FaissIndex()

    with pytest.raises(RuntimeError, match="built"):
        index.search(np.ones((1, 2)), k=1)

    index.build(np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones((1, 3)), k=1)
    with pytest.raises(ValueError, match="at least 1"):
        index.search(np.ones((1, 2)), k=0)


def test_build_rejects_non_2d_vectors() -> None:
    """Flat build requires a two-dimensional vector array."""
    with pytest.raises(ValueError, match="2D"):
        FaissIndex().build(np.array([1.0, 0.0]))


def test_invalid_index_type_is_rejected() -> None:
    """The constructor accepts only flat and ivf index types."""
    with pytest.raises(ValueError, match="flat.*ivf"):
        FaissIndex(index_type="hnsw")


def test_save_and_load_preserve_search_results(tmp_path: Path) -> None:
    """Saving creates parent directories and loading restores Flat search."""
    index = FaissIndex()
    index.build(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    path = tmp_path / "nested" / "index.faiss"
    expected_scores, expected_indices = index.search(np.array([[1.0, 0.0]]), k=2)

    index.save(path)
    loaded_index = FaissIndex()
    loaded_index.load(path)
    actual_scores, actual_indices = loaded_index.search(np.array([[1.0, 0.0]]), k=2)

    assert loaded_index.size == 2
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_scores, expected_scores)


def test_flat_matches_numpy_top_five() -> None:
    """FAISS Flat and NumPy return matching exact results on normalized vectors."""
    vectors = _normalized_vectors(500, 32)
    queries = _normalized_vectors(20, 32, seed=43)
    faiss_index = FaissIndex(index_type="flat")
    numpy_index = NumpyIndex()
    faiss_index.build(vectors)
    numpy_index.build(vectors)

    faiss_scores, faiss_indices = faiss_index.search(queries, k=5)
    numpy_scores, numpy_indices = numpy_index.search(queries, k=5)

    np.testing.assert_array_equal(faiss_indices, numpy_indices)
    np.testing.assert_allclose(faiss_scores, numpy_scores, atol=1e-5)


def test_ivf_with_all_lists_probed_matches_flat_top_ten() -> None:
    """Probing every IVF list returns the same exact top ten as Flat."""
    vectors = _normalized_vectors(1000, 32)
    queries = _normalized_vectors(20, 32, seed=43)
    flat_index = FaissIndex(index_type="flat")
    ivf_index = FaissIndex(index_type="ivf", nlist=10, nprobe=10)
    flat_index.build(vectors)
    ivf_index.build(vectors)

    _, flat_indices = flat_index.search(queries, k=10)
    _, ivf_indices = ivf_index.search(queries, k=10)

    np.testing.assert_array_equal(ivf_indices, flat_indices)


def test_ivf_recall_at_ten_is_at_least_point_nine() -> None:
    """IVF retrieves at least nine of Flat's top-ten clustered neighbors."""
    vectors = _clustered_normalized_vectors(50, 100, 64)
    queries = vectors[:20].copy()
    flat_index = FaissIndex(index_type="flat")
    ivf_index = FaissIndex(index_type="ivf", nlist=50, nprobe=10)
    flat_index.build(vectors)
    ivf_index.build(vectors)

    _, flat_indices = flat_index.search(queries, k=10)
    _, ivf_indices = ivf_index.search(queries, k=10)
    recall_at_ten = np.mean(
        [
            len(set(flat_row).intersection(ivf_row)) / 10
            for flat_row, ivf_row in zip(flat_indices, ivf_indices)
        ]
    )

    assert recall_at_ten >= 0.9


@pytest.mark.parametrize("index_type", ["flat", "ivf"])
def test_save_load_round_trip_preserves_search_and_nprobe(
    tmp_path: Path,
    index_type: str,
) -> None:
    """Both index types preserve search output and IVF probing configuration."""
    vectors = _normalized_vectors(400, 24)
    queries = _normalized_vectors(6, 24, seed=43)
    index = FaissIndex(index_type=index_type, nlist=8, nprobe=3)
    index.build(vectors)
    if index_type == "ivf":
        index.set_nprobe(5)
    expected_scores, expected_indices = index.search(queries, k=10)
    path = tmp_path / index_type / "index.faiss"

    index.save(path)
    loaded_index = FaissIndex(index_type=index_type, nlist=8, nprobe=1)
    loaded_index.load(path)
    actual_scores, actual_indices = loaded_index.search(queries, k=10)

    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_scores, expected_scores)
    if index_type == "ivf":
        assert loaded_index._index.nprobe == 5


@pytest.mark.parametrize("index_type", ["flat", "ivf"])
def test_large_k_pads_with_missing_result_sentinels(index_type: str) -> None:
    """Both index types pad top-k outputs with -1 and negative infinity."""
    vectors = _normalized_vectors(12, 8)
    index = FaissIndex(index_type=index_type, nlist=2, nprobe=2)
    index.build(vectors)

    scores, indices = index.search(_normalized_vectors(2, 8, seed=43), k=15)

    assert scores.shape == (2, 15)
    assert scores.dtype == np.float32
    assert indices.shape == (2, 15)
    assert indices.dtype == np.int64
    assert np.all(indices[:, 12:] == -1)
    assert np.isneginf(scores[:, 12:]).all()


def test_invalid_type_unbuilt_search_and_wrong_dimension() -> None:
    """The constructor and search reject unsupported or invalid states."""
    with pytest.raises(ValueError, match="flat.*ivf"):
        FaissIndex(index_type="hnsw")
    with pytest.raises(RuntimeError, match="built"):
        FaissIndex().search(np.ones((1, 3), dtype=np.float32), k=1)

    index = FaissIndex()
    index.build(_normalized_vectors(8, 3))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones((1, 4), dtype=np.float32), k=1)


def test_ivf_reduces_nlist_for_small_training_sets() -> None:
    """IVF uses one list when the training set is smaller than the heuristic."""
    index = FaissIndex(index_type="ivf", nlist=100, nprobe=10)
    vectors = _normalized_vectors(30, 8)

    index.build(vectors)

    assert index._index.nlist == 1
    scores, indices = index.search(vectors[:2], k=3)
    assert scores.shape == (2, 3)
    assert indices.shape == (2, 3)


def test_set_nprobe_validates_range_and_updates_ivf_index() -> None:
    """set_nprobe accepts values within the trained IVF list count only."""
    index = FaissIndex(index_type="ivf", nlist=8, nprobe=2)
    index.build(_normalized_vectors(400, 16))

    index.set_nprobe(5)
    assert index._index.nprobe == 5
    with pytest.raises(ValueError, match="nprobe"):
        index.set_nprobe(0)
    with pytest.raises(ValueError, match="nprobe"):
        index.set_nprobe(index._index.nlist + 1)


@pytest.mark.parametrize("index_type", ["flat", "ivf"])
def test_empty_query_batch_returns_empty_result_matrices(index_type: str) -> None:
    """Searching an empty query batch preserves the requested result width."""
    index = FaissIndex(index_type=index_type, nlist=2, nprobe=2)
    index.build(_normalized_vectors(100, 8))

    scores, indices = index.search(np.empty((0, 8), dtype=np.float32), k=4)

    assert scores.shape == (0, 4)
    assert scores.dtype == np.float32
    assert indices.shape == (0, 4)
    assert indices.dtype == np.int64
