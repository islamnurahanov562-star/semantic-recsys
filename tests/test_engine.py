"""Tests for recommendation ranking and metadata mapping."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine import RecommendationEngine
from src.index import BaseIndex


class FakeEncoder:
    """Return fixed normalized query vectors for known query strings."""

    def __init__(self) -> None:
        """Store deterministic three-dimensional query vectors."""
        self._query_vectors = {
            "space": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "romance": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

    def encode_query(self, query: str) -> np.ndarray:
        """Return the normalized vector associated with a known query."""
        return self._query_vectors[query.strip()][np.newaxis, :]


class BruteForceIndex(BaseIndex):
    """Use a full matrix product and argsort as a small exact test index."""

    def __init__(self) -> None:
        """Initialize an unbuilt brute-force index."""
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Store the supplied vectors as float32 values."""
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact dot-product results, padding when k exceeds size."""
        if self._vectors is None:
            raise RuntimeError("The index has not been built.")
        all_scores = np.asarray(queries, dtype=np.float32) @ self._vectors.T
        result_count = min(k, self.size)
        sorted_indices = np.argsort(-all_scores, axis=1)[:, :result_count]
        sorted_scores = np.take_along_axis(all_scores, sorted_indices, axis=1)
        scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
        indices = np.full((len(queries), k), -1, dtype=np.int64)
        scores[:, :result_count] = sorted_scores
        indices[:, :result_count] = sorted_indices
        return scores, indices

    def save(self, path: str | Path) -> None:
        """Save test vectors as a NumPy file."""
        if self._vectors is None:
            raise RuntimeError("The index has not been built.")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self._vectors)

    def load(self, path: str | Path) -> None:
        """Load test vectors from a NumPy file."""
        self._vectors = np.ascontiguousarray(np.load(Path(path)), dtype=np.float32)

    @property
    def size(self) -> int:
        """Return the number of indexed vectors."""
        return 0 if self._vectors is None else int(self._vectors.shape[0])


def _make_engine() -> RecommendationEngine:
    """Create the six-film fixture and its exact vector-search engine."""
    items = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3, 4, 5],
            "title": [
                "Starship Voyage",
                "Cosmic Frontier",
                "Alien World",
                "Love at Sunset",
                "A Romantic Tale",
                "Letters from Home",
            ],
            "description": [
                "A crew travels beyond the stars.",
                "Explorers cross a distant galaxy.",
                "A pilot discovers a new planet.",
                "Two strangers find love by the sea.",
                "A couple reunites after many years.",
                "A family reconnects through old letters.",
            ],
            "category": [
                "Sci-Fi",
                "Sci-Fi",
                "Sci-Fi",
                "Romance",
                "Romance",
                "Romance",
            ],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = BruteForceIndex()
    index.build(vectors)
    return RecommendationEngine(items, FakeEncoder(), index)


def test_top_one_is_the_semantically_closest_item() -> None:
    """The space query ranks the most space-aligned film first."""
    engine = _make_engine()

    result = engine.recommend("space", k=1)

    assert result.loc[0, "title"] == "Starship Voyage"
    assert result.loc[0, "item_id"] == 0


def test_recommendation_scores_are_sorted_descending() -> None:
    """Returned recommendations retain the index's descending score order."""
    engine = _make_engine()

    result = engine.recommend("space", k=6)

    assert np.all(np.diff(result["score"].to_numpy()) <= 0)


def test_k_larger_than_catalog_returns_all_items() -> None:
    """Missing index results are discarded when k exceeds the catalog size."""
    engine = _make_engine()

    result = engine.recommend("space", k=10)

    assert len(result) == 6
    assert result["item_id"].nunique() == 6


def test_category_filter_is_case_insensitive() -> None:
    """Category filtering keeps only matching categories regardless of case."""
    engine = _make_engine()

    result = engine.recommend("space", k=2, category="romance")

    assert len(result) == 2
    assert result["category"].tolist() == ["Romance", "Romance"]


def test_empty_query_and_zero_k_raise_value_error() -> None:
    """Blank queries and non-positive result counts are rejected."""
    engine = _make_engine()

    with pytest.raises(ValueError, match="query"):
        engine.recommend("   ")
    with pytest.raises(ValueError, match="positive integer"):
        engine.recommend("space", k=0)


def test_items_count_must_match_index_size() -> None:
    """Construction rejects metadata that cannot align with index positions."""
    engine = _make_engine()
    items = engine._items.iloc[:-1].copy()

    with pytest.raises(ValueError, match="must match index size"):
        RecommendationEngine(items, FakeEncoder(), engine._index)


def test_result_columns_and_ranks_follow_contract() -> None:
    """Results expose the contract columns and one-based sequential ranks."""
    engine = _make_engine()

    result = engine.recommend("romance", k=3)

    assert result.columns.tolist() == [
        "rank",
        "item_id",
        "title",
        "category",
        "score",
        "description",
    ]
    assert result["rank"].tolist() == [1, 2, 3]
    assert all(isinstance(score, float) for score in result["score"])
