"""Slow end-to-end tests for real catalog and vector-index integration."""

import numpy as np
import pandas as pd
import pytest

from src.engine import RecommendationEngine
from src.factory import build_engine
from src.index import FaissIndex, NumpyIndex

pytestmark = pytest.mark.slow
_INDEX_TYPES = ("numpy", "faiss-flat", "faiss-ivf")


@pytest.fixture(scope="module")
def engines() -> dict[str, RecommendationEngine]:
    """Build all supported engines once from the checked-in item and vector data."""
    return {
        index_type: build_engine(index_type=index_type) for index_type in _INDEX_TYPES
    }


def test_factory_builds_each_real_index_and_recommends_three_items(engines) -> None:
    """Every real index can serve a ranked top-three recommendation."""
    for index_type, engine in engines.items():
        result = engine.recommend("a funny movie about friends", k=3)

        assert len(engine) == 8804
        assert len(result) == 3
        assert np.all(np.diff(result["score"].to_numpy()) <= 0)
        if index_type == "numpy":
            assert isinstance(engine._index, NumpyIndex)
        else:
            assert isinstance(engine._index, FaissIndex)


def test_numpy_and_faiss_flat_match_on_five_evaluation_queries(engines) -> None:
    """Exact indexes return matching top-three IDs, allowing score ties."""
    eval_queries = pd.read_csv(
        "data/processed/eval_queries.csv",
        dtype={"query": "string"},
    )["query"].head(5)
    numpy_engine = engines["numpy"]
    faiss_engine = engines["faiss-flat"]

    for query in eval_queries:
        numpy_results = numpy_engine.recommend(str(query), k=3)
        faiss_results = faiss_engine.recommend(str(query), k=3)
        numpy_ids = numpy_results["item_id"].tolist()
        faiss_ids = faiss_results["item_id"].tolist()
        numpy_scores = numpy_results["score"].to_numpy()
        faiss_scores = faiss_results["score"].to_numpy()

        np.testing.assert_allclose(numpy_scores, faiss_scores, atol=1e-6)
        tied_scores = np.any(np.abs(np.diff(numpy_scores)) <= 1e-6) or np.any(
            np.abs(np.diff(faiss_scores)) <= 1e-6
        )
        if tied_scores:
            assert set(numpy_ids) == set(faiss_ids)
        else:
            assert numpy_ids == faiss_ids


def test_real_category_filter_returns_only_requested_category(engines) -> None:
    """Category filtering keeps only items matching the selected real category."""
    engine = engines["numpy"]
    query = "a funny movie about friends"
    unfiltered = engine.recommend(query, k=1)
    category = str(unfiltered.loc[0, "category"])

    filtered = engine.recommend(query, k=3, category=category)

    assert not filtered.empty
    assert filtered["category"].str.casefold().eq(category.casefold()).all()
