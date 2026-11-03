"""Combine text encoding and vector search into recommendations."""

from __future__ import annotations

import logging
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.index import BaseIndex

if TYPE_CHECKING:
    from src.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Retrieve and present semantically similar catalog items."""

    _OUTPUT_COLUMNS = [
        "rank",
        "item_id",
        "title",
        "category",
        "score",
        "description",
    ]
    _REQUIRED_ITEM_COLUMNS = {"item_id", "title", "description", "category"}

    def __init__(
        self,
        items: pd.DataFrame,
        encoder: EmbeddingEncoder,
        index: BaseIndex,
    ) -> None:
        """Store the pipeline components and validate item/index alignment.

        Args:
            items: Item metadata whose row positions correspond to index IDs.
            encoder: Text encoder used to vectorize queries.
            index: Built vector index used to retrieve item row positions.

        Raises:
            ValueError: If required item columns are missing or item count does
                not match the index size.
        """
        missing_columns = self._REQUIRED_ITEM_COLUMNS.difference(items.columns)
        if missing_columns:
            raise ValueError(
                f"Items are missing required columns: {sorted(missing_columns)}"
            )
        if len(items) != index.size:
            raise ValueError(
                f"Items row count ({len(items)}) must match index size ({index.size})."
            )

        self._items = items.reset_index(drop=True).copy()
        self._encoder = encoder
        self._index = index

    def recommend(
        self,
        query: str,
        k: int = 3,
        category: str | None = None,
    ) -> pd.DataFrame:
        """Return the top-k semantically similar items for a query.

        Args:
            query: Natural-language query to encode and search.
            k: Maximum number of recommendations to return.
            category: Optional exact category filter, compared case-insensitively.

        Returns:
            A DataFrame with columns rank, item_id, title, category, score, and
            description, ordered by descending search score.

        Raises:
            ValueError: If query is empty or k is not a positive integer.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer.")

        query_vector = self._encoder.encode_query(query)
        fetch_k = (
            min(int(k) * 5, self._index.size)
            if category is not None
            else min(int(k), self._index.size)
        )
        if fetch_k == 0:
            logger.debug("Recommendation query returned 0 results.")
            return pd.DataFrame(columns=self._OUTPUT_COLUMNS)

        scores, indices = self._index.search(query_vector, fetch_k)
        row_scores = scores[0]
        row_indices = indices[0]
        valid_mask = row_indices != -1
        valid_indices = row_indices[valid_mask].astype(np.int64, copy=False)
        valid_scores = row_scores[valid_mask]

        candidates = (
            self._items.iloc[valid_indices]
            .loc[:, ["item_id", "title", "category", "description"]]
            .copy()
        )
        candidates["score"] = [float(score) for score in valid_scores]

        if category is not None:
            category_mask = (
                candidates["category"]
                .astype("string")
                .str.casefold()
                .eq(category.casefold())
                .fillna(False)
            )
            candidates = candidates.loc[category_mask]

        candidates = candidates.head(int(k)).reset_index(drop=True)
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
        recommendations = candidates.loc[:, self._OUTPUT_COLUMNS]
        logger.debug("Recommendation query returned %d results.", len(recommendations))
        return recommendations

    def __len__(self) -> int:
        """Return the number of items available for recommendation."""
        return len(self._items)

    def __repr__(self) -> str:
        """Return a concise representation of this recommendation engine."""
        return (
            f"RecommendationEngine(items={len(self)}, "
            f"index={type(self._index).__name__})"
        )
