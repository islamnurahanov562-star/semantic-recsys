"""Build recommendation engines from processed items and embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoder import EmbeddingEncoder
from src.engine import RecommendationEngine
from src.index import FaissIndex, NumpyIndex

logger = logging.getLogger(__name__)


def build_engine(
    items_path: str | Path = "data/processed/items.csv",
    embeddings_path: str | Path | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "faiss-flat",
) -> RecommendationEngine:
    """Load catalog data, embeddings, and a vector index into an engine.

    Args:
        items_path: CSV file containing processed catalog items.
        embeddings_path: Optional embedding cache path. By default, the cache
            is stored beside ``items_path`` using the model name as its slug.
        model_name: SentenceTransformers model identifier or local path.
        index_type: One of ``numpy``, ``faiss-flat``, or ``faiss-ivf``.

    Returns:
        A recommendation engine whose index rows align with item IDs.

    Raises:
        ValueError: If the catalog, embedding matrix, or index type is invalid.
        FileNotFoundError: If ``items_path`` does not exist.
    """
    if index_type not in {"numpy", "faiss-flat", "faiss-ivf"}:
        raise ValueError("index_type must be 'numpy', 'faiss-flat', or 'faiss-ivf'.")

    catalog_path = Path(items_path)
    items = pd.read_csv(catalog_path)
    required_columns = {"item_id", "title", "description", "category"}
    missing_columns = required_columns.difference(items.columns)
    if missing_columns:
        raise ValueError(
            f"Items are missing required columns: {sorted(missing_columns)}"
        )

    item_ids = pd.to_numeric(items["item_id"], errors="raise").to_numpy()
    if not np.array_equal(item_ids, np.arange(len(items))):
        raise ValueError("item_id values must match row positions starting from zero.")

    model_slug = model_name.split("/")[-1]
    if not model_slug:
        raise ValueError("model_name must end with a non-empty model name.")
    cache_path = (
        Path(embeddings_path)
        if embeddings_path is not None
        else catalog_path.parent / f"embeddings_{model_slug}.npy"
    )

    encoder = EmbeddingEncoder(model_name=model_name)
    if cache_path.is_file():
        embeddings = EmbeddingEncoder.load_embeddings(cache_path)
        logger.info("Loaded embedding cache from %s.", cache_path)
    else:
        if "text" not in items.columns:
            raise ValueError(
                "Items CSV must contain a 'text' column to build embeddings."
            )
        if items["text"].isna().any():
            raise ValueError("Items CSV contains missing values in the 'text' column.")
        texts = items["text"].tolist()
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Every value in items['text'] must be a string.")
        embeddings = encoder.encode(texts)
        EmbeddingEncoder.save_embeddings(embeddings, cache_path)
        logger.info("Built and cached embeddings at %s.", cache_path)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
        raise ValueError(
            "Embedding matrix must have one two-dimensional row per item; "
            f"got shape {embeddings.shape} for {len(items)} items."
        )

    if index_type == "numpy":
        index = NumpyIndex()
    elif index_type == "faiss-flat":
        index = FaissIndex(index_type="flat")
    else:
        index = FaissIndex(index_type="ivf")
    index.build(embeddings)
    logger.info("Built %s index for %d items.", index_type, len(items))
    return RecommendationEngine(items=items, encoder=encoder, index=index)
