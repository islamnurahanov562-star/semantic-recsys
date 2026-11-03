"""Encode item and query text as normalized sentence embeddings."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Lazily load a SentenceTransformer and encode normalized text vectors."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        """Store encoder settings and select the compute device.

        Args:
            model_name: SentenceTransformers model identifier or local path.
            batch_size: Number of texts encoded per batch.
            device: Explicit device name, or None to select CUDA when available.
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device if device is not None else self._select_device()
        self._model: SentenceTransformer | None = None
        logger.info("Embedding device selected: %s", self._device)

    @staticmethod
    def _select_device() -> str:
        """Select CUDA when available, otherwise CPU.

        Returns:
            The selected device name.
        """
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_model(self) -> SentenceTransformer:
        """Load and cache the configured SentenceTransformer model.

        Returns:
            The cached SentenceTransformer instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info("Loaded SentenceTransformer model: %s", self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Return the model's sentence embedding dimension."""
        dimension = self._get_model().get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("The model did not report an embedding dimension.")
        return int(dimension)

    def encode(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a sequence of strings as normalized float32 vectors.

        Args:
            texts: Text strings to encode.
            show_progress: Whether to show the model's progress bar.

        Returns:
            A contiguous float32 array with shape (n_texts, embedding_dim).

        Raises:
            TypeError: If texts is a string, is not a sequence, or contains
                non-string values.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a single string.")
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings.")

        text_list = list(texts)
        for index, text in enumerate(text_list):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string; got {type(text).__name__}."
                )

        if not text_list:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self._get_model().encode(
            text_list,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode one non-empty query as a single normalized vector.

        Args:
            query: Query text to encode.

        Returns:
            A float32 array with shape (1, embedding_dim).

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or contains only whitespace.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty or whitespace-only.")
        return self.encode([normalized_query])

    @staticmethod
    def save_embeddings(emb: np.ndarray, path: str | Path) -> None:
        """Save embeddings to a NumPy file, creating parent directories.

        Args:
            emb: Embedding matrix to save.
            path: Destination path for the .npy file.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, emb)
        logger.info("Saved embeddings to %s", output_path)

    @staticmethod
    def load_embeddings(path: str | Path) -> np.ndarray:
        """Load a two-dimensional embedding matrix as float32.

        Args:
            path: Path to the .npy embedding file.

        Returns:
            A float32 embedding matrix.

        Raises:
            ValueError: If the loaded array is not two-dimensional.
        """
        embeddings = np.load(Path(path))
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array; got shape {embeddings.shape}."
            )
        return np.asarray(embeddings, dtype=np.float32)
