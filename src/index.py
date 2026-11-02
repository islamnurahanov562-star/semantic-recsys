"""Abstract interfaces for vector search indexes."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseIndex(ABC):
    """Define the common interface for vector search indexes."""

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build the index from vectors with shape (n_vectors, dimensions).

        Args:
            vectors: L2-normalized vectors as a float32 NumPy array with shape
                (n_vectors, dimensions).
        """
        ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for the nearest indexed vectors.

        Args:
            queries: Query vectors as a float32 NumPy array with shape
                (n_queries, dimensions).
            k: Number of results to return for each query.

        Returns:
            A tuple of scores and indices, each with shape (n_queries, k).
            Scores are float32 and sorted in descending order; indices are
            int64. Missing results use index -1 and score -inf.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save the index to a path.

        Args:
            path: Destination path for the serialized index.

        The indexed vectors are float32 values with shape
        (n_vectors, dimensions). Any missing search results must be represented
        by index -1 and score -inf.
        """
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load an index from a path.

        Args:
            path: Path to the serialized index.

        After loading, search results have shape (n_queries, k), with float32
        scores and int64 indices. Missing results use index -1 and score -inf.
        """
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed vectors.

        Returns:
            The number of vectors in the index.
        """
        ...
