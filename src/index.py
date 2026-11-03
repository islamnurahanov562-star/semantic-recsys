"""Abstract interfaces for vector search indexes."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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


class NumpyIndex(BaseIndex):
    """Perform exact inner-product search with a NumPy vector matrix."""

    def __init__(self) -> None:
        """Initialize an empty NumPy index."""
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Validate and store a two-dimensional vector matrix.

        Args:
            vectors: Vector matrix with at least one row.

        Raises:
            ValueError: If vectors is not two-dimensional or has no rows.
        """
        vector_array = np.asarray(vectors)
        if vector_array.ndim != 2:
            raise ValueError("vectors must be a 2D array.")
        if vector_array.shape[0] == 0:
            raise ValueError("vectors must contain at least one row.")

        stored_vectors = np.ascontiguousarray(vector_array, dtype=np.float32)
        norms = np.linalg.norm(stored_vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
            logger.warning(
                "Some indexed vectors are not L2-normalized within atol=1e-3."
            )
        self._vectors = stored_vectors

    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-k indexed vectors for every query.

        Args:
            queries: Query matrix with shape (n_queries, dimensions).
            k: Number of results to return per query.

        Returns:
            Scores as float32 and indices as int64, both with shape
            (n_queries, k). Missing results are padded with -inf and -1.

        Raises:
            RuntimeError: If the index has not been built.
            ValueError: If queries have an invalid shape or k is less than 1.
        """
        vectors = self._require_vectors()
        query_array = np.asarray(queries)
        if query_array.ndim != 2:
            raise ValueError("queries must be a 2D array.")
        if query_array.shape[1] != vectors.shape[1]:
            raise ValueError(
                "queries must have the same vector dimension as the index."
            )
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
            raise ValueError("k must be a positive integer.")
        if k < 1:
            raise ValueError("k must be at least 1.")

        query_array = np.ascontiguousarray(query_array, dtype=np.float32)
        # For unit vectors, both L2 norms are 1.
        # Therefore cosine(x, y) = dot(x, y) / (||x||2 * ||y||2) = dot(x, y).
        # The matrix product below is cosine similarity for normalized vectors.
        scores = query_array @ vectors.T
        result_count = min(int(k), vectors.shape[0])
        candidate_indices = np.argpartition(
            scores, kth=vectors.shape[0] - result_count, axis=1
        )[:, -result_count:]
        candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        top_indices = np.take_along_axis(candidate_indices, order, axis=1)
        top_scores = np.take_along_axis(candidate_scores, order, axis=1)

        result_scores = np.full(
            (query_array.shape[0], int(k)), -np.inf, dtype=np.float32
        )
        result_indices = np.full((query_array.shape[0], int(k)), -1, dtype=np.int64)
        result_scores[:, :result_count] = top_scores
        result_indices[:, :result_count] = top_indices
        return result_scores, result_indices

    def save(self, path: str | Path) -> None:
        """Save the indexed vectors as a NumPy .npy file.

        Args:
            path: Destination path for the vector matrix.

        Raises:
            RuntimeError: If the index has not been built.
        """
        vectors = self._require_vectors()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, vectors)
        logger.info("Saved NumPy index vectors to %s", output_path)

    def load(self, path: str | Path) -> None:
        """Load and validate vectors from a NumPy .npy file.

        Args:
            path: Source path for the vector matrix.

        Raises:
            ValueError: If the loaded array is not a valid vector matrix.
        """
        self.build(np.load(Path(path)))

    @property
    def size(self) -> int:
        """Return the number of indexed vectors, or zero when unbuilt."""
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def _require_vectors(self) -> np.ndarray:
        """Return the vector matrix or fail if the index is unbuilt.

        Returns:
            The indexed vector matrix.

        Raises:
            RuntimeError: If build has not been called.
        """
        if self._vectors is None:
            raise RuntimeError("The index must be built before this operation.")
        return self._vectors
