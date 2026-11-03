"""Abstract interfaces for vector search indexes."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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


class FaissIndex(BaseIndex):
    """Search vectors with exact Flat or approximate IVF FAISS indexes.

    ``nlist`` controls how many clusters IVF divides the vector space into.
    ``nprobe`` controls how many of those clusters are searched per query.
    More lists can reduce work per query but require more training; probing more
    lists improves recall at the cost of search time. Flat IP remains exact and
    equals cosine similarity when vectors are L2-normalized.
    """

    def __init__(
        self,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        """Initialize a FAISS index configuration.

        Args:
            index_type: ``flat`` for exact search or ``ivf`` for approximate search.
            nlist: Target number of IVF clusters.
            nprobe: Number of IVF clusters to probe per query.

        Raises:
            ValueError: If index_type is neither ``flat`` nor ``ivf``.
        """
        if index_type not in {"flat", "ivf"}:
            raise ValueError("index_type must be 'flat' or 'ivf'.")
        if not isinstance(nlist, (int, np.integer)) or isinstance(nlist, bool):
            raise ValueError("nlist must be a positive integer.")
        if not isinstance(nprobe, (int, np.integer)) or isinstance(nprobe, bool):
            raise ValueError("nprobe must be a positive integer.")
        if nlist < 1:
            raise ValueError("nlist must be at least 1.")
        if nprobe < 1:
            raise ValueError("nprobe must be at least 1.")
        self._index_type = index_type
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        self._index: Any | None = None
        self._quantizer: Any | None = None
        self._dimension: int | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Build a FAISS inner-product index from vector rows.

        For L2-normalized vectors, ``IndexFlatIP`` computes exact cosine
        similarity because each vector has unit norm, so cosine similarity
        reduces to the inner product.

        Args:
            vectors: Two-dimensional vector matrix.

        Raises:
            ValueError: If vectors are not two-dimensional or have no dimension.
        """
        vector_array = np.asarray(vectors)
        if vector_array.ndim != 2:
            raise ValueError("vectors must be a 2D array.")
        if vector_array.shape[1] < 1:
            raise ValueError("vectors must have at least one dimension.")
        if self._index_type == "ivf" and vector_array.shape[0] == 0:
            raise ValueError("IVF vectors must contain at least one row.")

        vector_array = np.ascontiguousarray(vector_array, dtype=np.float32)
        import faiss

        dimension = int(vector_array.shape[1])
        quantizer: Any | None = None
        if self._index_type == "flat":
            index = faiss.IndexFlatIP(dimension)
            index.add(vector_array)
        else:
            vector_count = int(vector_array.shape[0])
            effective_nlist = max(1, min(self.nlist, vector_count // 39))
            if effective_nlist < self.nlist:
                logger.info(
                    "Reducing IVF nlist from %d to %d for %d training vectors.",
                    self.nlist,
                    effective_nlist,
                    vector_count,
                )
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(
                quantizer,
                dimension,
                effective_nlist,
                faiss.METRIC_INNER_PRODUCT,
            )
            index.train(vector_array)
            index.add(vector_array)
            self.nlist = effective_nlist
            self.nprobe = min(self.nprobe, effective_nlist)
            index.nprobe = self.nprobe

        self._index = index
        self._quantizer = quantizer
        self._dimension = dimension
        logger.info(
            "Built FAISS %s index with %d vectors and dimension %d.",
            self._index_type.upper(),
            self.size,
            self._dimension,
        )

    def set_nprobe(self, nprobe: int) -> None:
        """Set the number of IVF clusters searched for each query.

        Increasing ``nprobe`` examines more clusters and can improve recall, but
        it also increases search time. It must remain between one and the
        trained ``nlist``; Flat indexes do not use this setting.

        Args:
            nprobe: Number of IVF clusters to probe.

        Raises:
            ValueError: If this is not an IVF index, nprobe is not an integer,
                or nprobe is outside the trained list range.
        """
        if self._index_type != "ivf":
            raise ValueError("nprobe can only be set on an IVF index.")
        if not isinstance(nprobe, (int, np.integer)) or isinstance(
            nprobe, (bool, np.bool_)
        ):
            raise ValueError("nprobe must be an integer.")
        if not 1 <= nprobe <= self.nlist:
            raise ValueError(f"nprobe must be between 1 and nlist ({self.nlist}).")

        self.nprobe = int(nprobe)
        if self._index is not None:
            self._index.nprobe = self.nprobe

    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top-k inner-product matches.

        Args:
            queries: Query matrix with shape (n_queries, dimensions).
            k: Number of results to return for each query.

        Returns:
            Scores as float32 and indices as int64, both with shape
            (n_queries, k). Missing results are padded with -inf and -1.

        Raises:
            RuntimeError: If the index has not been built or loaded.
            ValueError: If query shape, dimensions, or k are invalid.
        """
        if self._index is None or self._dimension is None:
            raise RuntimeError("The index must be built before searching.")

        query_array = np.asarray(queries)
        if query_array.ndim != 2:
            raise ValueError("queries must be a 2D array.")
        if query_array.shape[1] != self._dimension:
            raise ValueError("queries must have the same dimension as the index.")
        if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
            raise ValueError("k must be a positive integer.")
        if k < 1:
            raise ValueError("k must be at least 1.")

        query_array = np.ascontiguousarray(query_array, dtype=np.float32)
        if query_array.shape[0] == 0:
            return (
                np.empty((0, int(k)), dtype=np.float32),
                np.empty((0, int(k)), dtype=np.int64),
            )
        result_count = min(int(k), self.size)
        if result_count == 0:
            return (
                np.full((query_array.shape[0], int(k)), -np.inf, dtype=np.float32),
                np.full((query_array.shape[0], int(k)), -1, dtype=np.int64),
            )

        scores, indices = self._index.search(query_array, result_count)
        scores = np.asarray(scores, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        missing_mask = (indices == -1) | (scores <= np.finfo(np.float32).min / 2)
        scores[missing_mask] = -np.inf

        result_scores = np.full(
            (query_array.shape[0], int(k)), -np.inf, dtype=np.float32
        )
        result_indices = np.full((query_array.shape[0], int(k)), -1, dtype=np.int64)
        result_scores[:, :result_count] = scores
        result_indices[:, :result_count] = indices
        return result_scores, result_indices

    def save(self, path: str | Path) -> None:
        """Save the FAISS index to disk.

        Args:
            path: Destination path for the FAISS index file.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self._index is None:
            raise RuntimeError("The index must be built before saving.")
        import faiss

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(output_path))
        metadata_path = Path(f"{output_path}.json")
        metadata = {
            "index_type": self._index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
        }
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")
        logger.info("Saved FAISS index to %s.", output_path)

    def load(self, path: str | Path) -> None:
        """Load a serialized FAISS index and restore its vector dimension.

        Args:
            path: Source path for a FAISS index file.
        """
        import faiss

        index_path = Path(path)
        self._index = faiss.read_index(str(index_path))
        self._dimension = int(self._index.d)
        self._quantizer = getattr(self._index, "quantizer", None)
        if hasattr(self._index, "nlist"):
            self._index_type = "ivf"
            self.nlist = int(self._index.nlist)
            self.nprobe = int(self._index.nprobe)
        else:
            self._index_type = "flat"

        metadata_path = Path(f"{index_path}.json")
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            if metadata.get("index_type") == "ivf" and self._index_type == "ivf":
                self.nlist = int(metadata.get("nlist", self.nlist))
                restored_nprobe = int(metadata.get("nprobe", self._index.nprobe))
                self.set_nprobe(restored_nprobe)
        logger.info(
            "Loaded FAISS index with %d vectors and dimension %d.",
            self.size,
            self._dimension,
        )

    @property
    def size(self) -> int:
        """Return the number of vectors stored in the FAISS index."""
        return 0 if self._index is None else int(self._index.ntotal)
