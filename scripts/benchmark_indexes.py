"""Benchmark exact and approximate indexes on reproducible synthetic vectors."""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from src.index import FaissIndex, NumpyIndex

logger = logging.getLogger(__name__)
_DEFAULT_OUT = Path("results/benchmark_indexes.csv")
_OUTPUT_COLUMNS = [
    "n",
    "index",
    "build_s",
    "search_ms_per_query",
    "memory_mb",
    "recall_vs_flat",
]


def _normalized_vectors(
    generator: np.random.Generator,
    rows: int,
    dimensions: int,
) -> np.ndarray:
    """Create reproducible synthetic float32 vectors with unit L2 norms.

    Args:
        generator: Seeded random number generator.
        rows: Number of vectors to create.
        dimensions: Vector dimensionality.

    Returns:
        C-contiguous normalized float32 vectors.
    """
    vectors = generator.standard_normal((rows, dimensions), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _measure_search(
    index: NumpyIndex | FaissIndex,
    queries: np.ndarray,
    k: int,
    repeats: int,
) -> float:
    """Measure median search latency per query after one warm-up batch.

    Args:
        index: Built index to measure.
        queries: Query vectors searched as one batch.
        k: Number of neighbors requested per query.
        repeats: Number of timed batch searches.

    Returns:
        Median elapsed milliseconds per query.
    """
    index.search(queries, k)
    timings_ms_per_query: list[float] = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        index.search(queries, k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        timings_ms_per_query.append(elapsed_ms / len(queries))
    return median(timings_ms_per_query)


def _recall_at_k(reference: np.ndarray, candidate: np.ndarray, k: int) -> float:
    """Compute mean overlap recall against exact Flat top-k IDs.

    Args:
        reference: Exact top-k IDs, one row per query.
        candidate: Candidate top-k IDs, one row per query.
        k: Number of neighbors in each result row.

    Returns:
        Mean fraction of exact Flat neighbors recovered by the candidate.
    """
    recalls = [
        len(set(reference_row).intersection(candidate_row)) / k
        for reference_row, candidate_row in zip(reference, candidate)
    ]
    return float(np.mean(recalls))


def _benchmark_size(
    n: int,
    dimensions: int,
    query_count: int,
    k: int,
    repeats: int,
) -> list[dict[str, int | float | str]]:
    """Benchmark all three index implementations for one synthetic size.

    Args:
        n: Number of synthetic catalog vectors.
        dimensions: Vector dimensionality.
        query_count: Number of synthetic query vectors.
        k: Neighbors requested per query.
        repeats: Timed search batches after warm-up.

    Returns:
        One result row for each of NumPy, FAISS Flat, and FAISS IVF.
    """
    generator = np.random.default_rng(42)
    # These are synthetic vectors for an index microbenchmark, not catalog embeddings.
    vectors = _normalized_vectors(generator, n, dimensions)
    queries = _normalized_vectors(generator, query_count, dimensions)
    memory_mb = float(vectors.nbytes / 1_000_000)

    flat_index = FaissIndex(index_type="flat")
    start_time = time.perf_counter()
    flat_index.build(vectors)
    flat_build_s = time.perf_counter() - start_time
    _, flat_ids = flat_index.search(queries, k)
    flat_search_ms = _measure_search(flat_index, queries, k, repeats)

    numpy_index = NumpyIndex()
    start_time = time.perf_counter()
    numpy_index.build(vectors)
    numpy_build_s = time.perf_counter() - start_time
    _, numpy_ids = numpy_index.search(queries, k)
    numpy_search_ms = _measure_search(numpy_index, queries, k, repeats)

    ivf_index = FaissIndex(
        index_type="ivf",
        nlist=int(4 * np.sqrt(n)),
        nprobe=10,
    )
    start_time = time.perf_counter()
    ivf_index.build(vectors)
    ivf_build_s = time.perf_counter() - start_time
    _, ivf_ids = ivf_index.search(queries, k)
    ivf_search_ms = _measure_search(ivf_index, queries, k, repeats)

    return [
        {
            "n": n,
            "index": "numpy",
            "build_s": numpy_build_s,
            "search_ms_per_query": numpy_search_ms,
            "memory_mb": memory_mb,
            "recall_vs_flat": _recall_at_k(flat_ids, numpy_ids, k),
        },
        {
            "n": n,
            "index": "faiss-flat",
            "build_s": flat_build_s,
            "search_ms_per_query": flat_search_ms,
            "memory_mb": memory_mb,
            "recall_vs_flat": 1.0,
        },
        {
            "n": n,
            "index": "faiss-ivf",
            "build_s": ivf_build_s,
            "search_ms_per_query": ivf_search_ms,
            "memory_mb": memory_mb,
            "recall_vs_flat": _recall_at_k(flat_ids, ivf_ids, k),
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Benchmark index build/search time and save a CSV summary.

    Args:
        argv: Optional command-line arguments, used by tests.

    Returns:
        Zero on success, or one if arguments are invalid or no size fits memory.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark indexes on synthetic normalized vectors."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(1000, 10000, 100000))
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if any(size < 1 for size in args.sizes):
        parser.error("--sizes values must be positive")
    if args.dim < 1 or args.queries < 1 or args.k < 1 or args.repeats < 1:
        parser.error("--dim, --queries, --k, and --repeats must be positive")
    if args.k > min(args.sizes):
        parser.error("--k must not exceed the smallest requested size")

    rows: list[dict[str, int | float | str]] = []
    for size in args.sizes:
        try:
            rows.extend(
                _benchmark_size(size, args.dim, args.queries, args.k, args.repeats)
            )
        except MemoryError:
            warnings.warn(
                f"Skipping synthetic size {size}: insufficient memory.",
                RuntimeWarning,
                stacklevel=1,
            )
            logger.warning("Skipped size %d after MemoryError.", size)

    if not rows:
        logger.error("No benchmark sizes completed successfully.")
        return 1

    results = pd.DataFrame.from_records(rows, columns=_OUTPUT_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print("Synthetic vectors benchmark (not catalog embeddings)")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    logger.info("Saved synthetic index benchmark to %s.", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
