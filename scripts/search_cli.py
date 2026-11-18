"""Search the catalog from a command line or interactive prompt."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from src.factory import build_engine

if TYPE_CHECKING:
    from src.engine import RecommendationEngine

logger = logging.getLogger(__name__)
_INDEX_TYPES = ("numpy", "faiss-flat", "faiss-ivf")


def _search_and_print(
    engine: RecommendationEngine,
    query: str,
    k: int,
    category: str | None,
) -> int:
    """Run one query and print a compact result table and latency.

    Args:
        engine: Built recommendation engine.
        query: Natural-language search query.
        k: Maximum number of results to show.
        category: Optional exact category filter.

    Returns:
        Zero on success, or one if the query is invalid.
    """
    start_time = time.perf_counter()
    try:
        recommendations = engine.recommend(query, k=k, category=category)
    except ValueError as error:
        print(f"Invalid query: {error}")
        return 1
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    if recommendations.empty:
        print("No recommendations found.")
    else:
        display = recommendations.loc[
            :, ["rank", "score", "title", "category", "description"]
        ].copy()
        display["score"] = display["score"].map(lambda score: f"{score:.3f}")
        display["description"] = (
            display["description"].fillna("").astype("string").str.slice(0, 120)
        )
        print(display.to_string(index=False))
    print(f"Search time: {elapsed_ms:.2f} ms")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Build the search engine once and run one-shot or interactive queries.

    Args:
        argv: Optional command-line arguments, used by tests.

    Returns:
        Zero on successful searches or clean interactive exit, otherwise one.
    """
    parser = argparse.ArgumentParser(description="Search Netflix titles by meaning.")
    parser.add_argument("--query")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--index", choices=_INDEX_TYPES, default="faiss-flat")
    parser.add_argument("--category")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        engine = build_engine(model_name=args.model, index_type=args.index)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Could not build search engine: %s", error)
        print(f"Could not start search: {error}")
        return 1

    if args.query is not None:
        return _search_and_print(engine, args.query, args.k, args.category)

    while True:
        try:
            query = input("Query (empty or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.casefold() == "exit":
            break
        _search_and_print(engine, query, args.k, args.category)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
