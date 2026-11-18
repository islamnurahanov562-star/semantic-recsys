"""Evaluate a recommendation engine and save aggregate and per-query results."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluator import Evaluator
from src.factory import build_engine

logger = logging.getLogger(__name__)

_DEFAULT_QUERIES = Path("data/processed/eval_queries.csv")
_DEFAULT_RESULTS = Path("results")
_INDEX_TYPES = ("numpy", "faiss-flat", "faiss-ivf")
_ALL_INDEX_TYPES = ("all", *_INDEX_TYPES)


def main(argv: Sequence[str] | None = None) -> int:
    """Run retrieval evaluation and save JSON summary and per-query CSV.

    Args:
        argv: Optional command-line arguments, used by tests.

    Returns:
        Zero on success, or one when inputs or evaluation are invalid.
    """
    parser = argparse.ArgumentParser(description="Evaluate the recommendation engine.")
    parser.add_argument("--items", type=Path, default=Path("data/processed/items.csv"))
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--queries", type=Path, default=_DEFAULT_QUERIES)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index-type", choices=_ALL_INDEX_TYPES, default="all")
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        eval_df = pd.read_csv(
            args.queries,
            dtype={"query": "string", "relevant_ids": "string"},
        )
        index_types = _INDEX_TYPES if args.index_type == "all" else (args.index_type,)
        per_query_frames: list[pd.DataFrame] = []
        runs: dict[str, Any] = {}
        for index_type in index_types:
            engine = build_engine(
                items_path=args.items,
                embeddings_path=args.embeddings,
                model_name=args.model,
                index_type=index_type,
            )
            evaluator = Evaluator(engine, eval_df)
            engine.recommend(str(eval_df["query"].iloc[0]), k=10)
            logger.info("Warmed up %s engine before latency measurement.", index_type)
            metrics = evaluator.run_all()
            index_results = evaluator.per_query(k=10)
            index_results.insert(0, "index_type", index_type)
            per_query_frames.append(index_results)
            runs[index_type] = {"n_items": len(engine), "metrics": metrics}

        args.results_dir.mkdir(parents=True, exist_ok=True)
        per_query_path = args.results_dir / "retrieval_quality.csv"
        summary_path = args.results_dir / "eval_summary.json"
        per_query = pd.concat(per_query_frames, ignore_index=True)
        per_query.to_csv(per_query_path, index=False, encoding="utf-8")
        summary: dict[str, Any] = {
            "model": args.model,
            "latency_warmup_queries": 1,
            "runs": runs,
        }
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, ensure_ascii=False, indent=2)
            summary_file.write("\n")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        logger.error("Evaluation failed: %s", error)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Saved per-query results to %s.", per_query_path)
    logger.info("Saved evaluation summary to %s.", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
