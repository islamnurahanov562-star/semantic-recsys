"""Run retrieval evaluation for one index and save reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluator import Evaluator
from src.factory import build_engine

logger = logging.getLogger(__name__)

_DEFAULT_EVALUATION = Path("data/processed/eval_queries.csv")
_DEFAULT_ITEMS = Path("data/processed/items.csv")
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEX_TYPES = ("numpy", "faiss-flat", "faiss-ivf")
_RESULTS_DIR = Path("results")


def _environment_versions() -> dict[str, str]:
    """Return runtime and dependency versions used by an evaluation run.

    Returns:
        Python, NumPy, pandas, SentenceTransformers, and FAISS versions.
    """
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "sentence_transformers": version("sentence-transformers"),
        "faiss": version("faiss-cpu"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate one index and save aggregate and per-query results.

    Args:
        argv: Optional command-line arguments, used by tests.

    Returns:
        Zero on success, or one if input loading or evaluation fails.
    """
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--index", choices=_INDEX_TYPES, default="faiss-flat")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--eval", type=Path, default=_DEFAULT_EVALUATION)
    parser.add_argument("--items", type=Path, default=_DEFAULT_ITEMS)
    parser.add_argument("--ks", type=int, nargs="+", default=(1, 3, 10))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        eval_df = pd.read_csv(
            args.eval,
            dtype={"query": "string", "relevant_ids": "string"},
        )
        missing_columns = {"query", "relevant_ids"}.difference(eval_df.columns)
        if missing_columns:
            raise ValueError(
                f"Evaluation CSV is missing columns: {sorted(missing_columns)}"
            )
        if eval_df.empty:
            raise ValueError("Evaluation CSV must contain at least one query.")

        engine = build_engine(
            items_path=args.items,
            model_name=args.model,
            index_type=args.index,
        )
        evaluator = Evaluator(engine, eval_df)
        engine.recommend(str(eval_df["query"].iloc[0]), k=10)
        metrics = evaluator.run_all(tuple(args.ks))
        per_query = evaluator.per_query()

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = _RESULTS_DIR / f"metrics_{args.index}.json"
        per_query_path = _RESULTS_DIR / f"per_query_{args.index}.csv"
        result: dict[str, Any] = {
            "model": args.model,
            "index": args.index,
            "n_items": len(engine),
            "n_queries": metrics["n_queries"],
            "latency_warmup_queries": 1,
            "versions": _environment_versions(),
            "date": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }

        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(result, metrics_file, ensure_ascii=False, indent=2)
            metrics_file.write("\n")
        per_query.to_csv(per_query_path, index=False, encoding="utf-8")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        logger.error("Evaluation failed: %s", error)
        return 1

    print(f"Index: {args.index} | Model: {args.model}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    logger.info("Saved metrics to %s.", metrics_path)
    logger.info("Saved per-query results to %s.", per_query_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
