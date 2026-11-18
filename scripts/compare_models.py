"""Compare embedding models on the fixed retrieval evaluation set."""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.encoder import EmbeddingEncoder
from src.evaluator import Evaluator
from src.factory import build_engine

if TYPE_CHECKING:
    from src.engine import RecommendationEngine

logger = logging.getLogger(__name__)
_DEFAULT_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_DEFAULT_ITEMS = Path("data/processed/items.csv")
_DEFAULT_EVAL = Path("data/processed/eval_queries.csv")
_DEFAULT_INDEX = "faiss-flat"
_DEFAULT_OUT = Path("results/model_comparison.csv")
_OUTPUT_COLUMNS = [
    "model",
    "dim",
    "embed_time_s",
    "recall@1",
    "recall@3",
    "recall@10",
    "mrr",
    "avg_latency_ms",
]


def _embedding_cache_path(model_name: str) -> Path:
    """Return the standard processed embedding path for a model name.

    Args:
        model_name: SentenceTransformers model identifier or local path.

    Returns:
        The conventional model-slug cache path.
    """
    slug = model_name.split("/")[-1]
    if not slug:
        raise ValueError("Model name must end with a non-empty model slug.")
    return Path("data/processed") / f"embeddings_{slug}.npy"


def _encode_uncached_model(
    model_name: str,
    texts: list[str],
) -> tuple[Path, float, int, tempfile.TemporaryDirectory[str]]:
    """Encode uncached item texts into a temporary embedding file.

    Args:
        model_name: Model identifier to encode with.
        texts: Processed item text values.

    Returns:
        Temporary embedding path, encoding seconds, dimension, and its owner.
    """
    temporary_directory = tempfile.TemporaryDirectory(prefix="model-embeddings-")
    embedding_path = Path(temporary_directory.name) / "embeddings.npy"
    encoder = EmbeddingEncoder(model_name=model_name)
    start_time = time.perf_counter()
    embeddings = encoder.encode(texts, show_progress=True)
    embed_time_s = time.perf_counter() - start_time
    EmbeddingEncoder.save_embeddings(embeddings, embedding_path)
    return embedding_path, embed_time_s, int(embeddings.shape[1]), temporary_directory


def _evaluate_one_model(
    model_name: str,
    items: pd.DataFrame,
    eval_df: pd.DataFrame,
    index_type: str,
) -> dict[str, str | int | float]:
    """Encode or load one model and return its measured retrieval row.

    Args:
        model_name: SentenceTransformers model identifier or local path.
        items: Processed catalog data.
        eval_df: Evaluation queries and pipe-separated relevant IDs.
        index_type: Index name accepted by ``build_engine``.

    Returns:
        Comparison metrics and embedding metadata for the selected model.
    """
    cache_path = _embedding_cache_path(model_name)
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if cache_path.is_file():
        embeddings_path = cache_path
        embeddings = EmbeddingEncoder.load_embeddings(cache_path)
        dimension = int(embeddings.shape[1])
        embed_time_s = float("nan")
        print(f"Embedding cache exists for {model_name}; embed_time_s is NaN.")
    else:
        texts = items["text"].tolist()
        (
            embeddings_path,
            embed_time_s,
            dimension,
            temporary_directory,
        ) = _encode_uncached_model(model_name, texts)

    try:
        engine: RecommendationEngine = build_engine(
            items_path=_DEFAULT_ITEMS,
            embeddings_path=embeddings_path,
            model_name=model_name,
            index_type=index_type,
        )
        engine.recommend(str(eval_df["query"].iloc[0]), k=10)
        metrics = Evaluator(engine, eval_df).run_all((1, 3, 10))
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    return {
        "model": model_name,
        "dim": dimension,
        "embed_time_s": embed_time_s,
        "recall@1": float(metrics["recall@1"]),
        "recall@3": float(metrics["recall@3"]),
        "recall@10": float(metrics["recall@10"]),
        "mrr": float(metrics["mrr"]),
        "avg_latency_ms": float(metrics["avg_latency_ms"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Compare configured embedding models and save a CSV summary.

    Args:
        argv: Optional command-line arguments, used by tests.

    Returns:
        Zero if at least one model was evaluated, otherwise one.
    """
    parser = argparse.ArgumentParser(description="Compare embedding models.")
    parser.add_argument("--models", nargs="+", default=_DEFAULT_MODELS)
    parser.add_argument("--index", default=_DEFAULT_INDEX)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--items", type=Path, default=_DEFAULT_ITEMS)
    parser.add_argument("--eval", type=Path, default=_DEFAULT_EVAL)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        items = pd.read_csv(args.items)
        eval_df = pd.read_csv(
            args.eval,
            dtype={"query": "string", "relevant_ids": "string"},
        )
        if "text" not in items.columns:
            raise ValueError("Items CSV must contain a 'text' column.")
    except (OSError, ValueError) as error:
        logger.error("Could not load comparison inputs: %s", error)
        return 1

    rows: list[dict[str, str | int | float]] = []
    for model_name in args.models:
        try:
            rows.append(_evaluate_one_model(model_name, items, eval_df, args.index))
        except Exception:
            logger.exception("Model comparison failed for %s; continuing.", model_name)

    if not rows:
        logger.error("No models completed successfully; no comparison file written.")
        return 1

    results = pd.DataFrame.from_records(rows, columns=_OUTPUT_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    logger.info("Saved model comparison to %s.", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
