"""Build and cache normalized embeddings for the processed item dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)
_DEFAULT_ITEMS = Path("data/processed/items.csv")
_OUTPUT_DIR = Path("data/processed")


def _texts_sha256(texts: list[str]) -> str:
    """Hash the ordered texts joined with newline separators.

    Args:
        texts: Item texts in dataset row order.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """
    joined_texts = "\n".join(texts)
    return hashlib.sha256(joined_texts.encode("utf-8")).hexdigest()


def _metadata_matches(meta_path: Path, texts_sha256: str) -> bool:
    """Return whether existing metadata describes the current text content.

    Args:
        meta_path: Metadata JSON file path.
        texts_sha256: Digest of the current ordered texts.

    Returns:
        True when readable metadata contains the matching digest.
    """
    try:
        with meta_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read embedding metadata from %s.", meta_path)
        return False
    return metadata.get("sha256") == texts_sha256


def _validate_embeddings(embeddings: np.ndarray, item_count: int) -> None:
    """Check embedding shape, row count, and L2 normalization.

    Args:
        embeddings: Encoded item vectors.
        item_count: Number of input item records.

    Raises:
        ValueError: If embeddings do not match the data contract.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array; got {embeddings.shape}.")
    if embeddings.shape[0] != item_count:
        raise ValueError(
            f"Embedding row count ({embeddings.shape[0]}) does not match "
            f"items row count ({item_count})."
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError("All embedding vectors must have L2 norm within 1e-3 of 1.")


def _write_metadata(
    meta_path: Path,
    model_name: str,
    item_count: int,
    embeddings: np.ndarray,
    texts_sha256: str,
    batch_size: int,
) -> None:
    """Write embedding metadata as UTF-8 JSON.

    Args:
        meta_path: Destination JSON path.
        model_name: SentenceTransformers model identifier.
        item_count: Number of embedded items.
        embeddings: Encoded item matrix.
        texts_sha256: Digest of the ordered input texts.
        batch_size: Encoding batch size.
    """
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "n_items": item_count,
        "dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sha256": texts_sha256,
        "batch_size": batch_size,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Encode item texts and cache the embedding matrix and metadata.

    Args:
        argv: Optional command-line arguments, used for testing.

    Returns:
        Zero on success, or one when input or embedding validation fails.
    """
    parser = argparse.ArgumentParser(description="Build item text embeddings.")
    parser.add_argument("--items", type=Path, default=_DEFAULT_ITEMS)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model name or path.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Rebuild cached vectors.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.batch_size < 1:
        logger.error("--batch-size must be at least one.")
        return 1

    slug = args.model.split("/")[-1]
    if not slug:
        logger.error("--model must end with a non-empty model name.")
        return 1

    embeddings_path = _OUTPUT_DIR / f"embeddings_{slug}.npy"
    metadata_path = _OUTPUT_DIR / f"embeddings_{slug}.meta.json"

    try:
        items = pd.read_csv(args.items)
        if "text" not in items.columns:
            raise ValueError("Items CSV must contain a 'text' column.")
        if items.empty:
            raise ValueError("Items CSV contains no rows to encode.")

        texts = items["text"].tolist()
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("Every value in items['text'] must be a string.")
        texts_sha256 = _texts_sha256(texts)

        if (
            not args.force
            and embeddings_path.is_file()
            and metadata_path.is_file()
            and _metadata_matches(metadata_path, texts_sha256)
        ):
            print("up to date")
            return 0

        start_time = time.perf_counter()
        encoder = EmbeddingEncoder(model_name=args.model, batch_size=args.batch_size)
        embeddings = encoder.encode(texts, show_progress=True)
        _validate_embeddings(embeddings, len(items))
        EmbeddingEncoder.save_embeddings(embeddings, embeddings_path)
        _write_metadata(
            metadata_path,
            args.model,
            len(items),
            embeddings,
            texts_sha256,
            args.batch_size,
        )
        elapsed_seconds = time.perf_counter() - start_time
    except (OSError, TypeError, ValueError) as error:
        logger.error("Embedding build failed: %s", error)
        return 1

    file_size_mb = embeddings_path.stat().st_size / 1_000_000
    print(
        f"Embeddings built in {elapsed_seconds:.2f} seconds; "
        f"file size: {file_size_mb:.2f} MB."
    )
    logger.info("Saved embedding metadata to %s.", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
