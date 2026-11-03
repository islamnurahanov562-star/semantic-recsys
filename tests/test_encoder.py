"""Tests for sentence embedding generation and persistence."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.encoder import EmbeddingEncoder


class FakeSentenceTransformer:
    """Provide deterministic normalized vectors without loading model weights."""

    instances: list[FakeSentenceTransformer] = []

    def __init__(self, model_name: str, device: str | None = None) -> None:
        """Record each fake model instance and its configuration."""
        self.model_name = model_name
        self.device = device
        self.encoded_batches: list[list[str]] = []
        self.encode_options: list[dict[str, object]] = []
        type(self).instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        """Return the fixed dimension used by the tests."""
        return 8

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        """Return seeded, normalized float32 vectors for the input texts."""
        self.encoded_batches.append(list(texts))
        self.encode_options.append(kwargs)
        vectors = np.random.default_rng(42).random((len(texts), 8), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


@pytest.fixture
def fake_sentence_transformer() -> Iterator[type[FakeSentenceTransformer]]:
    """Patch the model constructor so fast tests never download weights."""
    FakeSentenceTransformer.instances = []
    with patch("sentence_transformers.SentenceTransformer", FakeSentenceTransformer):
        yield FakeSentenceTransformer


def test_encode_returns_expected_shape_and_float32(
    fake_sentence_transformer: type[FakeSentenceTransformer],
) -> None:
    """Encoding returns one eight-dimensional float32 vector per text."""
    encoder = EmbeddingEncoder(device="cpu")

    embeddings = encoder.encode(["first text", "second text"])

    assert embeddings.shape == (2, 8)
    assert embeddings.dtype == np.float32
    assert embeddings.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_encode_empty_list_returns_empty_matrix(
    fake_sentence_transformer: type[FakeSentenceTransformer],
) -> None:
    """An empty input returns an empty matrix with the model dimension."""
    encoder = EmbeddingEncoder(device="cpu")

    embeddings = encoder.encode([])

    assert embeddings.shape == (0, 8)
    assert embeddings.dtype == np.float32


def test_encode_rejects_a_string_instead_of_a_sequence(
    fake_sentence_transformer: type[FakeSentenceTransformer],
) -> None:
    """A bare string is not accepted as a sequence of texts."""
    encoder = EmbeddingEncoder(device="cpu")

    with pytest.raises(TypeError, match="sequence of strings"):
        encoder.encode("one string")


def test_encode_rejects_non_string_sequence_items(
    fake_sentence_transformer: type[FakeSentenceTransformer],
) -> None:
    """Every element in the input sequence must be a string."""
    encoder = EmbeddingEncoder(device="cpu")

    with pytest.raises(TypeError, match=r"texts\[1\]"):
        encoder.encode(["valid text", 42])


def test_encode_query_strips_text_and_rejects_blank_queries(
    fake_sentence_transformer: type[FakeSentenceTransformer],
) -> None:
    """A query is stripped and encoded as one vector; blanks are rejected."""
    encoder = EmbeddingEncoder(device="cpu")

    embeddings = encoder.encode_query("  a short query  ")

    assert embeddings.shape == (1, 8)
    assert fake_sentence_transformer.instances[0].encoded_batches == [["a short query"]]
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        encoder.encode_query("  \t ")


def test_model_is_loaded_once_and_cached(
    fake_sentence_transformer: type[FakeSentenceTransformer],
) -> None:
    """The model is loaded lazily and reused across encoder calls."""
    encoder = EmbeddingEncoder(device="cpu")
    assert fake_sentence_transformer.instances == []

    encoder.encode(["first call"])
    encoder.encode_query("second call")
    encoder.encode([])

    assert len(fake_sentence_transformer.instances) == 1


def test_embedding_save_and_load_round_trip(tmp_path: Path) -> None:
    """Saving creates missing folders and loading restores the exact matrix."""
    embeddings = np.random.default_rng(42).random((3, 8), dtype=np.float32)
    path = tmp_path / "new" / "nested" / "embeddings.npy"

    EmbeddingEncoder.save_embeddings(embeddings, path)
    loaded_embeddings = EmbeddingEncoder.load_embeddings(path)

    assert path.is_file()
    assert loaded_embeddings.dtype == np.float32
    np.testing.assert_array_equal(loaded_embeddings, embeddings)


@pytest.mark.slow
def test_real_model_semantic_similarity_and_normalization() -> None:
    """The real model ranks related sentences closer and normalizes vectors."""
    encoder = EmbeddingEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    embeddings = encoder.encode(
        [
            "a cat sat on the mat",
            "a kitten rests on a rug",
            "stock market crashed today",
        ]
    )

    related_similarity = float(embeddings[0] @ embeddings[1])
    unrelated_similarity = float(embeddings[0] @ embeddings[2])
    norms = np.linalg.norm(embeddings, axis=1)

    assert related_similarity > unrelated_similarity
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
