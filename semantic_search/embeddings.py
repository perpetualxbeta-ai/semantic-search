"""
Embeddings: turning text into vectors.

An "embedding" is just a list of numbers (a vector) that represents the
meaning of a piece of text. Texts with similar meaning end up with vectors
that point in similar directions — that's the whole trick behind semantic search.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from . import config


class ModelEmbedder:
    """Real semantic embeddings from a small pre-trained neural network."""

    def __init__(self, model_name: str = config.MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True scales every vector to length 1.
        # With unit-length vectors, cosine similarity == dot product.
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class ToyEmbedder:
    """
    A "fake" embedder: hashes each word into one of `dim` buckets and counts them.

    It needs no download and is great for seeing the mechanics — but it only
    matches shared WORDS, not shared MEANING. Search "car" and it will NOT find
    "automobile". Try the same query with the real model to feel the difference.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


@lru_cache(maxsize=1)
def get_embedder(kind: str = config.EMBEDDER):
    if kind == "toy":
        return ToyEmbedder()
    if kind == "model":
        return ModelEmbedder()
    raise ValueError(f"Unknown EMBEDDER {kind!r} (use 'model' or 'toy')")


def embed(texts: list[str]) -> np.ndarray:
    return get_embedder().embed(texts)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
