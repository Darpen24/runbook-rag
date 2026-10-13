"""Embeddings behind the standard LangChain interface.

HashingEmbeddings is an offline stand in: it hashes words and word pairs into a
fixed size vector. It matches shared words, not meaning, so paraphrases score
worse than they would with a real embedding model. Use it for tests and demos.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "my", "i", "it",
    "with", "from", "this", "that", "have", "has", "can", "do", "does", "me", "we", "be", "how",
    "what", "when", "which", "if", "at", "by", "as", "you", "your", "should", "will", "must",
}


def _terms(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class HashingEmbeddings(Embeddings):
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for term in _terms(text):
            digest = hashlib.md5(term.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_embeddings(provider: str, model: str | None = None) -> Embeddings:
    provider = provider.lower()
    if provider == "hashing":
        return HashingEmbeddings()
    if provider == "openai":
        if not model:
            raise ValueError("Set EMBEDDINGS_MODEL for the openai provider.")
        from langchain_openai import OpenAIEmbeddings  # needs langchain-openai

        return OpenAIEmbeddings(model=model)
    raise ValueError(f"Unknown embeddings provider: {provider}")
