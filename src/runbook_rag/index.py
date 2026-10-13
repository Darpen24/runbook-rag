"""A tiny FAISS index that stores the chunks next to the vectors."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class FaissIndex:
    """Inner product on normalised vectors, so the score is cosine similarity."""

    def __init__(self, index: faiss.Index, chunks: list[Document], embeddings: Embeddings):
        self.index = index
        self.chunks = chunks
        self.embeddings = embeddings

    @classmethod
    def build(cls, chunks: list[Document], embeddings: Embeddings) -> "FaissIndex":
        vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]), dtype=np.float32)
        vectors = _normalise(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, embeddings)

    def search(self, question: str, k: int = 3) -> list[tuple[Document, float]]:
        query = np.array([self.embeddings.embed_query(question)], dtype=np.float32)
        scores, ids = self.index.search(_normalise(query), min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        payload = [{"text": c.page_content, "metadata": c.metadata} for c in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embeddings: Embeddings) -> "FaissIndex":
        index = faiss.read_index(str(directory / "vectors.faiss"))
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Document(page_content=p["text"], metadata=p["metadata"]) for p in payload]
        return cls(index, chunks, embeddings)
