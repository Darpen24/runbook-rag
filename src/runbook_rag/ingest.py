"""Load markdown files, split them into overlapping chunks, embed and index them."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .embeddings import build_embeddings
from .index import FaissIndex


def load_documents(doc_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(doc_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        documents.append(Document(page_content=text, metadata={"source": path.name, "title": title}))
    return documents


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}#{counters[source]}"
    return chunks


def build_index(settings: Settings) -> FaissIndex:
    documents = load_documents(settings.doc_dir)
    if not documents:
        raise FileNotFoundError(f"No markdown files found in {settings.doc_dir}")
    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    embeddings = build_embeddings(settings.embeddings_provider, settings.embeddings_model)
    index = FaissIndex.build(chunks, embeddings)
    index.save(settings.index_dir)
    return index


def load_index(settings: Settings) -> FaissIndex:
    embeddings = build_embeddings(settings.embeddings_provider, settings.embeddings_model)
    return FaissIndex.load(settings.index_dir, embeddings)
