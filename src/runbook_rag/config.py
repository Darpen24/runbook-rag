from __future__ import annotations

import os

from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "resources"


@dataclass
class Settings:
    embeddings_provider: str = "hashing"
    embeddings_model: str | None = None
    llm_provider: str = "stub"
    llm_model: str | None = None
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3
    min_similarity: float = 0.12
    doc_dir: Path = field(default_factory=lambda: ROOT / "data" / "docs")
    index_dir: Path = field(default_factory=lambda: Path.cwd() / "index")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(Path.cwd() / ".env", override=False)
        return cls(
            doc_dir=Path(os.getenv("DOC_DIR", str(ROOT / "data" / "docs"))),
            index_dir=Path(os.getenv("INDEX_DIR", str(Path.cwd() / "index"))),
            embeddings_provider=os.getenv("EMBEDDINGS_PROVIDER", "hashing"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL"),
            llm_provider=os.getenv("LLM_PROVIDER", "stub"),
            llm_model=os.getenv("LLM_MODEL"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
            top_k=int(os.getenv("TOP_K", "3")),
            min_similarity=float(os.getenv("MIN_SIMILARITY", "0.12")),
        )
