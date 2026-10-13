"""Retrieve, build a prompt with numbered sources, answer and cite."""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .index import FaissIndex

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context. Cite sources like [1].
If the context does not contain the answer, say you could not find it.

Context:
{context}

Question:
{question}"""
)

NOT_FOUND = "I could not find this in the documents."


@dataclass
class Answer:
    text: str
    grounded: bool
    top_similarity: float
    sources: list[dict] = field(default_factory=list)


def answer_question(question: str, index: FaissIndex, llm: Runnable, k: int = 3, min_similarity: float = 0.12) -> Answer:
    hits = index.search(question, k=k)
    top = hits[0][1] if hits else 0.0
    if not hits or top < min_similarity:
        # Out of scope guard: do not ask the model when nothing relevant was retrieved.
        return Answer(text=NOT_FOUND, grounded=False, top_similarity=round(top, 3))

    context = "\n\n".join(
        f"[{number}] ({chunk.metadata['chunk_id']})\n{chunk.page_content}"
        for number, (chunk, _) in enumerate(hits, start=1)
    )
    text = (PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})
    sources = [
        {"n": number, "chunk_id": chunk.metadata["chunk_id"], "title": chunk.metadata["title"], "similarity": round(score, 3)}
        for number, (chunk, score) in enumerate(hits, start=1)
    ]
    return Answer(text=text, grounded=True, top_similarity=round(top, 3), sources=sources)
