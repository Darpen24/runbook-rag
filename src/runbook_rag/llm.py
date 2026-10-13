from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from .embeddings import _terms


class ExtractiveStubModel(BaseChatModel):
    """Not a language model. Returns the context sentences that best match the question."""

    @property
    def _llm_type(self) -> str:
        return "extractive-stub"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        question = re.search(r"Question:\n(.*)\Z", prompt, flags=re.S)
        question_terms = set(_terms(question.group(1))) if question else set()
        context = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, flags=re.S)
        scored = []
        for block in re.findall(r"\[(\d+)\] \(.*?\)\n(.*?)(?=\n\n\[|\Z)", context.group(1) if context else "", flags=re.S):
            number, text = block
            for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
                sentence = sentence.strip()
                if len(sentence) > 15 and not sentence.startswith("#"):
                    overlap = len(question_terms & set(_terms(sentence)))
                    scored.append((overlap, number, sentence))
        scored.sort(key=lambda item: -item[0])
        best = [s for s in scored[:2] if s[0] > 0] or scored[:1]
        content = " ".join(f"{sentence} [{number}]" for _, number, sentence in best) or "No context."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def build_llm(provider: str, model: str | None) -> Runnable:
    provider = provider.lower()
    if provider == "stub":
        return ExtractiveStubModel()
    if not model:
        raise ValueError(f"Set LLM_MODEL for provider '{provider}'.")
    from langchain.chat_models import init_chat_model  # needs the provider package

    return init_chat_model(model, model_provider=provider, temperature=0)
