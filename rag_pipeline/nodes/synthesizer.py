"""Synthesizer node — turn retrieved chunks into a cited answer via HF chat LLM."""
from __future__ import annotations

from typing import Iterable

from rag_pipeline.llm import LLMConfig, chat
from rag_pipeline.state import GraphState, RetrievedChunk

SYSTEM_PROMPT = """You are DocuMind, a precise question-answering assistant.

RULES:
- Answer using ONLY the information in the SOURCES section below.
- Cite the sources you used with inline markers [N] matching the source numbers.
- Multiple citations per claim are fine, e.g. "...as documented [1][3]."
- If the sources don't contain enough information to answer, reply exactly:
  "I don't have enough information in the provided sources to answer this."
- Never invent facts not present in the sources.
- Answer in the same language the user used in their question."""

NO_CONTEXT_REPLY = (
    "I don't have enough information in the provided sources to answer this."
)


def _location(meta: dict) -> str:
    filename = meta.get("filename", "unknown")
    page = meta.get("page")
    if page is not None:
        try:
            return f"{filename} (page {int(page) + 1})"
        except (TypeError, ValueError):
            pass
    return filename


def _format_sources(chunks: Iterable[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[{chunk['rank']}] {_location(chunk['metadata'])}:\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def synthesize(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", [])
    if not retrieved:
        return {"raw_answer": NO_CONTEXT_REPLY}

    config = LLMConfig()
    user_message = (
        f"SOURCES:\n{_format_sources(retrieved)}\n\nQUESTION: {state['question']}"
    )

    answer = chat(
        model=state["llm_name"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=config.hf_llm_max_tokens,
        temperature=config.hf_llm_temperature,
        provider=state.get("llm_provider"),
    )
    return {"raw_answer": answer}
