"""Cross-encoder re-ranker.

Cross-encoders score (query, passage) pairs jointly, which is far more
accurate than the bi-encoder cosine-similarity used at retrieval time, but
too slow to run over the full corpus. Reranking the top-N retrieval hits is
the standard production pattern.

The model loads lazily on first use, so users who only use single-domain
mode without rerank don't pay the import cost or model download.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_pipeline.state import GraphState, RetrievedChunk


class RerankerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # BAAI/bge-reranker-base — ~278M params, multilingual (German + English), CPU-friendly.
    # Swap to BAAI/bge-reranker-v2-m3 for higher quality at ~568M params.
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True
    reranker_top_k: int = 5  # keep top-K after rerank; 0 = keep all


@lru_cache(maxsize=1)
def _cross_encoder():
    """Lazy-load the cross-encoder. Heavy import, kept out of module top-level."""
    from sentence_transformers import CrossEncoder

    config = RerankerConfig()
    return CrossEncoder(config.reranker_model, max_length=512)


def rerank(state: GraphState) -> GraphState:
    config = RerankerConfig()
    retrieved = state.get("retrieved", [])

    if not retrieved or not config.reranker_enabled:
        return {}  # no-op — keep existing retrieved untouched

    question = state["question"]
    pairs = [(question, chunk["text"]) for chunk in retrieved]
    scores = _cross_encoder().predict(pairs)

    indexed = list(zip(scores, retrieved))
    indexed.sort(key=lambda x: x[0], reverse=True)

    keep = config.reranker_top_k if config.reranker_top_k > 0 else len(indexed)
    reranked: List[RetrievedChunk] = []
    for new_rank, (score, chunk) in enumerate(indexed[:keep], start=1):
        new_chunk = dict(chunk)
        new_chunk["rank"] = new_rank
        # Stash the score in metadata so the UI can show "reranker confidence".
        meta = dict(new_chunk.get("metadata") or {})
        meta["rerank_score"] = float(score)
        new_chunk["metadata"] = meta
        reranked.append(new_chunk)

    return {"retrieved": reranked}
