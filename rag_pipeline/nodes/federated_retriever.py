"""Federated retriever — queries all four namespaces in parallel and merges results.

Each chunk's metadata is tagged with its origin domain so the citation formatter
can display "(hr)" / "(tech)" / etc. next to each source.

Ranks are assigned sequentially across the merged set BEFORE re-ranking; the
re-ranker downstream resorts and renumbers them.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from rag_pipeline.nodes.retriever import _embedder  # reuse the embedder singleton
from rag_pipeline.state import GraphState, RetrievedChunk
from vectorstore import Domain, get_manager

DEFAULT_TOP_K_PER_DOMAIN = 5


def _query_domain(domain: Domain, query_embedding: List[float], top_k: int) -> Tuple[Domain, dict]:
    """Query a single collection. Returns (domain, raw chroma result) or (domain, None) on error."""
    try:
        collection = get_manager().get(domain)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return domain, result
    except Exception:
        # Empty / missing namespace shouldn't kill the whole federated query.
        return domain, None


def retrieve_federated(state: GraphState) -> GraphState:
    question = state["question"]
    top_k = state.get("top_k_per_domain") or state.get("top_k") or DEFAULT_TOP_K_PER_DOMAIN

    query_embedding = _embedder().embed_query(question)

    # ChromaDB queries are I/O-bound (HTTP) — threads are the right concurrency primitive.
    with ThreadPoolExecutor(max_workers=len(Domain)) as executor:
        futures = [
            executor.submit(_query_domain, domain, query_embedding, top_k)
            for domain in Domain
        ]
        results = [f.result() for f in futures]

    retrieved: List[RetrievedChunk] = []
    next_rank = 1
    for domain, result in results:
        if not result:
            continue
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        for doc, meta, dist in zip(documents, metadatas, distances):
            meta_copy = dict(meta or {})
            meta_copy["domain"] = domain.value  # tag origin domain for downstream nodes
            retrieved.append(
                {
                    "rank": next_rank,
                    "text": doc,
                    "metadata": meta_copy,
                    "distance": float(dist),
                }
            )
            next_rank += 1

    return {"retrieved": retrieved}
