"""Retriever node — query a single domain's Chroma collection."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from ingestion.embedder import HFInferenceEmbedder
from rag_pipeline.state import GraphState, RetrievedChunk
from vectorstore import Domain, get_manager

DEFAULT_TOP_K = 5


@lru_cache(maxsize=1)
def _embedder() -> HFInferenceEmbedder:
    return HFInferenceEmbedder()


def retrieve(state: GraphState) -> GraphState:
    question = state["question"]
    domain = Domain(state["domain"])
    top_k = state.get("top_k", DEFAULT_TOP_K)

    collection = get_manager().get(domain)
    query_embedding = _embedder().embed_query(question)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    retrieved: List[RetrievedChunk] = []
    for i, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        retrieved.append(
            {
                "rank": i,
                "text": doc,
                "metadata": meta or {},
                "distance": float(dist),
            }
        )

    return {"retrieved": retrieved}
