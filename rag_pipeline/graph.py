"""LangGraph wiring.

    START
      ↓
    classify
      ↓ (conditional)
      ├─ single ────→ retrieve ─────────┐
      └─ federated ─→ retrieve_federated ┤
                                          ↓
                                       rerank
                                          ↓
                                      synthesize
                                          ↓
                                    format_citations
                                          ↓
                                         END
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from rag_pipeline.nodes.citation_formatter import format_citations
from rag_pipeline.nodes.classifier import classify
from rag_pipeline.nodes.federated_retriever import retrieve_federated
from rag_pipeline.nodes.reranker import rerank
from rag_pipeline.nodes.retriever import retrieve
from rag_pipeline.nodes.synthesizer import synthesize
from rag_pipeline.state import GraphState


def _route_after_classify(state: GraphState) -> str:
    return "retrieve_federated" if state.get("query_mode") == "federated" else "retrieve"


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("classify", classify)
    builder.add_node("retrieve", retrieve)
    builder.add_node("retrieve_federated", retrieve_federated)
    builder.add_node("rerank", rerank)
    builder.add_node("synthesize", synthesize)
    builder.add_node("format_citations", format_citations)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "retrieve": "retrieve",
            "retrieve_federated": "retrieve_federated",
        },
    )
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("retrieve_federated", "rerank")
    builder.add_edge("rerank", "synthesize")
    builder.add_edge("synthesize", "format_citations")
    builder.add_edge("format_citations", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Process-wide compiled graph — the compile step is non-trivial, cache it."""
    return build_graph()
