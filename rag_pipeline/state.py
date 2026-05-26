"""LangGraph state schema.

Kept as plain TypedDicts so the graph state is trivially serializable —
no clients, no Pydantic models, no closures live in here. That matters
when the pipeline is later invoked across process boundaries (FastAPI, MCP).
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class RetrievedChunk(TypedDict):
    """One hit returned by the retriever, ranked 1..N to match [N] citations."""

    rank: int
    text: str
    metadata: Dict[str, Any]
    distance: float


class CitedSource(TypedDict):
    """A retrieved chunk that the LLM actually cited.

    `page` is 0-indexed when the source loader provides one, or -1 (NO_PAGE)
    for non-paginated formats like .txt / .md / .html. See citation_formatter.
    `domain` is "" for single-domain queries; populated for federated queries
    so the UI/CLI can show which knowledge space each citation came from.
    """

    n: int
    filename: str
    page: int
    source: str
    text: str
    domain: str


class GraphState(TypedDict, total=False):
    # --- Input ---
    question: str
    domain: str               # single-mode only
    query_mode: str           # "single" (default) or "federated"
    llm_name: str
    llm_provider: str         # HF Inference Provider override; absent = env default
    top_k: int                # single-mode: total results; federated: per-domain results
    top_k_per_domain: int     # federated-mode override; falls back to top_k

    # --- Populated during execution ---
    retrieved: List[RetrievedChunk]
    raw_answer: str
    answer: str
    sources: List[CitedSource]
