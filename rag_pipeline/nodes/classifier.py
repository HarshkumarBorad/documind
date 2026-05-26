"""Query classifier — validates inputs and decides single vs federated routing.

Deterministic for now: the caller passes `query_mode` explicitly. A future
revision can swap this for an LLM-based intent classifier that auto-routes
ambiguous questions to the federated path.
"""
from __future__ import annotations

from rag_pipeline.state import GraphState
from vectorstore import Domain

VALID_MODES = {"single", "federated"}


def classify(state: GraphState) -> GraphState:
    mode = state.get("query_mode", "single")
    if mode not in VALID_MODES:
        raise ValueError(
            f"query_mode must be one of {sorted(VALID_MODES)}; got {mode!r}"
        )

    if mode == "single":
        domain_value = state.get("domain")
        if not domain_value:
            raise ValueError("Single-domain mode requires `domain` in state.")
        # Surface invalid domain names early instead of failing inside ChromaDB.
        Domain(domain_value)

    return {"query_mode": mode}
