"""Query endpoints — single-domain and federated.

Both endpoints share the same QueryRequest body; the URL path picks the mode.
Routes are declared in literal-before-parameterized order so that
`/query/federated` doesn't try to validate "federated" against the Domain enum.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse, SourceCitation
from rag_pipeline.graph import get_graph
from rag_pipeline.llm import DEFAULT_LLM
from vectorstore import Domain

router = APIRouter(prefix="/query", tags=["query"])


def _build_response(
    mode: str,
    model: str,
    question: str,
    result: dict,
    domain: str | None = None,
) -> QueryResponse:
    return QueryResponse(
        mode=mode,
        domain=domain,
        model=model,
        question=question,
        answer=result.get("answer", ""),
        sources=[SourceCitation(**s) for s in result.get("sources", [])],
        retrieved_count=len(result.get("retrieved", [])),
    )


@router.post(
    "/federated",
    response_model=QueryResponse,
    summary="Search all four namespaces in parallel, re-rank merged results",
)
def query_federated(body: QueryRequest) -> QueryResponse:
    model = body.llm_name or DEFAULT_LLM
    graph_input: dict = {
        "question": body.question,
        "query_mode": "federated",
        "llm_name": model,
        "top_k_per_domain": body.top_k,
    }
    if body.llm_provider:
        graph_input["llm_provider"] = body.llm_provider

    try:
        result = get_graph().invoke(graph_input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline error: {exc}") from exc
    return _build_response("federated", model, body.question, result)


@router.post(
    "/{domain}",
    response_model=QueryResponse,
    summary="Query a single knowledge namespace",
)
def query_single(domain: Domain, body: QueryRequest) -> QueryResponse:
    model = body.llm_name or DEFAULT_LLM
    graph_input: dict = {
        "question": body.question,
        "query_mode": "single",
        "domain": domain.value,
        "llm_name": model,
        "top_k": body.top_k,
    }
    if body.llm_provider:
        graph_input["llm_provider"] = body.llm_provider

    try:
        result = get_graph().invoke(graph_input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline error: {exc}") from exc
    return _build_response("single", model, body.question, result, domain=domain.value)
