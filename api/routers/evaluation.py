"""Evaluation endpoints — RAGAS-driven scoring of the RAG pipeline."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import EvaluationRequest, EvaluationResponse, EvaluationQueriesResponse
from evaluation.eval_pipeline import evaluate_domain, load_test_queries
from rag_pipeline.llm import DEFAULT_LLM
from vectorstore import Domain

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get(
    "/queries",
    response_model=EvaluationQueriesResponse,
    summary="Return the test query set grouped by domain",
)
def get_test_queries() -> EvaluationQueriesResponse:
    return EvaluationQueriesResponse(queries=load_test_queries())


@router.post(
    "/run",
    response_model=EvaluationResponse,
    summary="Run RAGAS evaluation against the test query set for one domain",
)
def run_evaluation(body: EvaluationRequest) -> EvaluationResponse:
    try:
        result = evaluate_domain(
            domain=body.domain,
            model=body.model or DEFAULT_LLM,
            judge_model=body.judge_model,
            max_queries=body.max_queries,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return EvaluationResponse(**result)
