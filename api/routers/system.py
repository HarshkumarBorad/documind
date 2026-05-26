"""System endpoints — health, model registry, namespace inventory."""
from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse, ModelsResponse
from rag_pipeline.llm import DEFAULT_LLM, SUPPORTED_LLMS
from vectorstore import Domain, get_manager

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    manager = get_manager()
    try:
        manager.heartbeat()
    except Exception as exc:
        return HealthResponse(status="error", chroma=f"unreachable: {exc}", namespaces={})

    try:
        counts = {d.value: manager.get_or_create(d).count() for d in Domain}
        return HealthResponse(status="ok", chroma="connected", namespaces=counts)
    except Exception as exc:
        return HealthResponse(status="degraded", chroma=f"reachable but failing: {exc}", namespaces={})


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(default=DEFAULT_LLM, supported=SUPPORTED_LLMS)


@router.get("/namespaces")
def namespaces() -> dict:
    manager = get_manager()
    return {d.value: manager.get_or_create(d).count() for d in Domain}
