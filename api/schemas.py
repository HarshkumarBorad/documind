"""Pydantic request/response models for the DocuMind REST API.

These mirror the LangGraph state shape but are serialization-friendly and carry
explicit field descriptions that surface in the auto-generated OpenAPI docs.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from vectorstore import Domain


# ----- Query -----

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The natural-language question.")
    llm_name: Optional[str] = Field(
        None,
        description="Override the default chat model. See GET /models for valid IDs.",
    )
    llm_provider: Optional[str] = Field(
        None,
        description="HF Inference Provider override (auto / together / fireworks-ai / ...).",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve. For federated queries, this is per-domain.",
    )


class SourceCitation(BaseModel):
    n: int = Field(..., description="Citation number that appears as [N] in the answer text.")
    filename: str
    page: int = Field(..., description="0-indexed page number; -1 for non-paginated formats.")
    source: str
    text: str
    domain: str = Field("", description="Origin domain — populated only for federated queries.")


class QueryResponse(BaseModel):
    mode: Literal["single", "federated"]
    domain: Optional[str] = Field(None, description="The queried domain — null for federated.")
    model: str
    question: str
    answer: str
    sources: List[SourceCitation]
    retrieved_count: int = Field(..., description="Chunks reaching synthesis (post-rerank).")


# ----- Ingest -----

class IngestRequest(BaseModel):
    domain: Domain
    path: str = Field(
        ...,
        description="Server-local directory containing the documents to ingest.",
    )
    reset: bool = Field(False, description="Wipe the namespace before ingesting.")


class IngestResponse(BaseModel):
    domain: str
    chunks_added: int = Field(..., description="Number of chunks added in this ingest call.")
    total_chunks: int = Field(..., description="Total chunks in the namespace after ingest.")


# ----- System -----

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    chroma: str = Field(..., description="ChromaDB connection status / error message.")
    namespaces: Dict[str, int] = Field(
        default_factory=dict,
        description="Chunk counts per namespace.",
    )


class ModelsResponse(BaseModel):
    default: str
    supported: List[str]


# ----- Evaluation -----

class EvaluationRequest(BaseModel):
    domain: Domain
    model: Optional[str] = Field(
        None,
        description="LLM used to generate answers. Defaults to backend's DEFAULT_LLM.",
    )
    judge_model: Optional[str] = Field(
        None,
        description="LLM used by RAGAS to score the answers. Defaults to `model`.",
    )
    max_queries: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="Cap evaluated queries. RAGAS makes many LLM calls per query.",
    )


class PerQueryResult(BaseModel):
    question: str
    answer: str
    context_count: int
    ground_truth: Optional[str] = None
    # RAGAS metric columns get added dynamically — allow extra fields.
    model_config = {"extra": "allow"}


class EvaluationResponse(BaseModel):
    domain: str
    model: str
    judge_model: str
    has_ground_truths: bool
    metrics_run: List[str]
    overall: Dict[str, float]
    per_query: List[Dict]  # kept loose so dynamic metric fields survive


class EvaluationQueriesResponse(BaseModel):
    queries: Dict[str, List[Dict]]
