"""Ingest endpoint — kicks off document ingestion for a single namespace.

Runs synchronously. For large corpora a production deploy would queue this as
a background job; for the portfolio demo, sync is fine and the HTTP request
will simply take however long the embedding API calls take.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import IngestRequest, IngestResponse
from ingestion.ingest_pipeline import ingest
from vectorstore import get_manager

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest all documents from a server-local directory into a namespace",
)
def ingest_documents(body: IngestRequest) -> IngestResponse:
    path = Path(body.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {body.path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {body.path}")

    try:
        chunks_added = ingest(body.domain, path, reset=body.reset)
    except RuntimeError as exc:
        # E.g. HF_TOKEN missing — caller's fault, return 400.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc

    total_chunks = get_manager().get_or_create(body.domain).count()
    return IngestResponse(
        domain=body.domain.value,
        chunks_added=chunks_added,
        total_chunks=total_chunks,
    )
