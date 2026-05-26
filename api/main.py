"""DocuMind REST API.

Run locally:
    uvicorn api.main:app --reload --port 8001

Auto-generated docs:
    http://localhost:8001/docs       (Swagger UI)
    http://localhost:8001/redoc      (ReDoc)
    http://localhost:8001/openapi.json
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import evaluation, ingest, query, system

API_VERSION = "0.5.0"

app = FastAPI(
    title="DocuMind",
    description=(
        "Enterprise RAG platform with four isolated knowledge namespaces (HR, "
        "Tech, Research, Product) and a federated query mode that searches "
        "across all of them with cross-encoder re-ranking. The same pipeline "
        "is exposed as MCP tools in Phase 6 — this API and the MCP server "
        "share one LangGraph orchestration layer."
    ),
    version=API_VERSION,
)

# Permissive CORS for the Streamlit demo (Phase 7). Tighten for prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(evaluation.router)


@app.get("/", tags=["system"], summary="API metadata")
def root() -> dict:
    return {
        "name": "DocuMind",
        "version": API_VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
