"""Thin REST wrapper around the DocuMind FastAPI.

The UI deliberately goes through HTTP rather than calling the LangGraph in-process,
so it can be deployed separately from the backend if needed.
"""
from __future__ import annotations

import os
from typing import Optional

import requests


class APIError(Exception):
    """Raised when the API call fails or returns a non-2xx status."""


class APIClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get(
            "DOCUMIND_API_URL", "http://localhost:8001"
        )
        # Embedding + chat round-trips through HF can take a while.
        self.query_timeout = 180
        self.system_timeout = 10

    def _get(self, path: str, timeout: Optional[int] = None) -> dict:
        try:
            r = requests.get(
                f"{self.base_url}{path}", timeout=timeout or self.system_timeout
            )
        except requests.RequestException as exc:
            raise APIError(str(exc)) from exc
        if r.status_code >= 400:
            raise APIError(f"{r.status_code}: {r.text}")
        return r.json()

    def _post(self, path: str, json: dict, timeout: Optional[int] = None) -> dict:
        try:
            r = requests.post(
                f"{self.base_url}{path}",
                json=json,
                timeout=timeout or self.query_timeout,
            )
        except requests.RequestException as exc:
            raise APIError(str(exc)) from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise APIError(f"{r.status_code}: {detail}")
        return r.json()

    # ----- system -----

    def health(self) -> dict:
        return self._get("/health")

    def models(self) -> dict:
        return self._get("/models")

    def namespaces(self) -> dict:
        return self._get("/namespaces")

    # ----- query -----

    def query_single(
        self,
        domain: str,
        question: str,
        model: str,
        top_k: int = 5,
        provider: Optional[str] = None,
    ) -> dict:
        body: dict = {"question": question, "llm_name": model, "top_k": top_k}
        if provider:
            body["llm_provider"] = provider
        return self._post(f"/query/{domain}", body)

    def query_federated(
        self,
        question: str,
        model: str,
        top_k: int = 5,
        provider: Optional[str] = None,
    ) -> dict:
        body: dict = {"question": question, "llm_name": model, "top_k": top_k}
        if provider:
            body["llm_provider"] = provider
        return self._post("/query/federated", body)

    # ----- ingest -----

    def ingest(self, domain: str, path: str, reset: bool = False) -> dict:
        body = {"domain": domain, "path": path, "reset": reset}
        return self._post("/ingest", body, timeout=600)  # bulk ingest can be slow

    # ----- evaluation -----

    def evaluation_queries(self) -> dict:
        return self._get("/evaluation/queries")

    def run_evaluation(
        self,
        domain: str,
        model: Optional[str] = None,
        judge_model: Optional[str] = None,
        max_queries: Optional[int] = None,
    ) -> dict:
        body: dict = {"domain": domain}
        if model:
            body["model"] = model
        if judge_model:
            body["judge_model"] = judge_model
        if max_queries:
            body["max_queries"] = max_queries
        # RAGAS makes many sequential LLM calls — allow a generous timeout.
        return self._post("/evaluation/run", body, timeout=1800)
