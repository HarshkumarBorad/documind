"""In-process drop-in for `APIClient`.

Same shape as `APIClient`, but instead of HTTP calls it dispatches directly to
the LangGraph pipeline and ingest functions. Used by deployments where you
can't run a separate FastAPI service alongside Streamlit — most notably
HuggingFace Spaces, which only exposes one container port.

Switching the UI is a one-env-var change: set `DOCUMIND_MODE=local` and the UI
picks this instead of `APIClient`. See `ui/app.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from api_client import APIError  # raise the same exception type the UI catches


class LocalClient:
    """Duck-typed equivalent of `APIClient` that calls the graph directly."""

    base_url = "(in-process)"

    # ----- system -----

    def health(self) -> dict:
        from vectorstore import Domain, get_manager

        manager = get_manager()
        try:
            manager.heartbeat()
        except Exception as exc:
            return {"status": "error", "chroma": f"unreachable: {exc}", "namespaces": {}}
        try:
            counts = {d.value: manager.get_or_create(d).count() for d in Domain}
            return {"status": "ok", "chroma": "connected (embedded)", "namespaces": counts}
        except Exception as exc:
            return {
                "status": "degraded",
                "chroma": f"reachable but failing: {exc}",
                "namespaces": {},
            }

    def models(self) -> dict:
        from rag_pipeline.llm import DEFAULT_LLM, SUPPORTED_LLMS

        return {"default": DEFAULT_LLM, "supported": SUPPORTED_LLMS}

    def namespaces(self) -> dict:
        from vectorstore import Domain, get_manager

        manager = get_manager()
        return {d.value: manager.get_or_create(d).count() for d in Domain}

    # ----- query -----

    def _invoke_graph(self, graph_input: dict, mode: str, domain: Optional[str], model: str) -> dict:
        from rag_pipeline.graph import get_graph

        try:
            result = get_graph().invoke(graph_input)
        except Exception as exc:
            raise APIError(f"Pipeline error: {exc}") from exc
        sources = [
            # SourceCitation is a TypedDict-ish dict already — copy keys we want.
            {
                "n": s.get("n"),
                "filename": s.get("filename", "unknown"),
                "page": s.get("page", -1),
                "source": s.get("source", ""),
                "text": s.get("text", ""),
                "domain": s.get("domain", ""),
            }
            for s in result.get("sources", [])
        ]
        return {
            "mode": mode,
            "domain": domain,
            "model": model,
            "question": graph_input["question"],
            "answer": result.get("answer", ""),
            "sources": sources,
            "retrieved_count": len(result.get("retrieved", [])),
        }

    def query_single(
        self,
        domain: str,
        question: str,
        model: str,
        top_k: int = 5,
        provider: Optional[str] = None,
    ) -> dict:
        graph_input: dict = {
            "question": question,
            "query_mode": "single",
            "domain": domain,
            "llm_name": model,
            "top_k": top_k,
        }
        if provider:
            graph_input["llm_provider"] = provider
        return self._invoke_graph(graph_input, mode="single", domain=domain, model=model)

    def query_federated(
        self,
        question: str,
        model: str,
        top_k: int = 5,
        provider: Optional[str] = None,
    ) -> dict:
        graph_input: dict = {
            "question": question,
            "query_mode": "federated",
            "llm_name": model,
            "top_k_per_domain": top_k,
        }
        if provider:
            graph_input["llm_provider"] = provider
        return self._invoke_graph(graph_input, mode="federated", domain=None, model=model)

    # ----- ingest -----

    def ingest(self, domain: str, path: str, reset: bool = False) -> dict:
        from ingestion.ingest_pipeline import ingest as run_ingest
        from vectorstore import Domain, get_manager

        p = Path(path)
        if not p.exists():
            raise APIError(f"Path not found: {path}")
        if not p.is_dir():
            raise APIError(f"Not a directory: {path}")

        try:
            domain_enum = Domain(domain)
            chunks_added = run_ingest(domain_enum, p, reset=reset)
        except Exception as exc:
            raise APIError(f"Ingest failed: {exc}") from exc
        total = get_manager().get_or_create(domain_enum).count()
        return {
            "domain": domain_enum.value,
            "chunks_added": chunks_added,
            "total_chunks": total,
        }

    # ----- evaluation -----

    def evaluation_queries(self) -> dict:
        from evaluation.eval_pipeline import load_test_queries

        return {"queries": load_test_queries()}

    def run_evaluation(
        self,
        domain: str,
        model: Optional[str] = None,
        judge_model: Optional[str] = None,
        max_queries: Optional[int] = None,
    ) -> dict:
        from evaluation.eval_pipeline import evaluate_domain
        from rag_pipeline.llm import DEFAULT_LLM
        from vectorstore import Domain

        try:
            result = evaluate_domain(
                domain=Domain(domain),
                model=model or DEFAULT_LLM,
                judge_model=judge_model,
                max_queries=max_queries,
            )
        except Exception as exc:
            raise APIError(f"Evaluation failed: {exc}") from exc
        if "error" in result:
            raise APIError(result["error"])
        return result
