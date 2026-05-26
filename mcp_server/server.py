"""DocuMind MCP server.

Exposes each of the four knowledge namespaces as a named MCP tool, plus a
`federated_query` tool that searches all of them simultaneously. The tools are
the bridge to the next project (multi-agent system): each agent picks the
right domain tool for its task, or uses federated_query when the question
spans multiple domains.

The server reuses the same compiled LangGraph as the FastAPI layer — keep the
orchestration in one place, expose it through multiple front-ends.

Run over stdio (default — for Claude Desktop / Cline / Cursor / other MCP clients):
    python -m mcp_server.server

Run over Streamable HTTP for remote agents:
    python -m mcp_server.server --transport http --port 8002

Run over SSE (legacy MCP transport):
    python -m mcp_server.server --transport sse --port 8002
"""
from __future__ import annotations

import argparse

from fastmcp import FastMCP

from rag_pipeline.graph import get_graph
from rag_pipeline.llm import DEFAULT_LLM
from vectorstore import Domain, get_manager

mcp = FastMCP(
    name="DocuMind",
    instructions=(
        "RAG over four isolated knowledge namespaces: HR policies, technical "
        "documentation, research papers, and product documentation. Use the "
        "specific domain tool when you already know which namespace holds the "
        "answer. Use federated_query when the question spans multiple domains "
        "or you don't know which namespace to pick. Call list_namespaces "
        "first if you need to introspect what's available before choosing."
    ),
)

DEFAULT_TOP_K = 5


def _query_single(domain: Domain, question: str, top_k: int = DEFAULT_TOP_K) -> str:
    result = get_graph().invoke(
        {
            "question": question,
            "query_mode": "single",
            "domain": domain.value,
            "llm_name": DEFAULT_LLM,
            "top_k": top_k,
        }
    )
    return result.get("answer", "(no answer)")


# --- Domain tools ---------------------------------------------------------

@mcp.tool()
def query_hr_knowledge(question: str) -> str:
    """Query the HR policies and onboarding knowledge base.

    Use for questions about: employee handbooks, leave policies, code of
    conduct, onboarding procedures, benefits, compensation, time-off, and
    similar internal HR topics. Returns the answer with inline [N] citations
    and a Sources list naming the source documents.
    """
    return _query_single(Domain.HR, question)


@mcp.tool()
def query_tech_docs(question: str) -> str:
    """Query the technical documentation knowledge base.

    Use for questions about: API references, SDK documentation, architecture
    decision records (ADRs), engineering guidelines, technical specifications,
    and integration guides. Returns the answer with inline [N] citations.
    """
    return _query_single(Domain.TECH, question)


@mcp.tool()
def query_research_papers(question: str) -> str:
    """Query the research papers knowledge base.

    Use for questions about academic research — LLMs, RAG, vector search,
    embeddings, attention mechanisms, and similar ML topics. The corpus
    consists of arXiv papers. Returns the answer with inline [N] citations.
    """
    return _query_single(Domain.RESEARCH, question)


@mcp.tool()
def query_product_knowledge(question: str) -> str:
    """Query the product manuals and customer-facing FAQ knowledge base.

    Use for questions about: product features, user manuals, release notes,
    customer-facing FAQs, troubleshooting guides, and how-to questions about
    the product itself. Returns the answer with inline [N] citations.
    """
    return _query_single(Domain.PRODUCT, question)


# --- Federated tool -------------------------------------------------------

@mcp.tool()
def federated_query(question: str, top_k_per_domain: int = DEFAULT_TOP_K) -> str:
    """Search ALL four knowledge bases simultaneously and return a unified answer.

    Use this when:
    - The question spans multiple domains (e.g., GDPR — touches HR + Tech + Product).
    - You don't know which specific namespace holds the answer.
    - You want the most comprehensive answer drawing on every namespace.

    Slower than single-domain tools because it queries all four collections in
    parallel and applies cross-encoder reranking on the merged results.
    Citations are tagged with their origin domain: `[1] [hr] handbook.pdf`.
    """
    result = get_graph().invoke(
        {
            "question": question,
            "query_mode": "federated",
            "llm_name": DEFAULT_LLM,
            "top_k_per_domain": top_k_per_domain,
        }
    )
    return result.get("answer", "(no answer)")


# --- Introspection tool ---------------------------------------------------

@mcp.tool()
def list_namespaces() -> str:
    """List the four available knowledge namespaces with their content
    descriptions and current chunk counts.

    Call this first if you don't know which domain tool to use, or if you
    want to confirm a namespace has content before querying it.
    """
    manager = get_manager()
    lines = []
    for d in Domain:
        try:
            count = manager.get_or_create(d).count()
        except Exception as exc:
            lines.append(f"- {d.value}: {d.description} (unavailable: {exc})")
            continue
        lines.append(f"- {d.value}: {d.description} ({count} chunks indexed)")
    return "\n".join(lines)


# --- Entrypoint -----------------------------------------------------------

_TRANSPORT_ALIASES = {
    "stdio": "stdio",
    "http": "streamable-http",  # the modern MCP HTTP transport
    "sse": "sse",                # legacy
}


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuMind MCP server.")
    parser.add_argument(
        "--transport",
        choices=list(_TRANSPORT_ALIASES.keys()),
        default="stdio",
        help="Transport mode. Default 'stdio' — used by Claude Desktop, Cline, Cursor, etc.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port for http / sse transport (default: 8002).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for http / sse transport (default: 127.0.0.1).",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(
            transport=_TRANSPORT_ALIASES[args.transport],
            host=args.host,
            port=args.port,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
