"""Citation formatter — scan `[N]` markers, build the Sources list, drop the rest."""
from __future__ import annotations

import re
from typing import List

from rag_pipeline.state import CitedSource, GraphState

CITE_PATTERN = re.compile(r"\[(\d+)\]")
NO_PAGE = -1  # sentinel: chunk doesn't come from a paginated document


def _extract_page(meta: dict) -> int:
    """Return the 0-indexed page from metadata, or NO_PAGE if absent / invalid."""
    page = meta.get("page")
    if page is None:
        return NO_PAGE
    try:
        return int(page)
    except (TypeError, ValueError):
        return NO_PAGE


def _format_source_line(s: CitedSource) -> str:
    domain_tag = f" [{s['domain']}]" if s.get("domain") else ""
    if s["page"] == NO_PAGE:
        return f"[{s['n']}]{domain_tag} {s['filename']}"
    # Display is 1-indexed: page 0 → "page 1".
    return f"[{s['n']}]{domain_tag} {s['filename']} (page {s['page'] + 1})"


def format_citations(state: GraphState) -> GraphState:
    raw = state.get("raw_answer", "")
    retrieved = state.get("retrieved", [])

    # Preserve order of first appearance, dedupe.
    seen: set = set()
    cited_ranks: List[int] = []
    for match in CITE_PATTERN.finditer(raw):
        n = int(match.group(1))
        if n not in seen:
            seen.add(n)
            cited_ranks.append(n)

    rank_to_chunk = {c["rank"]: c for c in retrieved}
    sources: List[CitedSource] = []
    for n in cited_ranks:
        chunk = rank_to_chunk.get(n)
        if chunk is None:
            continue  # LLM hallucinated a citation number — drop it.
        meta = chunk["metadata"]
        sources.append(
            {
                "n": n,
                "filename": meta.get("filename", "unknown"),
                "page": _extract_page(meta),
                "source": meta.get("source", ""),
                "text": chunk["text"],
                "domain": meta.get("domain", ""),
            }
        )

    if not sources:
        return {"answer": raw, "sources": []}

    lines = [_format_source_line(s) for s in sources]
    answer = raw.rstrip() + "\n\nSources:\n" + "\n".join(lines)
    return {"answer": answer, "sources": sources}
