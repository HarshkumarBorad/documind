"""Per-domain chunking strategy.

HR / TECH / PRODUCT use a recursive character splitter — fast, deterministic,
works well for structured documents like handbooks, API references, manuals.

RESEARCH uses semantic chunking — splits on similarity drops between adjacent
sentences. Better for dense academic prose where paragraphs encode one idea.
Costs more embedding calls during chunking; worth it for the precision boost.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import Domain

RECURSIVE_CHUNK_SIZE = 800
RECURSIVE_CHUNK_OVERLAP = 100
SEMANTIC_BREAKPOINT_PERCENTILE = 95


def get_chunker(domain: Domain, embedder: Embeddings):
    """Return the text splitter configured for `domain`.

    The embedder is only consumed by the semantic chunker; the recursive
    splitter ignores it. Passing it unconditionally keeps the caller simple.
    """
    if domain == Domain.RESEARCH:
        return SemanticChunker(
            embeddings=embedder,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_PERCENTILE,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
