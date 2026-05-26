"""Ingest documents from a directory into a domain's namespace.

Idempotent: chunk IDs are content-hashed, so re-running the same files
overwrites instead of duplicating.

Usage:
    python -m ingestion.ingest_pipeline --domain hr       --path ./docs/hr
    python -m ingestion.ingest_pipeline --domain research --path ./docs/research --reset
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from ingestion.chunker import get_chunker
from ingestion.embedder import HFInferenceEmbedder
from ingestion.loaders import load_directory
from vectorstore import Domain, get_manager

CHROMA_METADATA_PRIMITIVES = (str, int, float, bool)


def stable_id(text: str, source: str, index: int) -> str:
    """Deterministic chunk ID — same content+source+position → same ID."""
    digest = hashlib.sha256(f"{source}::{index}::{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def _flatten_metadata(meta: dict) -> dict:
    """Chroma only stores primitive metadata values; drop the rest silently."""
    return {k: v for k, v in meta.items() if isinstance(v, CHROMA_METADATA_PRIMITIVES)}


def ingest(domain: Domain, path: Path, reset: bool = False) -> int:
    manager = get_manager()

    if reset:
        print(f"Resetting namespace: {manager.collection_name(domain)}")
        manager.reset(domain)

    collection = manager.get_or_create(domain)

    print(f"Loading from: {path}")
    docs = load_directory(path)
    if not docs:
        print(f"  No supported files found in {path}")
        return 0
    print(f"  Loaded {len(docs)} document(s)")

    embedder = HFInferenceEmbedder()
    chunker = get_chunker(domain, embedder)
    print(f"Chunking with: {type(chunker).__name__}")
    chunks = chunker.split_documents(docs)
    print(f"  Produced {len(chunks)} chunks")

    print(f"Embedding {len(chunks)} chunks via HF Inference API "
          f"({embedder.config.hf_embed_model})")
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)

    ids = [
        stable_id(c.page_content, c.metadata.get("source", ""), i)
        for i, c in enumerate(chunks)
    ]
    metadatas = [_flatten_metadata(c.metadata) for c in chunks]

    print(f"Upserting into {manager.collection_name(domain)}")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    final_count = collection.count()
    print(f"  Collection now contains {final_count} chunk(s)")
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents into a DocuMind knowledge namespace.",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=[d.value for d in Domain],
        help="Knowledge domain to ingest into.",
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="Directory containing PDF / DOCX / MD / TXT / HTML files.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the namespace before ingesting.",
    )
    args = parser.parse_args()

    try:
        ingest(Domain(args.domain), args.path, reset=args.reset)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
