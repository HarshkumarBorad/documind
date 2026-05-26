"""Initialize the four DocuMind knowledge-space collections.

Usage:
    python -m vectorstore.init_namespaces

Idempotent — safe to run repeatedly. Fails fast if ChromaDB is unreachable.
"""
from __future__ import annotations

import sys

from vectorstore.chroma_client import Domain, get_manager


def main() -> int:
    manager = get_manager()

    try:
        manager.heartbeat()
    except Exception as exc:
        print(
            f"ERROR: cannot reach ChromaDB at "
            f"{manager.config.chroma_host}:{manager.config.chroma_port}\n"
            f"  {exc}\n\n"
            f"Start it with:  docker compose up -d chromadb",
            file=sys.stderr,
        )
        return 1

    print(f"Connected to ChromaDB at {manager.config.chroma_host}:{manager.config.chroma_port}\n")

    for domain in Domain:
        collection = manager.get_or_create(domain)
        print(f"  [ok] {collection.name:<24} count={collection.count()}  ({domain.description})")

    print(f"\nAll {len(list(Domain))} namespaces ready.")
    print(f"Existing collections on server: {manager.list_namespaces()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
