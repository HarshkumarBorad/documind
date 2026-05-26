"""ChromaDB namespace manager.

Each of the four knowledge domains lives in its own isolated Chroma collection.
The single source of truth for domain names is the `Domain` enum — every other
layer (ingestion, retrieval, MCP server) imports from here.
"""
from __future__ import annotations

import os

# Must be set BEFORE chromadb is imported — its telemetry singleton reads
# this env var once at import time.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from enum import Enum
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from pydantic_settings import BaseSettings, SettingsConfigDict

# Belt and suspenders: chromadb-client 0.5.x ships a PostHog version with
# a `capture()` signature mismatch that prints a warning on every client init
# even when telemetry is "disabled". Monkey-patch the offending method.
try:
    from chromadb.telemetry.product.posthog import Posthog  # noqa: E402

    Posthog.capture = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass


class Domain(str, Enum):
    HR = "hr"
    TECH = "tech"
    RESEARCH = "research"
    PRODUCT = "product"

    @property
    def description(self) -> str:
        return _DOMAIN_DESCRIPTIONS[self]


_DOMAIN_DESCRIPTIONS: dict[Domain, str] = {
    Domain.HR: "HR policies, onboarding handbooks, and code of conduct",
    Domain.TECH: "Technical documentation, API references, and architecture decision records",
    Domain.RESEARCH: "Academic research papers (arXiv) on LLMs, RAG, and vector search",
    Domain.PRODUCT: "Product manuals, release notes, and customer-facing FAQs",
}


class ChromaConfig(BaseSettings):
    """Configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    documind_namespace_prefix: str = "documind"


class NamespaceManager:
    """Thin wrapper around the Chroma HTTP client that enforces one collection per domain."""

    def __init__(self, config: ChromaConfig | None = None) -> None:
        self.config = config or ChromaConfig()
        self.client = chromadb.HttpClient(
            host=self.config.chroma_host,
            port=self.config.chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )

    def collection_name(self, domain: Domain) -> str:
        return f"{self.config.documind_namespace_prefix}_{domain.value}"

    def get_or_create(self, domain: Domain) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name(domain),
            metadata={
                "domain": domain.value,
                "description": domain.description,
                "hnsw:space": "cosine",
            },
        )

    def get(self, domain: Domain) -> Collection:
        return self.client.get_collection(name=self.collection_name(domain))

    def reset(self, domain: Domain) -> None:
        """Drop and recreate the collection for a domain. Destructive."""
        name = self.collection_name(domain)
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass
        self.get_or_create(domain)

    def list_namespaces(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    def heartbeat(self) -> int:
        """Returns the server's nanosecond heartbeat — raises if unreachable."""
        return self.client.heartbeat()


@lru_cache(maxsize=1)
def get_manager() -> NamespaceManager:
    """Process-wide singleton. Cheap because the HTTP client is stateless."""
    return NamespaceManager()
