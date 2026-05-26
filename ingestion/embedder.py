"""HuggingFace Inference API embedder, langchain-compatible.

Implements the LangChain `Embeddings` interface so it can be passed to
`SemanticChunker`, retrievers, and anything else expecting that contract.
Batches requests to keep individual payloads under the HF serverless limit.
"""
from __future__ import annotations

from typing import List, Optional

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedderConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Optional[str] (not `str | None`) — pydantic evaluates field annotations at
    # class creation and the `|` union syntax needs Python 3.10+.
    hf_token: Optional[str] = None
    hf_embed_model: str = "BAAI/bge-m3"
    hf_embed_batch_size: int = 32


class HFInferenceEmbedder(Embeddings):
    """LangChain Embeddings backed by HuggingFace Inference API."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        if not self.config.hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. Add it to .env or your environment.\n"
                "Get a token (free tier works) at https://huggingface.co/settings/tokens"
            )
        self.client = InferenceClient(
            model=self.config.hf_embed_model,
            token=self.config.hf_token,
        )

    def _to_list(self, raw) -> List[List[float]]:
        # huggingface_hub usually returns numpy.ndarray; sometimes a list.
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], (int, float)):
            return [list(map(float, raw))]
        return [list(map(float, row)) for row in raw]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        bs = self.config.hf_embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            raw = self.client.feature_extraction(batch)
            results.extend(self._to_list(raw))
        return results

    def embed_query(self, text: str) -> List[float]:
        raw = self.client.feature_extraction(text)
        vectors = self._to_list(raw)
        return vectors[0]
