"""HF chat-completion client + curated model registry.

`SUPPORTED_LLMS` is the canonical list of models the UI/CLI offers. Some are
gated on Hugging Face — users must accept the license on the model page once
before the Inference API will serve them.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient
from pydantic_settings import BaseSettings, SettingsConfigDict

# Curated list of chat-capable models with reliable HF Inference Provider
# coverage (verified mid-2026). The HF native serverless tier ("hf-inference")
# is text-generation only for most models — real chat completion is served via
# partner providers (Together, Fireworks, Sambanova, Nebius, etc.) reached
# through HF's router. Models marked ‡ are gated and need a one-time license
# acceptance at huggingface.co/<model>.
SUPPORTED_LLMS: List[str] = [
    "Qwen/Qwen2.5-7B-Instruct",                  # default — non-gated, Together+Fireworks+Hyperbolic
    "Qwen/Qwen2.5-72B-Instruct",                 # non-gated, larger
    "meta-llama/Llama-3.1-8B-Instruct",          # ‡ gated, very wide provider coverage
    "meta-llama/Llama-3.3-70B-Instruct",         # ‡ gated, frontier-tier
    "mistralai/Mistral-Nemo-Instruct-2407",      # non-gated, newer Mistral
    "microsoft/Phi-3.5-mini-instruct",           # non-gated, lightweight
    "deepseek-ai/DeepSeek-V4-Flash",             # non-gated MIT, 158B, fast — practical DeepSeek
    "deepseek-ai/DeepSeek-V4-Pro",               # non-gated MIT, 862B, frontier — burns credits fast
]

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    hf_token: Optional[str] = None
    hf_llm_model: str = DEFAULT_LLM
    # HF Inference Provider. "auto" = router picks any chat-capable provider.
    # `hf-inference` (HF's native serverless) no longer hosts the chat-completion
    # task for most popular models — it's text-generation only. Real chat is via
    # partner providers: "together", "fireworks-ai", "novita", "nebius", "sambanova".
    hf_llm_provider: str = "auto"
    hf_llm_max_tokens: int = 512
    hf_llm_temperature: float = 0.2


@lru_cache(maxsize=16)
def _client_for(model: str, provider: str) -> InferenceClient:
    """One client per (model, provider) pair, cached for the process lifetime."""
    config = LLMConfig()
    if not config.hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to .env or your environment."
        )
    return InferenceClient(model=model, token=config.hf_token, provider=provider)


def chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.2,
    provider: Optional[str] = None,
) -> str:
    """Send a chat-completion request and return the assistant's text reply.

    `provider` lets the caller override the env-configured default — useful for
    sidestepping a temporarily unhealthy provider for a specific request.
    """
    if provider is None:
        provider = LLMConfig().hf_llm_provider
    client = _client_for(model, provider)
    response = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content
