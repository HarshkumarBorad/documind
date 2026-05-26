"""LangChain ChatModel that bridges DocuMind's HF Inference chat code into RAGAS.

RAGAS expects langchain-compatible LLMs and embeddings. The embeddings side is
already covered by `HFInferenceEmbedder` (Phase 2). For the LLM side, this
wrapper plugs our existing `chat()` call into the BaseChatModel interface so
RAGAS can use it as the judge model — same HF_TOKEN, same provider routing,
no second API key needed.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from rag_pipeline.llm import DEFAULT_LLM, chat


def _langchain_to_dicts(messages: List[BaseMessage]) -> List[dict]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in messages
    ]


class HFInferenceChatLLM(BaseChatModel):
    """Wraps DocuMind's `chat()` function as a langchain BaseChatModel.

    Used by RAGAS as the judge LLM. Both sync and async paths are implemented
    so RAGAS's internal async evaluation loop doesn't block on each call.
    """

    model: str = Field(default=DEFAULT_LLM)
    provider: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.0)  # deterministic judging
    max_tokens: int = Field(default=512)

    @property
    def _llm_type(self) -> str:
        return "hf-inference-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = chat(
            model=self.model,
            messages=_langchain_to_dicts(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            provider=self.provider,
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # `chat()` makes a blocking HTTP call — push it to a thread so we
        # don't stall RAGAS's asyncio event loop while it batches metrics.
        text = await asyncio.to_thread(
            chat,
            model=self.model,
            messages=_langchain_to_dicts(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            provider=self.provider,
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )
