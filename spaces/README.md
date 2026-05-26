---
title: DocuMind
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Multi-namespace federated RAG with cross-encoder reranking
models:
  - BAAI/bge-m3
  - BAAI/bge-reranker-base
  - Qwen/Qwen2.5-7B-Instruct
tags:
  - rag
  - retrieval-augmented-generation
  - langgraph
  - chromadb
  - model-context-protocol
  - streamlit
  - huggingface
  - ragas
---

# 📚 DocuMind

Multi-namespace federated RAG platform. Drop documents into one of four
isolated knowledge spaces (HR, Tech, Research, Product) and query them
individually or **federally across all of them**, with cross-encoder reranking
and inline `[N]` citations.

This Space runs the Streamlit UI talking to an embedded ChromaDB. The
backing pipeline is a LangGraph (`classify → retrieve | retrieve_federated →
rerank → synthesize → format_citations`) that uses HuggingFace Inference
Providers for embeddings and chat.

> 🛠️ Full source, FastAPI backend, MCP server, and Docker Compose setup:
> [github.com/HarshkumarBorad/documind](https://github.com/HarshkumarBorad/documind)

## Try it

The Space ships with sample documents already ingested. From the **🔍 Query**
tab try:

- *"What is the leave policy?"* (single domain — `hr`)
- *"How do I install Aurora on Windows?"* (single domain — `product`)
- *"What does the company say about authentication and rate limits?"* (federated)

## How it differs from the GitHub repo

| Aspect | GitHub repo (docker-compose) | This Space |
|---|---|---|
| ChromaDB | Separate service via HTTP | Embedded (PersistentClient) |
| FastAPI | Separate service on port 8001 | Bypassed — UI calls graph directly |
| MCP server | Separate service on port 8002 | Not included |
| Streamlit UI | Port 8501 | Port 7860 (HF Spaces standard) |

Same LangGraph pipeline under the hood.

## Configuration

You'll need a HuggingFace Inference token (free tier works). Add it as a
**Space secret** named `HF_TOKEN` — settings → "Variables and secrets" →
"New secret". The Space won't be able to query without it.
