# Research namespace

This namespace is for **academic papers** — arXiv PDFs on RAG, LLMs, vector search, and related ML topics.

Drop your PDFs into this directory, then ingest them:

```cmd
python -m ingestion.ingest_pipeline --domain research --path .\docs\research
```

The research domain uses the **SemanticChunker** rather than the recursive splitter used by the other three namespaces — academic prose benefits from similarity-aware boundaries.

## Suggested starter papers

These public arXiv papers exercise the research namespace well. Download them with `curl` or your browser:

| Paper | arXiv ID | URL |
|---|---|---|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) | 2005.11401 | https://arxiv.org/pdf/2005.11401.pdf |
| BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity (Chen et al., 2024) | 2402.03216 | https://arxiv.org/pdf/2402.03216.pdf |
| Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023) | 2307.03172 | https://arxiv.org/pdf/2307.03172.pdf |
| Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023) | 2310.11511 | https://arxiv.org/pdf/2310.11511.pdf |

Fetch them all with one command:

```cmd
curl -o lewis2020-rag.pdf https://arxiv.org/pdf/2005.11401.pdf
curl -o chen2024-bgem3.pdf https://arxiv.org/pdf/2402.03216.pdf
curl -o liu2023-lost-middle.pdf https://arxiv.org/pdf/2307.03172.pdf
curl -o asai2023-selfrag.pdf https://arxiv.org/pdf/2310.11511.pdf
```

These four cover the conceptual ground that the test queries in `evaluation/test_queries.json` probe (RAG, cross-encoder reranking, bi-encoder vs cross-encoder trade-offs).
