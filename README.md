# 📚 DocuMind — Enterprise RAG Platform

A multi-namespace, federated retrieval-augmented generation platform. Drop any
document corpus into one of four **isolated knowledge spaces** and query them
individually or **federally across all of them** — with cross-encoder reranking,
inline citations, RAGAS evaluation, and an MCP server that turns the whole
thing into a tool layer for multi-agent systems.

> Built end-to-end as a portfolio project — every layer is independently
> testable and was wired in nine deliberate phases (see [build phases](#build-phases)).

---

## ✨ Highlights

- **Four isolated knowledge namespaces** — HR, Tech, Research, Product — and a
  federated query mode that searches all of them simultaneously.
- **LangGraph orchestration** with a typed state, conditional routing, and
  individually testable nodes (`classify → retrieve | retrieve_federated →
  rerank → synthesize → format_citations`).
- **Cross-encoder reranking** with `BAAI/bge-reranker-base` for multilingual,
  precision-focused result ordering.
- **MCP server (FastMCP)** exposing each namespace as a named tool —
  drop-in memory layer for any LLM agent (Claude Desktop, Cline, Cursor, etc.).
- **REST API (FastAPI)** with auto-generated OpenAPI/Swagger docs.
- **Streamlit UI** with per-domain visual identity, drag-and-drop ingest, live
  health monitoring, and a RAGAS evaluation dashboard.
- **RAGAS evaluation** wired to the same HF Inference Providers stack —
  no second API key — with per-query metrics and bar-chart visualisation.
- **One-command full-stack via Docker Compose** — `docker compose up` boots
  ChromaDB, the FastAPI, the Streamlit UI, and the MCP server.

---

## 🎬 Demo

![DocuMind demo](docs/demo.gif)

> *Drop your `demo.gif` at `docs/demo.gif`. Suggested recording flow in the
> [Demo recording guide](#-demo-recording-guide) below.*

---

## 🚀 Quick demo with sample data (5 minutes)

Prereqs: Docker Desktop and Python 3.11+.

```cmd
:: 1. Configure your HuggingFace token (free tier works)
copy .env.example .env
::    Edit .env and set HF_TOKEN=<your token>
::    Get one at https://huggingface.co/settings/tokens

:: 2. Bring up the whole stack with Docker — 4 services in one command
docker compose up -d

:: 3. Create the four namespaces and ingest the bundled sample docs
docker compose exec api python -m vectorstore.init_namespaces
docker compose exec api python scripts/ingest_samples.py
```

Then open:

- **http://localhost:8501** — Streamlit UI
- **http://localhost:8001/docs** — Swagger API docs
- **http://localhost:8002/mcp** — MCP endpoint (Streamable HTTP)

Try the sample questions in the **🔍 Query** tab:

- *"What is the leave policy?"* — single-domain (`hr`)
- *"How do I install Aurora on Windows?"* — single-domain (`product`)
- *"What does the company say about authentication and rate limits?"* — federated

Each answer comes back with inline `[N]` citations and a Sources list naming
the exact document and page.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Streamlit UI                            │
│   🔍 Query   📥 Ingest   ⚡ Status   📊 Evaluation             │
└────────────────────────────┬─────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼─────────────────────────────────┐
│                       FastAPI Layer                           │
│   /query/{domain}    /query/federated                         │
│   /ingest            /evaluation/run                          │
│   /health            /models           /namespaces            │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                  LangGraph Pipeline                           │
│                                                                │
│   classify ─► (retrieve │ retrieve_federated) ─► rerank        │
│                       │                              │         │
│                       │                              ▼         │
│                       │                          synthesize    │
│                       │                              │         │
│                       │                              ▼         │
│                       │                      format_citations  │
└──────────┬─────────────────────────────────────────────┬──────┘
           │                                             │
┌──────────▼─────────┐                          ┌────────▼────────┐
│     ChromaDB       │                          │   MCP Server     │
│  4 isolated        │                          │  6 tools         │
│  namespaces        │                          │  (FastMCP)       │
└────────────────────┘                          └─────────────────┘
```

The **same compiled LangGraph** is the single source of truth for query
behaviour. The FastAPI, the MCP server, the CLI, and the Streamlit UI all
call `get_graph().invoke()` — zero duplication.

---

## 📂 Knowledge spaces

| Domain | Sample content | Chunking strategy |
|---|---|---|
| 🧑‍💼 **HR** | Onboarding handbook, leave policy, code of conduct | Recursive (800 / 100) |
| 🔧 **Tech** | API references, SDK docs, architecture decision records | Recursive (800 / 100) |
| 🔬 **Research** | arXiv papers on RAG / LLMs / vector search | Semantic (95th percentile) |
| 📦 **Product** | Installation guide, release notes, customer FAQ | Recursive (800 / 100) |

Each namespace is a separate ChromaDB collection — `documind_hr`,
`documind_tech`, etc. Cross-namespace contamination is impossible by
construction.

---

## 🛠️ Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Stateful pipeline with conditional routing and typed state |
| Vector store | **ChromaDB** (one collection per namespace) | Hosted in Docker; HTTP client only — lightweight install |
| Embeddings | **BGE-M3** via HF Inference API | Multilingual (German + English), strong benchmarks |
| Reranker | **BGE-reranker-base** (local, sentence-transformers) | Cross-encoder precision; runs on CPU |
| Chat LLM | **HF Inference Providers** (Qwen / Llama / DeepSeek / Mistral / Phi) | One token, multiple providers, runtime model swap |
| Ingestion | **LangChain** loaders + custom chunker per domain | Stable PDF / DOCX / MD / HTML support |
| API | **FastAPI** + Uvicorn | OpenAPI for free, async-ready |
| MCP server | **FastMCP** | Standard agentic tool protocol |
| Frontend | **Streamlit** | Native Python, fast iteration, polished out of the box |
| Evaluation | **RAGAS** | Industry-standard RAG metrics |
| Containers | **Docker Compose** | Single-command full-stack |

---

## 📁 Project structure

```
documind/
├── docker-compose.yml          ← 4 services: chromadb, api, ui, mcp
├── Dockerfile                  ← single image, per-service CMD override
├── .env.example                ← all configurable knobs
├── requirements.txt            ← phased — see comments for what each block adds
│
├── vectorstore/                ← Phase 1: namespace manager + Chroma client
│   ├── chroma_client.py
│   └── init_namespaces.py
│
├── ingestion/                  ← Phase 2: loaders, chunker, embedder
│   ├── loaders.py
│   ├── chunker.py
│   ├── embedder.py
│   └── ingest_pipeline.py      ← CLI: --domain X --path Y
│
├── rag_pipeline/               ← Phases 3 & 4: LangGraph pipeline
│   ├── graph.py                ← graph wiring + lru-cached compile
│   ├── state.py                ← typed GraphState / CitedSource
│   ├── llm.py                  ← HF chat client + model registry
│   ├── query.py                ← CLI for single + federated queries
│   └── nodes/
│       ├── classifier.py       ← validates input, decides routing
│       ├── retriever.py
│       ├── federated_retriever.py  ← parallel across all 4 namespaces
│       ├── reranker.py         ← cross-encoder, lazy-loaded
│       ├── synthesizer.py
│       └── citation_formatter.py
│
├── api/                        ← Phase 5: FastAPI REST layer
│   ├── main.py
│   ├── schemas.py
│   └── routers/
│       ├── system.py           ← /health /models /namespaces
│       ├── query.py            ← /query/{domain} /query/federated
│       ├── ingest.py           ← /ingest
│       └── evaluation.py       ← /evaluation/queries /evaluation/run
│
├── mcp_server/                 ← Phase 6: 6 MCP tools
│   └── server.py
│
├── ui/                         ← Phase 7: Streamlit
│   ├── app.py                  ← 4-tab UI
│   ├── api_client.py           ← thin requests wrapper
│   └── styles.py               ← per-domain colors + icons + CSS
│
├── evaluation/                 ← Phase 8: RAGAS
│   ├── eval_pipeline.py        ← main runner
│   ├── llm_wrapper.py          ← HF chat → langchain ChatModel
│   └── test_queries.json       ← Q&A ground truths per domain
│
├── docs/                       ← Phase 9: sample documents
│   ├── hr/                     ← onboarding, leave policy, code of conduct
│   ├── tech/                   ← API ref, SDK quickstart, ADR
│   ├── research/               ← (drop arXiv PDFs here)
│   └── product/                ← install guide, release notes, FAQ
│
├── scripts/
│   └── ingest_samples.py       ← one-shot ingest of all bundled samples
│
└── .streamlit/
    └── config.toml             ← theme, colors, gather-stats off
```

---

## 🐳 Run with Docker (recommended)

```cmd
:: 1. Set HF_TOKEN in .env (see .env.example)
copy .env.example .env

:: 2. Bring up the stack
docker compose up -d

:: 3. Initialize namespaces and ingest samples
docker compose exec api python -m vectorstore.init_namespaces
docker compose exec api python scripts/ingest_samples.py
```

Services:

| Service | URL | Purpose |
|---|---|---|
| `chromadb` | http://localhost:8000 | Vector store |
| `api` | http://localhost:8001 | FastAPI — see `/docs` for Swagger |
| `ui` | http://localhost:8501 | Streamlit UI |
| `mcp` | http://localhost:8002/mcp | MCP Streamable HTTP endpoint |

`docker compose logs -f` to follow output. `docker compose down` to stop;
`docker compose down -v` to also wipe the ChromaDB volume.

---

## 💻 Run locally (per-service)

Useful for development with auto-reload.

```cmd
:: cmd.exe syntax — line continuation = ^
:: For PowerShell, swap ^ for ` ; for bash, swap for \

:: 0. Setup once
copy .env.example .env
::    Edit .env: HF_TOKEN=<your token>

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 1. ChromaDB (always in Docker — easier than a local install)
docker compose up -d chromadb

:: 2. Namespaces
python -m vectorstore.init_namespaces

:: 3. Sample ingest
python scripts/ingest_samples.py

:: 4. Start services in separate terminals
uvicorn api.main:app --reload --port 8001
streamlit run ui/app.py
python -m mcp_server.server --transport http --port 8002
```

---

## 🔌 REST API

Auto-generated docs at `http://localhost:8001/docs` (Swagger UI) and
`http://localhost:8001/redoc`.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | ChromaDB connectivity + per-namespace chunk counts |
| `GET`  | `/models` | Default + supported LLM IDs |
| `GET`  | `/namespaces` | Per-namespace chunk counts |
| `POST` | `/query/{domain}` | Single-domain query — `hr`/`tech`/`research`/`product` |
| `POST` | `/query/federated` | Federated query across all four namespaces |
| `POST` | `/ingest` | Ingest a server-local directory into a namespace |
| `GET`  | `/evaluation/queries` | Test query set (grouped by domain) |
| `POST` | `/evaluation/run` | Run RAGAS evaluation for one domain |

Example — federated query via curl:

```cmd
curl -X POST http://localhost:8001/query/federated ^
    -H "Content-Type: application/json" ^
    -d "{\"question\": \"How do I authenticate API requests?\", \"top_k\": 5}"
```

---

## 🤖 MCP server — bridge to multi-agent systems

Six tools, one for each namespace plus a federated and an introspection tool:

| Tool | Returns |
|---|---|
| `query_hr_knowledge(question)` | Answer + citations from HR |
| `query_tech_docs(question)` | Answer + citations from Tech |
| `query_research_papers(question)` | Answer + citations from Research |
| `query_product_knowledge(question)` | Answer + citations from Product |
| `federated_query(question, top_k_per_domain=5)` | Cross-namespace answer with domain-tagged citations |
| `list_namespaces()` | Inventory + chunk counts (for agent cold-starts) |

**Wire it into Claude Desktop** via `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "documind": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/documind"
    }
  }
}
```

**Why this matters:** the next project (multi-agent system) plugs straight
into these tools. Each agent gets a narrow tool — `query_hr_knowledge` for an
HR agent, `query_tech_docs` for tech-support — and `federated_query` for a
coordinator that needs cross-domain answers. **Zero rework** between projects.

---

## 🔍 Querying — single + federated

CLI (single-domain):

```cmd
python -m rag_pipeline.query --domain hr --question "What is the leave policy?"
```

CLI (federated):

```cmd
python -m rag_pipeline.query --federated --question "What does the company say about authentication?"
```

Citations in federated mode include the origin domain:

```
Aurora authenticates API requests with a Bearer token [1]. Free-tier accounts
are capped at 60 requests per minute [1], and the SDK reads the API key from
the AURORA_API_KEY environment variable by default [2].

Sources:
[1] [tech] api_reference.md
[2] [tech] sdk_quickstart.md
```

**Supported LLMs** (curated, gated models marked ‡ — accept their license on
huggingface.co once):

- `Qwen/Qwen2.5-7B-Instruct` *(default — non-gated)*
- `Qwen/Qwen2.5-72B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct` ‡
- `meta-llama/Llama-3.3-70B-Instruct` ‡
- `mistralai/Mistral-Nemo-Instruct-2407`
- `microsoft/Phi-3.5-mini-instruct`
- `deepseek-ai/DeepSeek-V4-Flash` *(MIT)*
- `deepseek-ai/DeepSeek-V4-Pro` *(MIT — flagship, burns free credits fast)*

Pin a provider if HF's `auto` router hits an unhealthy backend:
`--provider together` / `--provider fireworks-ai` / `--provider sambanova`.

---

## 📊 RAGAS evaluation

Test queries live in `evaluation/test_queries.json`. The bundled set has
**ground truths matched to the sample docs**, so the evaluation tab gives
meaningful numbers out of the box.

| Metric | Needs ground truth? | What it measures |
|---|---|---|
| `faithfulness` | no | Does the answer factually align with the retrieved context? |
| `answer_relevancy` | no | Does the answer actually address the question? |
| `context_precision` | yes | Are the retrieved chunks relevant? |
| `context_recall` | yes | Do the retrieved chunks cover everything in the ground truth? |

Run from the **📊 Evaluation** tab in the UI, or via CLI:

```cmd
python -c "from vectorstore import Domain; from evaluation import evaluate_domain; import json; print(json.dumps(evaluate_domain(Domain.HR, max_queries=3), indent=2))"
```

The judging LLM uses the same HF Inference stack as generation — one
`HF_TOKEN` covers everything. The UI lets you pick a stronger model as the
judge if you want to reduce self-evaluation bias.

---

## 🧭 Build phases

- [x] **Phase 1** — Docker Compose + ChromaDB + namespace setup
- [x] **Phase 2** — Ingestion pipeline (loaders, chunker, embedder)
- [x] **Phase 3** — LangGraph pipeline (single domain)
- [x] **Phase 4** — Federated search + cross-encoder reranker
- [x] **Phase 5** — FastAPI layer
- [x] **Phase 6** — MCP server (FastMCP)
- [x] **Phase 7** — Streamlit UI
- [x] **Phase 8** — RAGAS evaluation
- [x] **Phase 9** — Sample documents, demo GIF, polished README

---

## 🎥 Demo recording guide

Record a short loop (40–60s) for `docs/demo.gif`. Suggested shot list:

1. **Title card** (1s) — DocuMind logo / repo name.
2. **`docker compose up`** (3s) — show the four services boot.
3. **Streamlit Status tab** (3s) — 🟢 connected, four namespaces with chunk counts.
4. **Single-domain query** (8s) — type *"What is the leave policy?"*, hit Enter,
   answer appears with `[1]` `[2]` citations, expand a source card to show the
   text.
5. **Federated query** (10s) — switch the mode pill to *Federated*, ask a
   cross-domain question, point out that citation badges now show different
   domain colors.
6. **Ingest tab** (6s) — drag a PDF in, click Ingest, watch the chunk count on
   the header strip go up.
7. **Evaluation tab** (10s) — pick a domain, click *Run evaluation*, show the
   metric bar chart filling in.
8. **(Optional) Claude Desktop** (8s) — ask Claude to use `query_hr_knowledge`
   and show it returning a cited answer.

**Tools that produce good GIFs:**
- **Windows:** [ScreenToGif](https://www.screentogif.com/) (free, native GIF export).
- **macOS:** [Kap](https://getkap.co/) (free).
- **Linux:** [Peek](https://github.com/phw/peek) (free).

Aim for ≤6 MB so GitHub embeds inline. ScreenToGif's "Reduce frame count"
option helps without obvious quality loss.

---

## 🛣️ Roadmap — what's next

The natural extension is a **multi-agent system** that consumes the MCP tools
as its memory layer:

- **HR agent** — owns `query_hr_knowledge`, handles employee Q&A.
- **Tech-support agent** — owns `query_tech_docs`, handles API/SDK questions.
- **Research agent** — owns `query_research_papers`, summarises and compares papers.
- **Product agent** — owns `query_product_knowledge`, handles user-facing FAQ.
- **Coordinator agent** — calls `federated_query` when the question is ambiguous,
  delegates to a specialist agent based on the answer's domain mix.

Because each domain is already its own MCP tool, the agent project starts
with **zero re-implementation of retrieval, ranking, or synthesis**.

---

## 📜 License

MIT — see [`LICENSE`](LICENSE) for the full text. Sample documents under
`docs/` are entirely fictional and depict a made-up company called "Aurora Labs".
