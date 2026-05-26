# Deploying DocuMind to HuggingFace Spaces

A HuggingFace Space is its own Git repo (separate from your GitHub repo).
The flow is: create an empty Space → clone its Git repo → copy the right
files in → push → wait for the build.

Sample documents are pre-ingested on first container start, so a recruiter
landing on the Space URL gets a working demo with no setup.

## Prerequisites

- A free HuggingFace account.
- The same `HF_TOKEN` you've been using locally
  (https://huggingface.co/settings/tokens — "Read" access is enough for
  inference; "Write" is not needed).
- Git installed locally (you already have it).

## Step 1 — create the Space

1. Open https://huggingface.co/new-space
2. Fill in:
   - **Owner:** your HF username (`HarshkumarBorad`)
   - **Space name:** `documind`
   - **License:** MIT
   - **Space SDK:** **Docker** → pick the **Blank** template
   - **Hardware:** *CPU basic — free* (2 vCPU, 16 GB RAM)
   - **Visibility:** Public
3. Click **Create Space**. The new Space will be at
   `https://huggingface.co/spaces/HarshkumarBorad/documind`.

## Step 2 — set the HF_TOKEN secret

The Space needs your token to call the Inference API (for embeddings, chat,
and the reranker model download).

1. On the Space page, click **Settings**.
2. Scroll to **Variables and secrets** → click **New secret**.
3. **Name:** `HF_TOKEN`. **Value:** paste your token.
4. Save.

Without this secret, the Space will boot but every query will error out with
`HF_TOKEN is not set`.

## Step 3 — clone the Space's Git repo

The Space URL is also a Git remote:

```cmd
cd C:\Users\harsh\.gemini\antigravity\scratch
git clone https://huggingface.co/spaces/HarshkumarBorad/documind documind-space
```

This creates a sibling directory `documind-space/` with only a default
README inside.

## Step 4 — copy the application files in

From the **documind** repo, copy everything the Space needs into the
**documind-space** repo:

```cmd
cd C:\Users\harsh\.gemini\antigravity\scratch\documind

:: Source packages — the Space needs these unchanged
xcopy /E /I /Y vectorstore   ..\documind-space\vectorstore
xcopy /E /I /Y ingestion     ..\documind-space\ingestion
xcopy /E /I /Y rag_pipeline  ..\documind-space\rag_pipeline
xcopy /E /I /Y ui            ..\documind-space\ui
xcopy /E /I /Y evaluation    ..\documind-space\evaluation
xcopy /E /I /Y scripts       ..\documind-space\scripts
xcopy /E /I /Y docs          ..\documind-space\docs
xcopy /E /I /Y .streamlit    ..\documind-space\.streamlit

:: License
copy LICENSE ..\documind-space\LICENSE

:: HF Spaces-specific overrides (these REPLACE files of the same name)
copy spaces\Dockerfile       ..\documind-space\Dockerfile
copy spaces\README.md        ..\documind-space\README.md
copy spaces\requirements.txt ..\documind-space\requirements.txt
copy spaces\entrypoint.sh    ..\documind-space\entrypoint.sh
```

> **What you're deliberately NOT copying:** `api/`, `mcp_server/`,
> `docker-compose.yml`, the root `Dockerfile`. None of them are needed on
> the Space — embedded ChromaDB + in-process Streamlit replaces all of it.

## Step 5 — commit and push

```cmd
cd ..\documind-space
git add -A
git commit -m "initial deployment: DocuMind"
git push
```

You'll likely be prompted for HuggingFace credentials. Use:

- **Username:** your HF username (`HarshkumarBorad`)
- **Password:** an HF **access token** (not your account password — HF
  deprecated password auth). Create one at
  https://huggingface.co/settings/tokens with "Write" scope just for this push.

## Step 6 — watch the build

Back on the Space page, you'll see a build log. First build takes 5–10
minutes (downloading torch, sentence-transformers, building the reranker
deps).

Once it shows "Running", the Streamlit UI is live at:

```
https://huggingface.co/spaces/HarshkumarBorad/documind
```

First load triggers sample-doc ingestion (~30–60s — embeds the bundled docs
through the HF API). After that, queries are fast.

## Updating the Space later

Same flow: in `documind`, change code → re-run the `xcopy` block from Step 4
→ commit + push from `documind-space`. The Space rebuilds automatically.

For frequent updates you can also enable **GitHub-to-HF sync** in the Space's
settings, but that requires the Dockerfile and README to be at the GitHub
repo root — which conflicts with our project layout. The xcopy approach is
the simplest.

## Troubleshooting

**"HF_TOKEN is not set"** — you didn't add the secret in Step 2, or you
named it differently. Settings → Variables and secrets → confirm there's a
secret named exactly `HF_TOKEN`.

**Build fails with `chromadb-client` errors** — make sure you copied
`spaces/requirements.txt` (which uses the full `chromadb` package), not the
project root's `requirements.txt` (which uses the slimmer `chromadb-client`).

**Sample ingest takes forever** — the first run after a rebuild re-embeds
all 9 sample docs through HF. Subsequent wake-from-sleep cycles skip this
because `--skip-if-populated` sees the existing chunks.

**Queries are slow / time out** — the reranker model (~1 GB) downloads
lazily on first query, not at startup. The first query after a rebuild
will be 30–60s slower than subsequent ones. You can disable the reranker by
adding `RERANKER_ENABLED=False` as a Space variable.

**Persistent storage** — the `chroma_data/` directory lives inside the
container's writable layer. It persists across wake-from-sleep cycles but
gets wiped on rebuilds (i.e., when you push new code). Re-ingestion runs on
each rebuild — acceptable for a portfolio demo. For real persistence
across rebuilds, add the **Persistent Storage** addon to the Space (~$5/mo).
