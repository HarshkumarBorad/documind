# syntax=docker/dockerfile:1.6
#
# HuggingFace Spaces deployment of DocuMind.
#
# Single container: embedded ChromaDB + Streamlit UI calling the graph
# in-process. No separate FastAPI / ChromaDB / MCP services — HF Spaces
# only exposes one port (7860) per Space.
#
# To use: rename this to `Dockerfile` at the root of your HF Space's git repo.
# See spaces/DEPLOY.md for the full deployment walkthrough.
#
FROM python:3.11-slim

# HF Spaces convention: the container must run as a non-root user with UID 1000.
RUN useradd -m -u 1000 user

# Build tools needed by some Python wheels (lxml, sentence-transformers, etc.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install Python deps first so they layer-cache independently of source changes.
COPY --chown=user:user requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r /tmp/requirements.txt

# Copy the application source.
COPY --chown=user:user . .

# DOCUMIND_MODE=local makes the Streamlit UI call the graph in-process
# instead of going through the (absent) FastAPI service.
# CHROMA_MODE=persistent makes the vectorstore use chromadb.PersistentClient
# with on-disk storage instead of HttpClient.
ENV DOCUMIND_MODE=local \
    CHROMA_MODE=persistent \
    CHROMA_PERSIST_PATH=/home/user/app/chroma_data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN chmod +x entrypoint.sh

EXPOSE 7860

CMD ["./entrypoint.sh"]
