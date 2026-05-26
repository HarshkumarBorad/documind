# syntax=docker/dockerfile:1.6
#
# Single image, multiple commands — `docker compose` runs the same image with
# different CMDs for the api / ui / mcp services.
#
FROM python:3.11-slim

WORKDIR /app

# Build tooling for native wheels (lxml, sentence-transformers' rust deps, etc.).
# `git` is required at runtime by ragas → gitpython, which probes for the
# binary on import even when no git operations are actually performed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so they layer-cache independently of source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_PYTHON_REFRESH=quiet

# Default to the API; docker-compose overrides per service.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
