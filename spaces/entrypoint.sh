#!/bin/bash
#
# Container entrypoint for the HuggingFace Space.
# 1. Create the four namespaces (idempotent).
# 2. Ingest sample docs IF the namespaces are empty (skips on wake-from-sleep).
# 3. Start Streamlit on the HF Spaces standard port 7860.
#
set -euo pipefail

echo "==> Initializing ChromaDB namespaces..."
python -m vectorstore.init_namespaces

echo "==> Checking sample documents..."
# Idempotent — re-ingestion would just upsert, but each call hits the HF API
# to embed. --skip-if-populated avoids that cost when waking from sleep.
python scripts/ingest_samples.py --skip-if-populated || {
    echo "WARN: ingest_samples failed — Streamlit will still start, namespaces may be empty"
}

echo "==> Starting Streamlit on port 7860..."
exec streamlit run ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none
