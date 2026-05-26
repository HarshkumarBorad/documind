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
# --enableXsrfProtection / --enableCORS disabled because HF Spaces reverse-proxies
# the connection — Streamlit's XSRF cookie doesn't round-trip cleanly, so file
# uploads fail with 403 without these flags. Safe for a public demo; revisit if
# you ever add auth / sensitive mutations.
exec streamlit run ui/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none \
    --server.enableXsrfProtection false \
    --server.enableCORS false
