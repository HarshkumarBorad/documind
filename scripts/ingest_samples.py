"""One-shot ingestion of the sample documents in `docs/` into all four namespaces.

Run after `python -m vectorstore.init_namespaces`:

    python scripts/ingest_samples.py
    python scripts/ingest_samples.py --reset      # wipe namespaces first

Empty / missing domain directories are skipped (e.g. `docs/research/` ships
empty so users drop their own arXiv PDFs in).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable regardless of where this script is invoked from.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.ingest_pipeline import ingest  # noqa: E402
from vectorstore import Domain  # noqa: E402

DOCS_ROOT = ROOT / "docs"

SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm"}


def _has_ingestible_files(path: Path) -> bool:
    return any(p.is_file() and p.suffix.lower() in SUPPORTED_EXTS for p in path.rglob("*"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest sample docs into all four namespaces.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe each namespace before ingesting.",
    )
    args = parser.parse_args()

    if not DOCS_ROOT.is_dir():
        print(f"ERROR: docs/ directory not found at {DOCS_ROOT}", file=sys.stderr)
        return 1

    total = 0
    for domain in Domain:
        path = DOCS_ROOT / domain.value
        if not path.is_dir():
            print(f"  [skip] {domain.value}: {path} does not exist")
            continue
        if not _has_ingestible_files(path):
            print(f"  [skip] {domain.value}: no supported files in {path}")
            continue

        print(f"\n=== Ingesting {domain.value} from {path} ===")
        try:
            count = ingest(domain, path, reset=args.reset)
            total += count
        except Exception as exc:
            print(f"  ERROR ingesting {domain.value}: {exc}", file=sys.stderr)

    print(f"\nDone. Total chunks added across all namespaces: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
