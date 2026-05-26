"""Document loaders — one entry point that dispatches by file extension."""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm"}


def load_file(path: Path | str) -> List[Document]:
    """Load a single file into one or more LangChain Documents.

    PDFs become one Document per page; other formats become one Document per file.
    `source` and `filename` metadata are normalized regardless of loader.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
    elif ext in {".md", ".markdown", ".txt"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext in {".html", ".htm"}:
        loader = BSHTMLLoader(str(path))
    else:
        raise ValueError(f"Unsupported file extension '{ext}' for {path}")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path.resolve()))
        doc.metadata.setdefault("filename", path.name)
    return docs


def load_directory(directory: Path | str) -> List[Document]:
    """Recursively load every supported file in `directory`.

    Skips and warns on unreadable files rather than aborting the whole ingest.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    docs: List[Document] = []
    skipped: List[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            docs.extend(load_file(path))
        except Exception as exc:
            skipped.append((path, str(exc)))

    for path, reason in skipped:
        print(f"  [skip] {path.name}: {reason}")
    return docs
