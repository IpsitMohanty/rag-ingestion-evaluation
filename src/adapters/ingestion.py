"""Ingestion adapter: turns raw corpus files into `Document` objects.

Pipeline code calls `load_source` / `split_documents` and never imports a
loader class directly -- this is the seam that lets a loader implementation
change, or a new source type get added, without touching pipeline code.

Deliberately does not use langchain_community's document loaders (archived
19 June 2026, no further releases -- see README). PDF text extraction is
done directly with `pypdf`, a plain, actively maintained library that
langchain_community's own PyPDFLoader was only ever a thin wrapper around;
FAQ records are read with the stdlib `csv`/`json` modules. Both loaders
build `langchain_core.documents.Document` objects directly, so nothing here
depends on a LangChain integration package for loading at all.
"""
import csv
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import SourceTypeSettings


def load_faq(path: Path) -> list[Document]:
    """One Document per Q&A pair -- never split (see split_documents).
    tab/subcategory become queryable metadata. subcategory is null for 31
    of the 126 source rows; normalized to "" since Chroma's metadata store
    rejects None values.
    """
    path = Path(path)
    if path.suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported FAQ file extension: {path.suffix}")

    documents = []
    for record in records:
        documents.append(
            Document(
                page_content=f"Question: {record['question']}\nAnswer: {record['answer']}",
                metadata={
                    "source": str(path),
                    "tab": record["tab"] or "",
                    "subcategory": record.get("subcategory") or "",
                    "question": record["question"],
                },
            )
        )
    return documents


def load_policy_pdf(path: Path) -> list[Document]:
    """One Document per page. Splitting (if configured) happens separately
    in split_documents -- loading and splitting are independent per the
    per-source-type design, so a caller can load raw pages without forcing
    a split.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # blank/unextractable page -- nothing to index
        documents.append(
            Document(page_content=text, metadata={"source": str(path), "page": page_number})
        )
    return documents


_LOADERS = {
    "faq": load_faq,
    "policy_pdf": load_policy_pdf,
}


def load_source(source_type: str, path: Path) -> list[Document]:
    if source_type not in _LOADERS:
        raise ValueError(f"Unknown source_type {source_type!r}; known: {sorted(_LOADERS)}")
    return _LOADERS[source_type](path)


def split_documents(documents: list[Document], settings: SourceTypeSettings) -> list[Document]:
    """Apply `settings` to `documents`. settings.split=False (the FAQ path)
    returns documents unchanged -- each Q&A stays one Document. settings.
    split=True (the PDF path) runs RecursiveCharacterTextSplitter. Either
    way, empty/whitespace-only chunks are dropped.
    """
    if not settings.split:
        return [d for d in documents if d.page_content.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.splitter.chunk_size,
        chunk_overlap=settings.splitter.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if c.page_content.strip()]
