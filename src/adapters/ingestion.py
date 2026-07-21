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
import re
from collections import Counter
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

# A page whose only dot-leader-style lines ("Section title .......... 12")
# number at least this many is treated as a table of contents. Checked
# against this corpus's actual TOC pages (21-23 matches each) and every
# other page (0-1 matches, never close) -- a wide margin, not a tuned edge.
_TOC_DOT_LEADER = re.compile(r"\.{4,}\s*\d+")
_MIN_TOC_DOT_LEADERS = 3

# A cover/title page has little running text once the header's stripped.
# This corpus's real title page is ~140 chars after stripping; its
# shortest real *content* pages (closing paragraphs, "***" section-end
# markers) run 150-250 chars once you're past page 1 -- see
# tests/test_ingestion_pdf.py for the specific pages this must not flag.
_TITLE_PAGE_MAX_CHARS = 300

# A header/footer line must recur on at least this fraction of pages to
# be treated as running page furniture rather than coincidentally-
# repeated content (e.g. this corpus's "***" section-end marker, which
# closes some sections but appears on only ~6% of pages -- nowhere near
# this threshold, so it correctly stays untouched as real content).
_RUNNING_LINE_THRESHOLD = 0.6


def _detect_running_line(pages_text: list[str], *, from_end: bool) -> str | None:
    """The most common non-empty first (or, if from_end, last) line across
    pages, if it recurs on a strong majority of them. None if nothing
    recurs often enough to be page furniture rather than real content."""
    lines = []
    for text in pages_text:
        candidates = reversed(text.splitlines()) if from_end else text.splitlines()
        for line in candidates:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
                break
    if not lines:
        return None
    line, count = Counter(lines).most_common(1)[0]
    return line if count / len(pages_text) >= _RUNNING_LINE_THRESHOLD else None


def _strip_running_header(text: str, header: str | None) -> str:
    """Remove `header`'s line (if it's actually this page's first line), a
    page-number-only line following it, and any leading blank lines --
    including blank lines *between* the header and the page number: most
    pages have them adjacent ("header\\npage_num"), but some have a blank
    line in between ("header\\n \\npage_num"), so blanks are skipped
    before checking for the page-number line, not just after.
    """
    if header is None:
        return text
    lines = text.splitlines()
    if not lines or lines[0].strip() != header:
        return text
    lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    if lines and lines[0].strip().isdigit():
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines)


def _strip_running_footer(text: str, footer: str | None) -> str:
    if footer is None:
        return text
    lines = text.splitlines()
    if not lines or lines[-1].strip() != footer:
        return text
    lines = lines[:-1]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines)


def _is_table_of_contents_page(text: str) -> bool:
    """Content-based, not position-based: a page dominated by dot-leader
    lines is a TOC regardless of where in the document it falls."""
    return len(_TOC_DOT_LEADER.findall(text)) >= _MIN_TOC_DOT_LEADERS


def _is_likely_title_page(text: str, page_index: int) -> bool:
    """Position AND content: only the document's very first page is
    eligible, and only if what's left after stripping header/footer is
    short. Deliberately narrow -- this is not a general cover-page
    classifier, just enough to catch this corpus's actual title page
    without flagging real short content pages elsewhere in the document.
    """
    return page_index == 0 and len(text.strip()) < _TITLE_PAGE_MAX_CHARS


def clean_policy_pdf_documents(documents: list[Document]) -> list[Document]:
    """Strip running headers/footers from every page and drop front
    matter (title page, table of contents). Genuinely blank pages are
    already dropped by load_policy_pdf; this removes pages that loaded
    real but non-substantive text.

    `metadata["page"]` is preserved unchanged from the source PDF's page
    numbers on every surviving Document -- eval/queries.yaml's ground-
    truth page references, and anything else keyed on page number, must
    keep working identically whether this ran or not.

    Heuristic, checked against this specific corpus (see
    eval/METHODOLOGY.md #8), not a general-purpose PDF cleaner -- a
    differently-structured document could defeat any of these signals.
    """
    texts = [d.page_content for d in documents]
    header = _detect_running_line(texts, from_end=False)
    footer = _detect_running_line(texts, from_end=True)

    cleaned = []
    for index, d in enumerate(documents):
        text = _strip_running_header(d.page_content, header)
        text = _strip_running_footer(text, footer)
        if _is_table_of_contents_page(text):
            continue
        if _is_likely_title_page(text, index):
            continue
        if not text.strip():
            continue
        cleaned.append(Document(page_content=text, metadata=dict(d.metadata)))
    return cleaned


def load_source(source_type: str, path: Path, clean: bool = False) -> list[Document]:
    """`clean` only affects `policy_pdf` (see clean_policy_pdf_documents);
    ignored for other source types rather than erroring, so callers that
    thread a single per-source-type setting through multiple source types
    don't need to special-case which ones it applies to.
    """
    if source_type not in _LOADERS:
        raise ValueError(f"Unknown source_type {source_type!r}; known: {sorted(_LOADERS)}")
    documents = _LOADERS[source_type](path)
    if source_type == "policy_pdf" and clean:
        documents = clean_policy_pdf_documents(documents)
    return documents


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
