"""Builds the Documents each sweep cell indexes. Reuses src/adapters/
ingestion.py's loaders (never re-implements loading), and adds the two
ingestion-mode behaviors settled in eval/METHODOLOGY.md #5a: format_aware
(the phase 1 default -- one Document per FAQ row, never split) and
uniform (concatenate all FAQ rows into one blob first, discarding row/
tab/subcategory boundaries, then run the same splitter used for the PDF).
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from adapters import ingestion

FAQ_JSON_PATH = REPO_ROOT / "data" / "faq" / "all_faq.json"
POLICY_PDF_PATH = REPO_ROOT / "data" / "policy" / "Mission Saksham Anganwadi and Poshan 2.0 scheme guidelines.pdf"


def build_faq_documents(mode: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    raw = ingestion.load_source("faq", FAQ_JSON_PATH)

    if mode == "format_aware":
        docs = []
        for i, d in enumerate(raw):
            docs.append(Document(page_content=d.page_content, metadata={**d.metadata, "faq_index": i}))
        return docs

    if mode == "uniform":
        # Track each row's character span in the concatenated blob so a
        # chunk that crosses row boundaries can be checked against every
        # row it overlaps, not just one -- this is what makes the
        # "merges unrelated Q&As" failure mode countable rather than
        # merely inferred from a hit-rate gap (METHODOLOGY.md #5a).
        separator = "\n\n"
        blob_parts = []
        spans: list[tuple[int, int]] = []
        offset = 0
        for d in raw:
            start = offset
            blob_parts.append(d.page_content)
            offset += len(d.page_content)
            spans.append((start, offset))
            blob_parts.append(separator)
            offset += len(separator)
        blob = "".join(blob_parts)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
            add_start_index=True,
        )
        chunks = splitter.create_documents([blob])

        docs = []
        for c in chunks:
            if not c.page_content.strip():
                continue
            start = c.metadata["start_index"]
            end = start + len(c.page_content)
            overlapping_rows = tuple(
                i for i, (row_start, row_end) in enumerate(spans)
                if row_start < end and start < row_end
            )
            docs.append(Document(
                # Chroma's metadata store accepts str/int/float/bool/list/None,
                # not tuples -- list() here, tuple() again on the eval side
                # when building ScoredHit (see eval/retrievers.py).
                page_content=c.page_content,
                metadata={"source": "faq_uniform_blob", "faq_indices": list(overlapping_rows)},
            ))
        return docs

    raise ValueError(f"Unknown FAQ ingestion mode {mode!r}")


def build_policy_raw_pages(pdf_variant: str = "raw") -> list[Document]:
    """pdf_variant: "raw" (all 77 pages, running header intact -- phase
    1's existing default) or "cleaned" (header/footer stripped, title
    page + table of contents dropped -- see
    adapters/ingestion.py::clean_policy_pdf_documents and
    eval/METHODOLOGY.md #8). Both are real, swept alternatives; neither
    replaces the other.
    """
    if pdf_variant not in ("raw", "cleaned"):
        raise ValueError(f"Unknown pdf_variant {pdf_variant!r}; expected 'raw' or 'cleaned'")
    return ingestion.load_source("policy_pdf", POLICY_PDF_PATH, clean=(pdf_variant == "cleaned"))


def build_policy_chunks(chunk_size: int, chunk_overlap: int, pdf_variant: str = "raw") -> list[Document]:
    from config import SourceTypeSettings, SplitterSettings

    settings = SourceTypeSettings(split=True, splitter=SplitterSettings(chunk_size, chunk_overlap))
    return ingestion.split_documents(build_policy_raw_pages(pdf_variant), settings)


def mean_rows_per_uniform_chunk(uniform_faq_docs: list[Document]) -> float:
    """Direct measurement of the "merges unrelated Q&As" failure mode:
    average number of distinct original FAQ rows each uniform-mode chunk
    overlaps. 1.0 means no merging happened at that chunk size."""
    if not uniform_faq_docs:
        return 0.0
    counts = [len(d.metadata["faq_indices"]) for d in uniform_faq_docs]
    return sum(counts) / len(counts)
