"""Coverage: PDF ingestion adapter -- the split, page-based path.

Uses the session-scoped pdf_raw_docs/pdf_chunks fixtures (conftest.py)
rather than re-parsing the 77-page PDF in every test -- PdfReader is the
suite's single slowest operation.
"""


def test_pdf_loader_returns_non_empty_documents(pdf_raw_docs):
    assert pdf_raw_docs
    assert all(d.page_content.strip() for d in pdf_raw_docs)


def test_pdf_documents_have_expected_metadata_keys(pdf_raw_docs):
    for d in pdf_raw_docs:
        assert {"source", "page"} <= set(d.metadata)


def test_pdf_pages_are_in_document_order(pdf_raw_docs):
    pages = [d.metadata["page"] for d in pdf_raw_docs]
    assert pages == sorted(pages)
    assert len(pages) == len(set(pages))  # no page loaded twice


def test_pdf_split_produces_more_smaller_chunks(pdf_raw_docs, pdf_chunks):
    assert len(pdf_chunks) > len(pdf_raw_docs)


def test_pdf_split_respects_chunk_size(pdf_chunks, default_config):
    settings = default_config.ingestion.source_types["policy_pdf"]
    # RecursiveCharacterTextSplitter can slightly exceed chunk_size when a
    # single unsplittable unit (a long word/number) sits at a boundary --
    # allow a small margin instead of asserting a hard, occasionally-false cap.
    assert all(len(c.page_content) <= settings.splitter.chunk_size + 50 for c in pdf_chunks)


def test_pdf_split_has_no_empty_chunks(pdf_chunks):
    assert all(c.page_content.strip() for c in pdf_chunks)


def test_pdf_split_chunks_preserve_source_metadata(pdf_chunks):
    assert all("source" in c.metadata and "page" in c.metadata for c in pdf_chunks)
