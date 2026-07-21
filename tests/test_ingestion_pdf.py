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


class TestPdfCleaning:
    """Coverage: the cleaned-PDF ingestion variant (adapters/ingestion.py's
    clean_policy_pdf_documents) -- a config option alongside raw, not a
    replacement for it (see config.SourceTypeSettings.clean,
    eval/METHODOLOGY.md #8). Uses the real corpus (via the session-scoped
    pdf_raw_docs/pdf_cleaned_docs fixtures), since the heuristics here are
    explicitly checked against this specific document's actual structure,
    not asserted as a general-purpose PDF cleaner.
    """

    def test_drops_exactly_the_known_front_matter_pages(self, pdf_raw_docs, pdf_cleaned_docs):
        raw_pages = {d.metadata["page"] for d in pdf_raw_docs}
        cleaned_pages = {d.metadata["page"] for d in pdf_cleaned_docs}
        # Page 1 (title page) + pages 2-4 (table of contents) -- see
        # eval/METHODOLOGY.md #8 for how these were identified.
        assert raw_pages - cleaned_pages == {1, 2, 3, 4}

    def test_page_numbers_are_unchanged_on_surviving_pages(self, pdf_raw_docs, pdf_cleaned_docs):
        """Ground-truth page references in eval/queries.yaml must keep
        meaning the same thing whether cleaning ran or not."""
        raw_by_page = {d.metadata["page"]: d for d in pdf_raw_docs}
        for cleaned_doc in pdf_cleaned_docs:
            page = cleaned_doc.metadata["page"]
            assert page in raw_by_page
            # cleaned content is a (stripped) subset of the raw page's
            # content, not a different page's content under the same number
            assert cleaned_doc.page_content in raw_by_page[page].page_content

    def test_running_header_is_stripped_from_every_surviving_page(self, pdf_cleaned_docs):
        for d in pdf_cleaned_docs:
            first_line = d.page_content.splitlines()[0].strip() if d.page_content.splitlines() else ""
            assert first_line != "Saksham Anganwadi and Poshan 2.0"
            assert not first_line.isdigit()  # the page-number line right after it

    def test_no_page_becomes_empty_after_cleaning(self, pdf_cleaned_docs):
        assert all(d.page_content.strip() for d in pdf_cleaned_docs)

    def test_a_known_content_page_keeps_its_real_content(self, pdf_cleaned_docs):
        """Page 18 (printed page 14): '3.1 Package of Services' -- the
        six-service list pol-07 in eval/queries.yaml is grounded on.
        Regression check that cleaning removes furniture, not content.
        """
        page_18 = [d for d in pdf_cleaned_docs if d.metadata["page"] == 18][0]
        assert "Package of Services" in page_18.page_content

    def test_raw_variant_is_unaffected_by_cleaning_existing(self, pdf_raw_docs):
        """clean=False (the default) must still return all 77 pages,
        header intact -- adding the cleaned variant must not change the
        existing raw path's behavior."""
        assert len(pdf_raw_docs) == 77
        page_1 = [d for d in pdf_raw_docs if d.metadata["page"] == 1][0]
        assert page_1.page_content.startswith("Saksham Anganwadi and Poshan 2.0")
