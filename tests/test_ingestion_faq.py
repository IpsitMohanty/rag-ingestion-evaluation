"""Coverage: FAQ ingestion adapter -- the no-split, metadata-preserving path.
"""
from adapters import ingestion

EXPECTED_TABS = {
    "Beneficiary",
    "AWC Users",
    "Officials (Dashboard)",
    "Poshan Tracker Application",
    "Miscellaneous",
}


def test_faq_json_produces_126_documents(faq_json_path):
    docs = ingestion.load_source("faq", faq_json_path)
    assert len(docs) == 126


def test_faq_documents_are_non_empty(faq_json_path):
    docs = ingestion.load_source("faq", faq_json_path)
    assert docs
    assert all(d.page_content.strip() for d in docs)


def test_faq_metadata_has_expected_keys(faq_json_path):
    docs = ingestion.load_source("faq", faq_json_path)
    for d in docs:
        assert {"source", "tab", "subcategory", "question"} <= set(d.metadata)


def test_faq_preserves_tab(faq_json_path):
    docs = ingestion.load_source("faq", faq_json_path)
    assert {d.metadata["tab"] for d in docs} == EXPECTED_TABS


def test_faq_preserves_subcategory_and_normalizes_null(faq_json_path):
    docs = ingestion.load_source("faq", faq_json_path)
    # Chroma's metadata store rejects None -- the adapter must normalize
    # null subcategories (31 of 126 source rows) to "", never leave None.
    assert all(d.metadata["subcategory"] is not None for d in docs)
    named_subcategories = {d.metadata["subcategory"] for d in docs if d.metadata["subcategory"]}
    assert len(named_subcategories) == 13
    unnamed = [d for d in docs if d.metadata["subcategory"] == ""]
    assert len(unnamed) == 31


def test_faq_ingestion_does_not_split(faq_json_path, default_config):
    settings = default_config.ingestion.source_types["faq"]
    docs = ingestion.load_source("faq", faq_json_path)
    result = ingestion.split_documents(docs, settings)
    # One Document per Q&A pair, untouched -- this is the "FAQ pairs ingest
    # WITHOUT splitting" requirement, exercised through the same
    # split_documents() call path the PDF documents go through.
    assert len(result) == len(docs)
    assert [d.page_content for d in result] == [d.page_content for d in docs]


def test_faq_json_and_csv_agree_on_row_count(faq_json_path, faq_csv_path):
    """Both supplied formats are parseable and structurally equivalent --
    the pipeline defaults to JSON (see README for why), but the CSV loader
    path must not silently drop or mis-split rows if ever used instead."""
    json_docs = ingestion.load_source("faq", faq_json_path)
    csv_docs = ingestion.load_source("faq", faq_csv_path)
    assert len(json_docs) == len(csv_docs) == 126
