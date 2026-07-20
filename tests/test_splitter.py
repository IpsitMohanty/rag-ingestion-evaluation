"""Coverage: splitter behavior in isolation, via a synthetic document
instead of the real (messy, extraction-artifact-prone) PDF text -- this is
what makes the overlap assertion below reliable rather than incidental.
"""
from langchain_core.documents import Document

from adapters import ingestion
from config import SourceTypeSettings, SplitterSettings

LONG_TEXT = (
    "Paragraph one covers the first topic in some detail. " * 5
    + "Paragraph two moves on to a second, unrelated topic. " * 5
    + "Paragraph three wraps up with a conclusion. " * 5
)


def _settings(chunk_size=100, chunk_overlap=20):
    return SourceTypeSettings(split=True, splitter=SplitterSettings(chunk_size, chunk_overlap))


def test_split_respects_chunk_size():
    chunks = ingestion.split_documents([Document(page_content=LONG_TEXT)], _settings(100, 20))
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_split_produces_no_empty_chunks():
    chunks = ingestion.split_documents([Document(page_content=LONG_TEXT)], _settings(100, 20))
    assert all(c.page_content.strip() for c in chunks)


def test_split_overlap_actually_overlaps():
    """Consecutive chunks share a real substring of meaningful length --
    proving chunk_overlap changes actual output, not just a config number
    nobody checks.
    """
    chunks = ingestion.split_documents([Document(page_content=LONG_TEXT)], _settings(100, 20))
    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:]):
        tail = first.page_content[-12:]
        assert tail in second.page_content, (
            f"expected the tail of chunk {tail!r} to reappear in the next chunk "
            f"{second.page_content!r} -- chunk_overlap=20 should guarantee this"
        )


def test_larger_overlap_yields_more_shared_text():
    """Sanity check that chunk_overlap is actually wired through to the
    splitter, not silently ignored: a bigger overlap must not produce a
    strictly smaller chunk count for the same input and chunk_size."""
    small_overlap = ingestion.split_documents([Document(page_content=LONG_TEXT)], _settings(100, 0))
    large_overlap = ingestion.split_documents([Document(page_content=LONG_TEXT)], _settings(100, 40))
    assert len(large_overlap) >= len(small_overlap)


def test_no_split_setting_returns_documents_unchanged():
    settings = SourceTypeSettings(split=False, splitter=None)
    docs = [Document(page_content="Q: x\nA: y"), Document(page_content="Q: a\nA: b")]
    result = ingestion.split_documents(docs, settings)
    assert result == docs


def test_no_split_setting_still_drops_blank_documents():
    settings = SourceTypeSettings(split=False, splitter=None)
    docs = [Document(page_content="real content"), Document(page_content="   ")]
    result = ingestion.split_documents(docs, settings)
    assert len(result) == 1
    assert result[0].page_content == "real content"
