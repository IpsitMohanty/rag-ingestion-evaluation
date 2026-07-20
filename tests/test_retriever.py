"""Coverage: retriever adapter -- similarity and parent_document
strategies return exactly k documents with source metadata intact, and
still-unimplemented strategies fail loudly and specifically rather than
silently falling back to similarity.
"""
import pytest
from langchain_core.documents import Document

from adapters import retriever as retriever_adapter
from adapters import vectorstore as vectorstore_adapter
from config import RetrieverConfig, VectorStoreConfig

DOCS = [
    Document(page_content=f"document number {i} about topic {i % 3}", metadata={"source": f"doc-{i}", "idx": i})
    for i in range(10)
]


def _store(tmp_path, fake_embeddings):
    config = VectorStoreConfig(backend="chroma", persist_directory=tmp_path / "chroma", collection_name="test-retriever")
    store = vectorstore_adapter.get_vectorstore(config, fake_embeddings)
    vectorstore_adapter.index_documents(store, DOCS)
    return store


def test_similarity_returns_exactly_k(tmp_path, fake_embeddings):
    store = _store(tmp_path, fake_embeddings)
    retriever = retriever_adapter.get_retriever(RetrieverConfig(strategy="similarity", k=3), store)
    results = retriever.invoke("topic 1")
    assert len(results) == 3


def test_similarity_preserves_source_metadata(tmp_path, fake_embeddings):
    store = _store(tmp_path, fake_embeddings)
    retriever = retriever_adapter.get_retriever(RetrieverConfig(strategy="similarity", k=4), store)
    results = retriever.invoke("topic 0")
    assert all("source" in d.metadata and d.metadata["source"].startswith("doc-") for d in results)


def test_similarity_k_is_configurable(tmp_path, fake_embeddings):
    store = _store(tmp_path, fake_embeddings)
    for k in (1, 5):
        retriever = retriever_adapter.get_retriever(RetrieverConfig(strategy="similarity", k=k), store)
        assert len(retriever.invoke("topic 2")) == k


@pytest.mark.parametrize("strategy", ["multi_query", "self_query"])
def test_unimplemented_strategies_raise_not_implemented(tmp_path, fake_embeddings, strategy):
    """Real, selectable config values (seams needing an LLM at retrieval
    time), not silently ignored -- picking one must fail loudly, never
    fall back to similarity."""
    store = _store(tmp_path, fake_embeddings)
    with pytest.raises(NotImplementedError):
        retriever_adapter.get_retriever(RetrieverConfig(strategy=strategy, k=4), store)


def test_parent_document_without_documents_raises_value_error(tmp_path, fake_embeddings):
    """parent_document needs raw source documents to build its own
    parent/child index -- omitting them is a caller error, not a missing
    feature (see test_parent_document_returns_larger_parent_chunks below
    for the implemented path)."""
    store = _store(tmp_path, fake_embeddings)
    with pytest.raises(ValueError, match="documents"):
        retriever_adapter.get_retriever(RetrieverConfig(strategy="parent_document", k=4), store)


def test_parent_document_returns_larger_parent_chunks(tmp_path, fake_embeddings):
    raw_docs = [
        Document(page_content=("Paragraph about topic zero. " * 20), metadata={"page": i})
        for i in range(3)
    ]
    config = VectorStoreConfig(
        backend="chroma", persist_directory=tmp_path / "chroma-parent", collection_name="test-parent"
    )
    store = vectorstore_adapter.get_vectorstore(config, fake_embeddings)
    retriever = retriever_adapter.get_retriever(
        RetrieverConfig(
            strategy="parent_document", k=2,
            child_chunk_size=50, child_chunk_overlap=10,
            parent_chunk_size=200, parent_chunk_overlap=20,
        ),
        store, documents=raw_docs,
    )
    results = retriever.invoke("topic zero")
    assert len(results) == 2
    assert all(len(r.page_content) > 50 for r in results)  # parent-sized, not child-sized
    assert all("page" in r.metadata for r in results)


def test_unknown_strategy_raises_value_error(tmp_path, fake_embeddings):
    store = _store(tmp_path, fake_embeddings)
    with pytest.raises(ValueError):
        retriever_adapter.get_retriever(RetrieverConfig(strategy="not-a-strategy", k=4), store)
