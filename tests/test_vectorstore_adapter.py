"""Coverage: vector store adapter contract (Chroma), using a deterministic
fake embeddings backend -- no network, no model download, no torch.
"""
import pytest

from adapters import vectorstore as vectorstore_adapter
from config import VectorStoreConfig
from langchain_core.documents import Document


def _config(tmp_path):
    return VectorStoreConfig(
        backend="chroma", persist_directory=tmp_path / "chroma", collection_name="test"
    )


def test_unknown_backend_raises(tmp_path, fake_embeddings):
    with pytest.raises(ValueError):
        vectorstore_adapter.get_vectorstore(
            VectorStoreConfig(backend="not-a-real-backend", persist_directory=tmp_path),
            fake_embeddings,
        )


def test_index_and_persist_directory_is_created(tmp_path, fake_embeddings):
    config = _config(tmp_path)
    assert not config.persist_directory.exists()
    vectorstore_adapter.get_vectorstore(config, fake_embeddings)
    assert config.persist_directory.exists()


def test_index_documents_round_trips_through_similarity_search(tmp_path, fake_embeddings):
    docs = [
        Document(page_content="email policy details", metadata={"source": "a"}),
        Document(page_content="smoking policy details", metadata={"source": "b"}),
    ]
    store = vectorstore_adapter.get_vectorstore(_config(tmp_path), fake_embeddings)
    vectorstore_adapter.index_documents(store, docs)

    results = store.similarity_search("email policy", k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] in {"a", "b"}
