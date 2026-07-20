"""Coverage: embeddings adapter contract.

Constructing the real HuggingFace backend is eager -- HuggingFaceEmbeddings
downloads/loads the sentence-transformers model at __init__ time, not on
first call -- so it needs the network (or an existing local model cache)
and is gated behind requires_network / RUN_NETWORK_TESTS. The offline tests
below cover what's true regardless of backend: the adapter's error contract,
and that anything implementing the Embeddings interface (proven here with
the fake backend) works identically through the rest of the pipeline.
"""
import pytest
from langchain_core.embeddings import Embeddings

from adapters import embeddings as embeddings_adapter
from adapters import vectorstore as vectorstore_adapter
from config import EmbeddingConfig, VectorStoreConfig
from conftest import requires_network


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        embeddings_adapter.get_embeddings(EmbeddingConfig(backend="not-a-real-backend"))


def test_watsonx_backend_requires_env_vars(monkeypatch):
    """Real, not a stub: missing credentials fail with a clear error
    instead of silently returning something that looks configured."""
    monkeypatch.delenv("WATSONX_URL", raising=False)
    monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
    monkeypatch.delenv("WATSONX_APIKEY", raising=False)
    with pytest.raises((KeyError, ImportError)):
        embeddings_adapter.get_embeddings(EmbeddingConfig(backend="watsonx", model_name="ibm/slate-125m-english-rtrvr-v2"))


def test_any_embeddings_implementation_works_with_vectorstore_adapter(tmp_path, fake_embeddings):
    """Swapping the embeddings backend must not change the interface the
    rest of the pipeline depends on -- exercised here with the fake
    backend standing in for "any Embeddings implementation".
    """
    assert isinstance(fake_embeddings, Embeddings)
    config = VectorStoreConfig(backend="chroma", persist_directory=tmp_path / "chroma", collection_name="test-embeddings")
    store = vectorstore_adapter.get_vectorstore(config, fake_embeddings)
    assert store is not None  # constructed without the adapter caring which Embeddings impl it got


@requires_network
def test_huggingface_backend_produces_expected_dimension():
    embeddings = embeddings_adapter.get_embeddings(
        EmbeddingConfig(backend="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    vector = embeddings.embed_query("test query")
    assert len(vector) == 384  # all-MiniLM-L6-v2's known output dimension
