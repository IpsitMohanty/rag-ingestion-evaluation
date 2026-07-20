"""Vector store adapter: Chroma persisted to disk. Swapping the backend
means adding a branch here and a new VectorStoreConfig.backend value --
pipeline/ and cli.py only ever see the returned VectorStore object.
"""
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from config import VectorStoreConfig


def get_vectorstore(config: VectorStoreConfig, embeddings: Embeddings) -> VectorStore:
    if config.backend != "chroma":
        raise ValueError(f"Unknown vectorstore backend {config.backend!r}")

    from langchain_chroma import Chroma

    config.persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.persist_directory),
    )


def index_documents(vectorstore: VectorStore, documents: list[Document]) -> list[str]:
    return vectorstore.add_documents(documents)
