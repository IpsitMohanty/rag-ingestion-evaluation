"""Pipeline stage 3: embed and store Documents in the vector store."""
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from adapters import embeddings as embeddings_adapter
from adapters import vectorstore as vectorstore_adapter
from config import Config


def build_index(documents: list[Document], config: Config) -> VectorStore:
    embeddings = embeddings_adapter.get_embeddings(config.embedding)
    store = vectorstore_adapter.get_vectorstore(config.vectorstore, embeddings)
    if documents:
        vectorstore_adapter.index_documents(store, documents)
    return store
