"""Pipeline stage 4: build a retriever over an existing vector store and
run a query against it.
"""
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from adapters import retriever as retriever_adapter
from config import Config


def retrieve(query: str, store: VectorStore, config: Config) -> list[Document]:
    retriever = retriever_adapter.get_retriever(config.retriever, store)
    return retriever.invoke(query)
