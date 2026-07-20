"""Retriever adapter: strategy is a config value (config.RetrieverConfig.
strategy), not a code branch the caller chooses. "similarity" and
"parent_document" are implemented. The other values are real, selectable
config for when phase 2 implements them -- each raises NotImplementedError
naming the specific reason it isn't wired up yet, rather than silently
falling back to similarity, so a phase 2 evaluation run fails loudly
instead of quietly testing the wrong strategy.

ParentDocumentRetriever lives in `langchain_classic`, not `langchain` --
LangChain's 1.0 line slimmed the `langchain` package down to
agents/chat_models/embeddings/messages/rate_limiters/tools and moved
legacy chain/retriever abstractions like this one into the separate
`langchain-classic` compatibility package. Confirmed by import error
against `langchain==1.3.14`; not something the phase 1 brief anticipated.
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from config import RetrieverConfig


def get_retriever(
    config: RetrieverConfig,
    vectorstore: VectorStore,
    documents: list[Document] | None = None,
) -> BaseRetriever:
    if config.strategy == "similarity":
        return vectorstore.as_retriever(search_kwargs={"k": config.k})

    if config.strategy == "parent_document":
        # Unlike a plain vectorstore retriever, ParentDocumentRetriever
        # does its own splitting -- it needs the raw, unsplit source
        # documents (e.g. one Document per PDF page), not pre-chunked
        # ones, so it can build its own parent/child chunks and populate
        # both the vectorstore (child chunks) and docstore (parent
        # chunks) itself.
        if documents is None:
            raise ValueError(
                "parent_document retriever needs the raw (unsplit) source "
                "documents to build its own parent/child index -- pass documents="
            )
        from langchain_classic.retrievers import ParentDocumentRetriever
        from langchain_classic.storage import InMemoryStore
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
        )
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=InMemoryStore(),
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": config.k},
        )
        retriever.add_documents(documents)
        return retriever

    if config.strategy == "multi_query":
        raise NotImplementedError(
            "multi_query retriever needs an LLM at retrieval time to generate query "
            "variants -- out of scope while phase 1 has no LLM adapter wired in"
        )

    if config.strategy == "self_query":
        raise NotImplementedError(
            "self_query retriever needs an LLM at retrieval time to build a structured "
            "filter, and the reference lab flags it as flaky even with one -- seam only"
        )

    raise ValueError(f"Unknown retriever strategy {config.strategy!r}")
