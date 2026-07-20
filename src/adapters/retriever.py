"""Retriever adapter: strategy is a config value (config.RetrieverConfig.
strategy), not a code branch the caller chooses. Only "similarity" is
implemented in phase 1. The other values are real, selectable config for
when phase 2 implements them -- each raises NotImplementedError naming the
specific reason it isn't wired up yet, rather than silently falling back to
similarity, so a phase 2 evaluation run fails loudly instead of quietly
testing the wrong strategy.
"""
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from config import RetrieverConfig


def get_retriever(config: RetrieverConfig, vectorstore: VectorStore) -> BaseRetriever:
    if config.strategy == "similarity":
        return vectorstore.as_retriever(search_kwargs={"k": config.k})

    if config.strategy == "parent_document":
        # No LLM needed at retrieval time (unlike multi_query/self_query
        # below) -- promoted to a phase 2 implementation target, not just a
        # seam, because small-chunk-match/large-parent-return pairs well
        # with the format-aware ingestion thesis. Needs a docstore for
        # parent chunks alongside the vectorstore, which this adapter's
        # signature doesn't carry yet -- phase 2 work.
        raise NotImplementedError(
            "parent_document retriever is a phase 2 implementation target, not yet built"
        )

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
