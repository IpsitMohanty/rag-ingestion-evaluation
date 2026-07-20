"""Builds a FAQ sub-index (always similarity, per METHODOLOGY.md #5b) and
a policy sub-index (similarity or parent_document, per the sweep cell's
retriever strategy), then answers a query against both and merges scored
hits into one ranked top-k list.

Scores from both sub-indexes are directly comparable: every sub-index in
a given sweep cell is built with the same embedding model and Chroma's
default distance metric, regardless of which strategy retrieved it.
"""
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from adapters import vectorstore as vectorstore_adapter
from config import RetrieverConfig, VectorStoreConfig

from .metrics import ScoredHit


def _vectorstore(embeddings: Embeddings, persist_dir: Path, collection_name: str):
    config = VectorStoreConfig(
        backend="chroma", persist_directory=persist_dir, collection_name=collection_name
    )
    return vectorstore_adapter.get_vectorstore(config, embeddings)


def build_faq_index(faq_documents: list[Document], embeddings: Embeddings, persist_dir: Path):
    store = _vectorstore(embeddings, persist_dir, f"faq-{uuid.uuid4().hex[:12]}")
    if faq_documents:
        vectorstore_adapter.index_documents(store, faq_documents)
    return store


def build_policy_index(
    strategy: str,
    policy_chunks: list[Document],
    policy_raw_pages: list[Document],
    embeddings: Embeddings,
    persist_dir: Path,
    retriever_config: RetrieverConfig,
):
    """Returns an object with a `.query_scored(text, k)` -> list[(Document, score)]
    method, regardless of strategy -- similarity and parent_document expose
    that differently under the hood (see _PolicySimilarityIndex /
    _PolicyParentDocumentIndex below).
    """
    store = _vectorstore(embeddings, persist_dir, f"policy-{uuid.uuid4().hex[:12]}")

    if strategy == "similarity":
        if policy_chunks:
            vectorstore_adapter.index_documents(store, policy_chunks)
        return _PolicySimilarityIndex(store)

    if strategy == "parent_document":
        from adapters import retriever as retriever_adapter

        retriever = retriever_adapter.get_retriever(retriever_config, store, documents=policy_raw_pages)
        return _PolicyParentDocumentIndex(retriever)

    raise ValueError(f"Unknown policy retriever strategy {strategy!r}")


class _PolicySimilarityIndex:
    def __init__(self, store):
        self._store = store

    def query_scored(self, text: str, k: int) -> list[tuple[Document, float]]:
        if not self._has_documents():
            return []
        return self._store.similarity_search_with_score(text, k=k)

    def _has_documents(self) -> bool:
        return self._store._collection.count() > 0


class _PolicyParentDocumentIndex:
    """ParentDocumentRetriever has no native `_with_score` method -- scores
    come from its internal (child-level) vectorstore, and each child hit's
    `doc_id` metadata is used to resolve the parent Document via its
    docstore. Documented as the harness's own scoring shim (METHODOLOGY.md
    #5b), not a LangChain-provided capability.
    """

    def __init__(self, retriever):
        self._retriever = retriever

    def query_scored(self, text: str, k: int) -> list[tuple[Document, float]]:
        child_hits = self._retriever.vectorstore.similarity_search_with_score(text, k=k)
        results = []
        for child_doc, score in child_hits:
            doc_id = child_doc.metadata.get("doc_id")
            parent_docs = self._retriever.docstore.mget([doc_id]) if doc_id else [None]
            parent = parent_docs[0] if parent_docs and parent_docs[0] is not None else child_doc
            results.append((parent, score))
        return results


def query_combined(
    query: str,
    k: int,
    faq_store,
    policy_index,
) -> list[ScoredHit]:
    faq_hits = faq_store.similarity_search_with_score(query, k=k) if _count(faq_store) else []
    policy_hits = policy_index.query_scored(query, k=k)

    scored: list[ScoredHit] = []
    for doc, score in faq_hits:
        scored.append(ScoredHit(
            source_type="faq",
            score=score,
            document=doc,
            faq_index=doc.metadata.get("faq_index"),
            faq_indices=tuple(doc.metadata["faq_indices"]) if "faq_indices" in doc.metadata else None,
        ))
    for doc, score in policy_hits:
        scored.append(ScoredHit(
            source_type="policy_pdf",
            score=score,
            document=doc,
            page=doc.metadata.get("page"),
        ))

    scored.sort(key=lambda h: h.score)  # lower distance = more similar
    return scored[:k]


def _count(store) -> int:
    return store._collection.count()
