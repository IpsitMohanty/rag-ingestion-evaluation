"""Single place for pipeline configuration: chunk sizes, model names, k, and
per-source-type ingestion behavior.

Nothing here talks to a real backend -- it's plain data. Adapters read a
Config and decide what to do; application code never hardcodes a chunk size
or model name outside of the defaults below.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SplitterSettings:
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass(frozen=True)
class SourceTypeSettings:
    """How one source type is turned into Documents.

    split=False means the loader's Documents are indexed as-is (one
    Document per record) -- no text splitter runs. split=True means the
    loader's Documents are passed through `splitter`.
    """
    split: bool
    splitter: SplitterSettings | None = None


@dataclass(frozen=True)
class IngestionConfig:
    # Keyed by source_type name, not file extension -- a source type is a
    # content shape ("faq", "policy_pdf"), not a format, so a future format
    # sharing the same shape (e.g. a markdown FAQ export) reuses the same
    # settings instead of forking a new branch.
    source_types: dict[str, SourceTypeSettings] = field(default_factory=lambda: {
        "faq": SourceTypeSettings(split=False, splitter=None),
        "policy_pdf": SourceTypeSettings(
            split=True,
            splitter=SplitterSettings(chunk_size=800, chunk_overlap=100),
        ),
    })


@dataclass(frozen=True)
class EmbeddingConfig:
    # CPU-friendly, no API key, ~90MB. Swap to a bigger model (e.g. the
    # course's sentence-transformers/all-mpnet-base-v2, ~420MB, 768-dim)
    # by changing model_name -- backend stays "huggingface" either way.
    # backend="watsonx"/"openai" are wired in the adapter interface but
    # not implemented in phase 1 (no cloud credentials required to run
    # this repo).
    backend: Literal["huggingface", "watsonx", "openai"] = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class VectorStoreConfig:
    backend: Literal["chroma"] = "chroma"
    persist_directory: Path = REPO_ROOT / ".chroma"
    collection_name: str = "poshan_corpus"


@dataclass(frozen=True)
class RetrieverConfig:
    # Only "similarity" is implemented in phase 1. The other values are
    # valid config, not implemented strategies -- selecting them raises
    # NotImplementedError from the adapter (see adapters/retriever.py)
    # rather than silently falling back, so a phase 2 evaluation run
    # fails loudly instead of quietly testing the wrong strategy.
    strategy: Literal[
        "similarity", "multi_query", "self_query", "parent_document"
    ] = "similarity"
    k: int = 4


@dataclass(frozen=True)
class LLMConfig:
    # Unused in phase 1 (no generation step). Kept here so the seam for
    # multi-query/self-query retrieval and phase 2's optional LLM-judge
    # arm has a single place to configure a backend later.
    backend: Literal["huggingface", "watsonx", "openai"] | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


DEFAULT_CONFIG = Config()
