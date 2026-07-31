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

    clean is policy_pdf-specific (harmlessly ignored for other source
    types -- see adapters/ingestion.py's load_source): strips running
    headers/footers and drops front matter (title page, table of
    contents) before splitting. False (raw) is the default so existing
    CLI/app behavior is unchanged; True is a real, swept alternative, not
    a replacement -- see eval/METHODOLOGY.md #8 for why both are kept and
    results/ANALYSIS.md for whether cleaning measurably helps.
    """
    split: bool
    splitter: SplitterSettings | None = None
    clean: bool = False


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
    # Only read when strategy="parent_document". Parent splitter is
    # deliberately not independently configurable beyond a fixed ratio to
    # the child settings -- see eval/METHODOLOGY.md #6.
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 150


@dataclass(frozen=True)
class LLMConfig:
    # Unused in phase 1 (no generation step). Kept here so the seam for
    # multi-query/self-query retrieval and phase 2's optional LLM-judge
    # arm has a single place to configure a backend later.
    backend: Literal["huggingface", "watsonx", "openai"] | None = None
    model_name: str | None = None
    # None means "don't pass it, use the API's own default" -- existing
    # callers of this shared seam are unaffected unless they set these
    # explicitly. AgenticConfig below is the one caller that does.
    temperature: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class AgenticConfig:
    # Phase 4 seam: LLM-routed retrieval with an LLM abstention judge,
    # built on LangGraph (src/adapters/agentic.py). Tests whether an LLM
    # judge can do what phase 2's Finding #1 showed a similarity threshold
    # cannot -- see eval/METHODOLOGY.md #9-18. Not called by any phase
    # 1/2/3 code path; the eval harness (eval/agentic_sweep.py) is the
    # only caller, run locally, never in CI.
    #
    # temperature=0.0, seed=42: eval/METHODOLOGY.md #15's commitment,
    # made explicit and reviewable here rather than buried in the
    # adapter. Deliberately the most deterministic setting the API
    # offers, held FIXED across all 3 runs (not varied per run) -- the
    # 3 runs exist specifically to test whether the API is actually
    # stable under these settings (OpenAI's own docs note seed-based
    # reproducibility is best-effort, not guaranteed), not to manufacture
    # variance by changing the seed each time.
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        backend="openai", model_name="gpt-4o-mini", temperature=0.0, seed=42,
    ))


@dataclass(frozen=True)
class CorrectiveAgenticConfig:
    # Phase 5 seam: the corrective, self-reflective LangGraph loop
    # (src/adapters/corrective_rag.py) -- grade_documents/rewrite_query
    # around retrieval (CRAG), generate/grade_generation around the answer
    # (Self-RAG). Same deterministic settings as AgenticConfig, for the
    # same reason (eval/METHODOLOGY.md #15): comparability across phases
    # matters more here than picking new values.
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        backend="openai", model_name="gpt-4o-mini", temperature=0.0, seed=42,
    ))
    # Total retrieve+generate passes allowed before a graceful abstention
    # (never a hallucinated answer) -- see eval/METHODOLOGY.md #19 for why
    # 3, and src/adapters/corrective_rag.py's decide_to_generate /
    # decide_after_generation for the only two places a cycle can continue.
    max_iterations: int = 3


@dataclass(frozen=True)
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    corrective_agentic: CorrectiveAgenticConfig = field(default_factory=CorrectiveAgenticConfig)


DEFAULT_CONFIG = Config()
