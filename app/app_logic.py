"""Pure, Streamlit-free query logic for the demo app.

Kept separate from streamlit_app.py so it's testable without Streamlit
installed at all -- mirrors ibm-ai-eng/cnn-vit-land-classification's
demo convention (predict()/CLASS_NAMES importable and testable without
the UI framework in the loop).

Two vectorstore paths, deliberately different:
  - `load_app_vectorstore()` -- what the deployed app actually calls.
    Loads the prebuilt, committed index (app/prebuilt_index/) with
    ONNX query embeddings (app/onnx_embeddings.py). No torch.
  - `build_vectorstore()` / `load_corpus_documents()` -- the dev-only
    path used by app/build_index.py to (re)create that prebuilt index,
    via the real phase 1 pipeline and torch/HuggingFace embeddings.
    Never called by the running app; importing THIS module must not
    require pypdf/langchain-text-splitters/torch to be installed, so
    the src/pipeline imports these two functions need are deferred
    (see their bodies) rather than imported at module load time.
"""
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
PREBUILT_INDEX_DIR = Path(__file__).resolve().parent / "prebuilt_index"

# Matches the convention in tests/conftest.py: src/ modules use plain
# top-level imports (`from config import ...`), so src/ itself must be
# on sys.path, not the repo root.
sys.path.insert(0, str(SRC_DIR))

from adapters import vectorstore as vectorstore_adapter  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.vectorstores import VectorStore  # noqa: E402

# Drawn verbatim from eval/queries.yaml's labeled set (pol-11, faq-01,
# either-01, neither-01) -- the demo's "known unanswerable" example is
# the exact query the evaluation used, not a different, untested one
# invented just for the UI.
PRESET_QUERIES = [
    {
        "label": "FAQ example",
        "query": "How many kinds of beneficiaries can be registered in the Application?",
    },
    {
        "label": "Policy example",
        "query": "What do you understand by Poshan Vatika (Nutri-garden)?",
    },
    {
        "label": "Cross-source example",
        "query": "Why was the Poshan Tracker platform introduced?",
    },
    {
        "label": '⚠️ Known "neither" query -- the corpus cannot answer this',
        "query": "What are Poshan ke Paanch Sutra?",
    },
]


def load_corpus_documents(ingestion_config=None) -> list[Document]:
    """Load and split the full corpus (both source types).

    Defaults to `DEFAULT_CONFIG.ingestion` (raw PDF -- phase 1's general
    default, unchanged, used by `python src/cli.py ingest`).
    `app/build_index.py` passes a cleaned-PDF variant instead: the app's
    prebuilt index uses `clean=True` for demo legibility (results/
    ANALYSIS.md found no retrieval-quality difference between raw and
    cleaned at the swept scale -- this is a presentation choice, not a
    quality claim). Phase 1's own default stays raw; only the app's
    build opts into cleaning.

    Dev-only path (app/build_index.py). Imports are deferred here, not
    at module load time: `pipeline.ingest`/`pipeline.split` pull in
    `adapters.ingestion`, which imports `pypdf` and
    `langchain_text_splitters` at its own top level -- neither is in
    app/requirements.txt, so importing app_logic.py itself (which the
    deployed app does) must not require them.
    """
    from pipeline import ingest as ingest_stage
    from pipeline import split as split_stage

    if ingestion_config is None:
        ingestion_config = DEFAULT_CONFIG.ingestion

    grouped: dict[str, list[Document]] = {}
    for folder in (DATA_DIR / "faq", DATA_DIR / "policy"):
        for source_type, docs in ingest_stage.ingest_folder(folder, ingestion_config).items():
            grouped.setdefault(source_type, []).extend(docs)
    return split_stage.split_documents(grouped, ingestion_config)


def build_vectorstore(
    embeddings: Embeddings | None = None,
    persist_directory: Path | None = None,
    ingestion_config=None,
) -> VectorStore:
    """Build a fresh, combined FAQ+policy index using the real
    HuggingFace/torch embeddings backend.

    Dev-only (app/build_index.py) and test-only path -- the running app
    never calls this; see `load_app_vectorstore()` for that. Deliberately
    does NOT default to phase 1's shared `.chroma` persist directory:
    that path is meant for `python src/cli.py ingest`, built and re-run
    independently of this app, and a Chroma collection isn't upserted --
    adding the corpus to an already-populated collection duplicates every
    document rather than replacing it.
    """
    if embeddings is None:
        from adapters import embeddings as embeddings_adapter

        embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    if persist_directory is None:
        persist_directory = Path(tempfile.mkdtemp(prefix="poshan-rag-app-"))

    vs_config = replace(DEFAULT_CONFIG.vectorstore, persist_directory=persist_directory)

    store = vectorstore_adapter.get_vectorstore(vs_config, embeddings)
    documents = load_corpus_documents(ingestion_config)
    if documents:
        vectorstore_adapter.index_documents(store, documents)
    return store


def load_app_vectorstore(persist_directory: Path | None = None) -> VectorStore:
    """What the deployed app actually calls: load the prebuilt, committed
    index (app/prebuilt_index/, built offline by app/build_index.py using
    the real torch embeddings backend) with ONNX query embeddings.

    No torch, no re-embedding of the corpus at startup -- Chroma only
    calls the embedding function to embed the *query* being searched,
    never to re-embed documents already stored in a loaded collection.
    See app/onnx_embeddings.py's docstring for why this is safe, and
    tests/test_onnx_parity.py for the verification that it actually is.

    `persist_directory` defaults to the committed app/prebuilt_index/;
    tests pass a scratch copy instead, since opening a Chroma directory
    writes internal bookkeeping (HNSW segment/WAL state) even for
    read-only queries, which would otherwise dirty the committed index
    in git on every test run for no functional reason.
    """
    from onnx_embeddings import OnnxEmbeddings

    if persist_directory is None:
        persist_directory = PREBUILT_INDEX_DIR

    vs_config = replace(DEFAULT_CONFIG.vectorstore, persist_directory=persist_directory)
    return vectorstore_adapter.get_vectorstore(vs_config, OnnxEmbeddings())


def describe_source(metadata: dict) -> str:
    """FAQ documents carry `tab`; policy chunks carry `page`. That's
    enough to tell the two apart for display without a dedicated
    source_type metadata field.
    """
    return "faq" if "tab" in metadata else "policy_pdf"


def run_query(vectorstore: VectorStore, query: str, k: int) -> list[dict]:
    """Return up to k results: content, source, metadata, and the raw
    Chroma distance (lower = more similar). See results/ANALYSIS.md
    Finding #1 for why that score can't be read as a confidence gate.

    Empty/whitespace-only queries return no results without touching the
    vectorstore at all -- not an error, just nothing to search for.
    """
    if not query or not query.strip():
        return []

    hits = vectorstore.similarity_search_with_score(query, k=k)
    return [
        {
            "content": document.page_content,
            "source": describe_source(document.metadata),
            "metadata": document.metadata,
            "score": score,
        }
        for document, score in hits
    ]


def generate_answer(query: str, results: list[dict], api_key: str | None) -> str | None:
    """Generate a grounded answer from the retrieved chunks using a
    user-supplied OpenAI API key.

    Returns None (not an exception) when no key is given or there's
    nothing to ground on -- callers should treat None as "skip the
    generation section, show retrieval only."

    `api_key` is never logged, written to a file, or set as an
    environment variable -- it's passed directly to the client
    constructor for this one call and goes out of scope when this
    function returns. On any failure (invalid key, network error, rate
    limit), a short, generic message is returned instead of the raw
    exception, since an SDK error can otherwise echo request details
    back in ways that risk leaking the key into logs or the UI.
    """
    if not api_key or not api_key.strip():
        return None
    if not results:
        return None

    try:
        from langchain_openai import ChatOpenAI

        context = "\n\n".join(f"[{r['source']}] {r['content']}" for r in results)
        prompt = (
            "Answer the question using ONLY the excerpts below. If the "
            "excerpts don't contain the answer, say you don't know rather "
            "than guessing.\n\n"
            f"Excerpts:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, timeout=30)
        response = llm.invoke(prompt)
        return response.content
    except Exception:
        return (
            "Answer generation failed (invalid key, network error, or rate "
            "limit) -- showing retrieved chunks only."
        )
