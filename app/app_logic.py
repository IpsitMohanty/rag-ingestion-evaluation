"""Pure, Streamlit-free query logic for the demo app.

Kept separate from streamlit_app.py so it's testable without Streamlit
installed at all -- mirrors ibm-ai-eng/cnn-vit-land-classification's
demo convention (predict()/CLASS_NAMES importable and testable without
the UI framework in the loop).

Indexes the same corpus, through the same phase 1 pipeline
(adapters/pipeline in src/), that `python src/cli.py ingest` does --
this app is a thin viewer over that pipeline, not a second
implementation of it.
"""
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"

# Matches the convention in tests/conftest.py: src/ modules use plain
# top-level imports (`from config import ...`), so src/ itself must be
# on sys.path, not the repo root.
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from adapters import vectorstore as vectorstore_adapter  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402
from pipeline import ingest as ingest_stage  # noqa: E402
from pipeline import split as split_stage  # noqa: E402
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


def load_corpus_documents() -> list[Document]:
    """Load and split the full corpus (both source types), exactly as
    `python src/cli.py ingest data/faq data/policy` would.
    """
    grouped: dict[str, list[Document]] = {}
    for folder in (DATA_DIR / "faq", DATA_DIR / "policy"):
        for source_type, docs in ingest_stage.ingest_folder(folder).items():
            grouped.setdefault(source_type, []).extend(docs)
    return split_stage.split_documents(grouped, DEFAULT_CONFIG.ingestion)


def build_vectorstore(
    embeddings: Embeddings | None = None, persist_directory: Path | None = None
) -> VectorStore:
    """Build a fresh, combined FAQ+policy index.

    Defaults to the real HuggingFace embeddings backend. Deliberately does
    NOT default to phase 1's shared `.chroma` persist directory: that path
    is meant for `python src/cli.py ingest`, built and re-run independently
    of this app, and a Chroma collection isn't upserted -- adding the
    corpus to an already-populated collection duplicates every document
    rather than replacing it. Streamlit's `@st.cache_resource` already
    guarantees this function runs exactly once per running app instance,
    so a fresh, disposable temp directory per instance is correct and
    avoids ever indexing into a stale or previously-populated collection.
    Tests pass their own tmp_path for the same reason (isolation), not to
    work around this default.
    """
    if embeddings is None:
        embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    if persist_directory is None:
        persist_directory = Path(tempfile.mkdtemp(prefix="poshan-rag-app-"))

    vs_config = replace(DEFAULT_CONFIG.vectorstore, persist_directory=persist_directory)

    store = vectorstore_adapter.get_vectorstore(vs_config, embeddings)
    documents = load_corpus_documents()
    if documents:
        vectorstore_adapter.index_documents(store, documents)
    return store


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
