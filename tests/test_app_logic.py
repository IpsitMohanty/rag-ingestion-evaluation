"""Coverage: the Streamlit demo's query path (app/app_logic.py).

app_logic.py never imports streamlit -- app/streamlit_app.py is UI wiring
only, imported by nothing here -- so these tests exercise the same logic
the deployed app runs without needing streamlit installed. Mirrors
ibm-ai-eng/cnn-vit-land-classification's test_demo_streamlit.py convention
(import the demo's logic directly, keep the UI framework out of the test).

Uses a small synthetic vectorstore (fake embeddings, a handful of FAQ- and
policy-shaped Documents), not the real corpus -- ingestion/splitting of
the real 126-FAQ/77-page-PDF corpus is already covered by
test_ingestion_faq.py/test_ingestion_pdf.py; re-parsing the PDF here too
would just duplicate that cost and push the suite over budget. This file
tests app_logic's own contract (metadata/score shape, empty-query
handling, API-key degradation), not the ingestion pipeline underneath it.
"""
import sys
import types
from pathlib import Path

import pytest
from langchain_core.documents import Document

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from app_logic import (  # noqa: E402
    PRESET_QUERIES,
    describe_source,
    generate_answer,
    run_query,
)
from adapters import vectorstore as vectorstore_adapter  # noqa: E402
from config import VectorStoreConfig  # noqa: E402

SYNTHETIC_DOCS = [
    Document(
        page_content="Question: How many kinds of beneficiaries can be registered?\nAnswer: Six kinds.",
        metadata={"source": "faq.json", "tab": "Beneficiary", "subcategory": "General", "question": "q1"},
    ),
    Document(
        page_content="Question: What is a Home Visit?\nAnswer: A monthly visit by an AWW.",
        metadata={"source": "faq.json", "tab": "Poshan Tracker Application", "subcategory": "Home Visit", "question": "q2"},
    ),
    Document(
        page_content="Question: What is Saksham AWC?\nAnswer: An upgraded Anganwadi Centre.",
        metadata={"source": "faq.json", "tab": "Officials (Dashboard)", "subcategory": "", "question": "q3"},
    ),
    Document(
        page_content="Poshan Vatika is a nutrition garden cultivated at or near the Anganwadi Centre.",
        metadata={"source": "policy.pdf", "page": 28},
    ),
    Document(
        page_content="The Anganwadi Services Scheme provides a package of six services to beneficiaries.",
        metadata={"source": "policy.pdf", "page": 18},
    ),
    Document(
        page_content="Severely malnourished children receive enhanced nutrition under the scheme.",
        metadata={"source": "policy.pdf", "page": 22},
    ),
]


@pytest.fixture(scope="module")
def vectorstore(fake_embeddings, tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("app_chroma")
    config = VectorStoreConfig(
        backend="chroma", persist_directory=persist_dir, collection_name="app-logic-test"
    )
    store = vectorstore_adapter.get_vectorstore(config, fake_embeddings)
    vectorstore_adapter.index_documents(store, SYNTHETIC_DOCS)
    return store


def test_preset_queries_include_a_known_neither_query():
    """The brief's explicit requirement: one preset must be the corpus's
    known-unanswerable probe, not a synthetic one."""
    texts = [p["query"] for p in PRESET_QUERIES]
    assert "What are Poshan ke Paanch Sutra?" in texts


def test_valid_query_returns_results_with_metadata_intact(vectorstore):
    results = run_query(vectorstore, "beneficiary registration", k=5)
    assert results
    assert len(results) <= 5
    for r in results:
        assert set(r) == {"content", "source", "metadata", "score"}
        assert r["source"] in {"faq", "policy_pdf"}
        assert isinstance(r["score"], float)
        assert r["content"].strip()


def test_valid_query_k_is_respected(vectorstore):
    for k in (1, 3, len(SYNTHETIC_DOCS)):
        results = run_query(vectorstore, "beneficiary registration", k=k)
        assert len(results) <= k


def test_empty_query_returns_no_results_without_touching_the_store(vectorstore):
    assert run_query(vectorstore, "", k=5) == []
    assert run_query(vectorstore, "   ", k=5) == []


def test_describe_source_distinguishes_faq_from_policy():
    assert describe_source({"tab": "Beneficiary", "subcategory": ""}) == "faq"
    assert describe_source({"page": 12, "source": "x.pdf"}) == "policy_pdf"


def test_no_api_key_path_returns_none_without_any_network_call():
    """The default, no-key path must work with zero network activity --
    generate_answer must short-circuit before ever importing langchain_openai."""
    results = [{"content": "some chunk", "source": "faq", "metadata": {}, "score": 0.5}]
    assert generate_answer("a question", results, api_key=None) is None
    assert generate_answer("a question", results, api_key="") is None
    assert generate_answer("a question", results, api_key="   ") is None


def test_generate_answer_with_no_results_returns_none_even_with_a_key():
    assert generate_answer("a question", [], api_key="sk-fake-key") is None


def test_generate_answer_degrades_cleanly_on_invalid_key(monkeypatch):
    """An invalid/fake key must not crash the app -- it should return a
    clean failure message, never raise, and never echo the key itself.

    Installs a fake `langchain_openai` module into sys.modules rather than
    monkeypatching the real package -- this test must not depend on
    langchain_openai being installed (it isn't in requirements-dev.txt,
    only requirements-app.txt) or on a real network round-trip to OpenAI
    (this suite stays network-free, same reasoning as
    tests/conftest.py's requires_network gate).
    """

    class _FakeAuthError(Exception):
        pass

    class _FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt):
            raise _FakeAuthError("Incorrect API key provided: sk-definitely-not-a-real-key")

    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = _FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    results = [{"content": "some chunk", "source": "faq", "metadata": {}, "score": 0.5}]
    message = generate_answer("a question", results, api_key="sk-definitely-not-a-real-key")
    assert message is not None
    assert "sk-definitely-not-a-real-key" not in message
