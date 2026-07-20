"""Coverage: the eval harness is deterministic -- running the same cell
twice produces bit-identical metrics (see eval/METHODOLOGY.md #7: this is
what's asserted here instead of repeating the real sweep to "measure"
variance that a deterministic pipeline doesn't have).
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from eval import retrievers
from eval.sweep import evaluate_cell
from config import RetrieverConfig

QUERIES = [
    {"id": "q1", "text": "email policy details", "expected_source": "faq",
     "ground_truth": [{"source": "faq", "faq_index": 0}]},
    {"id": "q2", "text": "smoking rules at work", "expected_source": "policy_pdf",
     "ground_truth": [{"source": "policy_pdf", "page": 5}]},
]
FAQ_DOCS = [
    Document(page_content="email policy details", metadata={"faq_index": 0}),
    Document(page_content="vacation policy details", metadata={"faq_index": 1}),
]
POLICY_CHUNKS = [
    Document(page_content="smoking rules at work", metadata={"page": 5}),
    Document(page_content="parking rules at work", metadata={"page": 6}),
]


def _run_once(tmp_path, suffix):
    embeddings = DeterministicFakeEmbedding(size=16)
    faq_store = retrievers.build_faq_index(FAQ_DOCS, embeddings, tmp_path / f"faq-{suffix}")
    policy_index = retrievers.build_policy_index(
        "similarity", POLICY_CHUNKS, POLICY_CHUNKS, embeddings,
        tmp_path / f"policy-{suffix}", RetrieverConfig(strategy="similarity", k=5),
    )
    return evaluate_cell(QUERIES, faq_store, policy_index)


def test_repeated_runs_produce_identical_results(tmp_path):
    first = _run_once(tmp_path, "a")
    second = _run_once(tmp_path, "b")
    # JSON round-trip so the comparison is over the same plain-data shape
    # the real sweep serializes to, not incidental float/object identity.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
