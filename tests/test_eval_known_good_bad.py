"""Coverage: the harness isn't always-passing -- a query engineered to
obviously match returns a hit, and one engineered to be unreachable
(its ground-truth document isn't even in the index) scores zero. This
exercises the real retrieval -> scoring wiring, not just eval/metrics.py
in isolation (see tests/test_eval_metrics.py for that).
"""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import RetrieverConfig
from eval import retrievers
from eval.sweep import evaluate_cell

FAQ_DOCS = [
    Document(page_content="How do I reset my forgotten MPIN?", metadata={"faq_index": 0}),
    Document(page_content="What vaccines are given to pregnant women?", metadata={"faq_index": 1}),
]
POLICY_CHUNKS = [
    Document(page_content="Anganwadi Centres provide supplementary nutrition services.", metadata={"page": 10}),
]


def _build(tmp_path, suffix):
    embeddings = DeterministicFakeEmbedding(size=16)
    faq_store = retrievers.build_faq_index(FAQ_DOCS, embeddings, tmp_path / f"faq-{suffix}")
    policy_index = retrievers.build_policy_index(
        "similarity", POLICY_CHUNKS, POLICY_CHUNKS, embeddings,
        tmp_path / f"policy-{suffix}", RetrieverConfig(strategy="similarity", k=5),
    )
    return faq_store, policy_index


def test_known_good_query_returns_a_hit(tmp_path):
    """Querying with a document's OWN exact text guarantees a top rank
    match under any embedding backend (identical text -> identical
    vector -> zero distance), independent of semantic quality."""
    faq_store, policy_index = _build(tmp_path, "good")
    queries = [{
        "id": "known-good", "text": "How do I reset my forgotten MPIN?",
        "expected_source": "faq", "ground_truth": [{"source": "faq", "faq_index": 0}],
    }]
    result = evaluate_cell(queries, faq_store, policy_index)
    assert result["buckets"]["faq"]["hit_rate_at_k"][1] == 1.0
    assert result["buckets"]["faq"]["mrr"] == 1.0


def test_known_bad_query_scores_zero(tmp_path):
    """Ground truth points at a faq_index that was never indexed at all
    -- no embedding backend, however good, could ever retrieve it, so
    this must score exactly zero. Proves the harness can fail, not just
    always report success."""
    faq_store, policy_index = _build(tmp_path, "bad")
    queries = [{
        "id": "known-bad", "text": "How do I reset my forgotten MPIN?",
        "expected_source": "faq", "ground_truth": [{"source": "faq", "faq_index": 999}],
    }]
    result = evaluate_cell(queries, faq_store, policy_index)
    assert result["buckets"]["faq"]["hit_rate_at_k"][1] == 0.0
    assert result["buckets"]["faq"]["hit_rate_at_k"][10] == 0.0
    assert result["buckets"]["faq"]["mrr"] == 0.0
