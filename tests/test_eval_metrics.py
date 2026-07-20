"""Coverage: eval/metrics.py correctness on hand-computed toy cases --
scoring rules must match eval/METHODOLOGY.md exactly.
"""
import pytest
from langchain_core.documents import Document

from eval.metrics import (
    ScoredHit,
    confidence_separation,
    hit_at_k,
    jaccard_overlap,
    mean_reciprocal_rank,
    reciprocal_rank,
    source_routing_correct,
)


def _hit(source_type, score, faq_index=None, faq_indices=None, page=None):
    return ScoredHit(
        source_type=source_type, score=score, document=Document(page_content="x"),
        faq_index=faq_index, faq_indices=faq_indices, page=page,
    )


def test_hit_at_k_true_on_exact_faq_match():
    hits = [_hit("faq", 0.1, faq_index=5), _hit("policy_pdf", 0.5, page=3)]
    assert hit_at_k(hits, [{"source": "faq", "faq_index": 5}]) is True


def test_hit_at_k_false_when_nothing_matches():
    hits = [_hit("faq", 0.1, faq_index=5)]
    assert hit_at_k(hits, [{"source": "faq", "faq_index": 6}]) is False


def test_hit_at_k_matches_uniform_chunk_via_faq_indices():
    hits = [_hit("faq", 0.2, faq_indices=(3, 4, 5))]
    assert hit_at_k(hits, [{"source": "faq", "faq_index": 4}]) is True
    assert hit_at_k(hits, [{"source": "faq", "faq_index": 9}]) is False


def test_hit_at_k_policy_pdf_matches_on_page():
    hits = [_hit("policy_pdf", 0.3, page=22)]
    assert hit_at_k(hits, [{"source": "policy_pdf", "page": 22}]) is True
    assert hit_at_k(hits, [{"source": "policy_pdf", "page": 23}]) is False


def test_hit_at_k_either_or_logic_across_sources():
    """METHODOLOGY.md #3: a hit from EITHER listed ground truth ref counts,
    even if only the policy_pdf one, not the faq one, was actually hit."""
    ground_truth = [{"source": "faq", "faq_index": 116}, {"source": "policy_pdf", "page": 16}]
    only_policy_hit = [_hit("policy_pdf", 0.4, page=16)]
    assert hit_at_k(only_policy_hit, ground_truth) is True
    only_faq_hit = [_hit("faq", 0.4, faq_index=116)]
    assert hit_at_k(only_faq_hit, ground_truth) is True
    neither_hit = [_hit("policy_pdf", 0.4, page=99)]
    assert hit_at_k(neither_hit, ground_truth) is False


def test_reciprocal_rank_hand_computed():
    ground_truth = [{"source": "faq", "faq_index": 5}]
    hits = [_hit("faq", 0.1, faq_index=1), _hit("faq", 0.2, faq_index=5), _hit("faq", 0.3, faq_index=2)]
    assert reciprocal_rank(hits, ground_truth) == pytest.approx(1 / 2)


def test_reciprocal_rank_zero_when_absent():
    hits = [_hit("faq", 0.1, faq_index=1)]
    assert reciprocal_rank(hits, [{"source": "faq", "faq_index": 99}]) == 0.0


def test_mean_reciprocal_rank_hand_computed():
    # query A: hit at rank 1 (rr=1.0); query B: hit at rank 4 (rr=0.25)
    all_hits = [
        [_hit("faq", 0.1, faq_index=1)],
        [_hit("faq", 0.1, faq_index=9), _hit("faq", 0.2, faq_index=8),
         _hit("faq", 0.3, faq_index=7), _hit("faq", 0.4, faq_index=1)],
    ]
    all_gt = [[{"source": "faq", "faq_index": 1}], [{"source": "faq", "faq_index": 1}]]
    assert mean_reciprocal_rank(all_hits, all_gt) == pytest.approx((1.0 + 0.25) / 2)


def test_source_routing_correct_checks_top1_only():
    hits = [_hit("policy_pdf", 0.1, page=1), _hit("faq", 0.2, faq_index=1)]
    assert source_routing_correct(hits, "policy_pdf") is True
    assert source_routing_correct(hits, "faq") is False


def test_source_routing_correct_false_on_empty_hits():
    assert source_routing_correct([], "faq") is False


def test_source_routing_correct_rejects_either_and_neither():
    """METHODOLOGY.md #3: routing accuracy is undefined for either/neither -- calling it that way is a bug, not a valid zero."""
    with pytest.raises(ValueError):
        source_routing_correct([_hit("faq", 0.1, faq_index=1)], "either")
    with pytest.raises(ValueError):
        source_routing_correct([_hit("faq", 0.1, faq_index=1)], "neither")


def test_confidence_separation_detects_no_overlap():
    neither = [10.0, 11.0, 12.0]
    should_hit = [1.0, 2.0, 3.0]
    result = confidence_separation(neither, should_hit)
    assert result["ranges_overlap"] is False
    assert result["neither"]["mean"] == pytest.approx(11.0)
    assert result["should_hit"]["mean"] == pytest.approx(2.0)


def test_confidence_separation_detects_overlap():
    neither = [1.0, 5.0, 9.0]
    should_hit = [2.0, 4.0, 6.0]
    result = confidence_separation(neither, should_hit)
    assert result["ranges_overlap"] is True


def test_confidence_separation_handles_empty_input():
    result = confidence_separation([], [1.0, 2.0])
    assert result["ranges_overlap"] is None
    assert result["neither"]["n"] == 0


def test_jaccard_overlap_identical_text_is_one():
    assert jaccard_overlap("what is the home visit feature", "what is the home visit feature") == 1.0


def test_jaccard_overlap_no_shared_tokens_is_zero():
    assert jaccard_overlap("who can be a nominee", "does the app support hindi") == 0.0


def test_jaccard_overlap_hand_computed():
    # normalized content tokens: {profile, rch} vs {profile, rch} minus stopwords "what/is/the"
    a = "What does the RCH Profile feature refer to?"
    b = "What is RCH Profile?"
    # shared content tokens: {rch, profile}; union: {rch, profile, feature, refer}
    assert jaccard_overlap(a, b) == pytest.approx(2 / 4)
