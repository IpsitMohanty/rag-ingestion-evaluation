"""Metric implementations. Scoring rules match eval/METHODOLOGY.md exactly
-- this module is the enforcement point, not the sweep runner.
"""
import re
import statistics
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "as", "or", "and", "but", "if", "then", "this", "that",
    "these", "those", "it", "its", "i", "you", "your", "we", "us", "our", "they", "their",
    "what", "which", "who", "whom", "how", "why", "when", "where", "does", "do", "did", "can",
    "could", "should", "would", "will", "shall", "has", "have", "had", "not", "no", "so", "into",
    "out", "up", "down", "about", "app",
}


def normalized_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard_overlap(a: str, b: str) -> float:
    ta, tb = normalized_tokens(a), normalized_tokens(b)
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


@dataclass
class ScoredHit:
    """One retrieved result, tagged with enough identity to check against
    a query's ground_truth without re-deriving it from page_content.
    """
    source_type: str  # "faq" | "policy_pdf"
    score: float  # raw distance from the embedding's vectorstore -- lower is more similar
    document: Document
    faq_index: int | None = None  # format_aware: this chunk IS row faq_index
    faq_indices: tuple[int, ...] | None = None  # uniform: this chunk OVERLAPS these rows
    page: int | None = None  # policy_pdf page number


def _hit_matches_ref(hit: ScoredHit, ref: dict[str, Any]) -> bool:
    if ref["source"] == "faq" and hit.source_type == "faq":
        if hit.faq_index is not None:
            return hit.faq_index == ref["faq_index"]
        if hit.faq_indices is not None:
            return ref["faq_index"] in hit.faq_indices
        return False
    if ref["source"] == "policy_pdf" and hit.source_type == "policy_pdf":
        return hit.page == ref["page"]
    return False


def hit_at_k(hits: list[ScoredHit], ground_truth: list[dict[str, Any]]) -> bool:
    """either-scoring is inherent here: True if ANY hit matches ANY ref,
    regardless of which source the ref or the hit came from (see
    METHODOLOGY.md #3)."""
    return any(_hit_matches_ref(h, ref) for h in hits for ref in ground_truth)


def reciprocal_rank(hits: list[ScoredHit], ground_truth: list[dict[str, Any]]) -> float:
    for rank, h in enumerate(hits, start=1):
        if any(_hit_matches_ref(h, ref) for ref in ground_truth):
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(all_hits: list[list[ScoredHit]], all_ground_truth: list[list[dict]]) -> float:
    rrs = [reciprocal_rank(hits, gt) for hits, gt in zip(all_hits, all_ground_truth)]
    return statistics.mean(rrs) if rrs else 0.0


def hit_rate(all_hits: list[list[ScoredHit]], all_ground_truth: list[list[dict]]) -> float:
    scores = [hit_at_k(hits, gt) for hits, gt in zip(all_hits, all_ground_truth)]
    return sum(scores) / len(scores) if scores else 0.0


def source_routing_correct(hits: list[ScoredHit], expected_source: str) -> bool:
    """Top-1 source type matches the query's single correct source type.
    Only meaningful for expected_source in {faq, policy_pdf} -- callers
    must not call this for "either"/"neither" (see METHODOLOGY.md #3)."""
    if expected_source not in ("faq", "policy_pdf"):
        raise ValueError(
            f"source_routing_correct is undefined for expected_source={expected_source!r}; "
            "either/neither queries are excluded from this metric by design"
        )
    if not hits:
        return False
    return hits[0].source_type == expected_source


def top1_score(hits: list[ScoredHit]) -> float | None:
    return hits[0].score if hits else None


def _distribution_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def confidence_separation(neither_scores: list[float], should_hit_scores: list[float]) -> dict:
    """Compares top-1 distance distributions. Distance is lower-is-better,
    so good separation means neither_scores are systematically HIGHER
    (worse/less confident) than should_hit_scores. `ranges_overlap=True`
    means the two distributions intersect -- the retriever can't tell
    "I found it" from "this is the least-bad option" (METHODOLOGY.md #3).
    """
    neither_stats = _distribution_stats(neither_scores)
    hit_stats = _distribution_stats(should_hit_scores)
    ranges_overlap = None
    if neither_scores and should_hit_scores:
        ranges_overlap = not (
            max(neither_scores) < min(should_hit_scores)
            or max(should_hit_scores) < min(neither_scores)
        )
    return {"neither": neither_stats, "should_hit": hit_stats, "ranges_overlap": ranges_overlap}
