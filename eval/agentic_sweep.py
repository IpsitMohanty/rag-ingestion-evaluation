"""Phase 4 driver: four arms on one cell, per eval/METHODOLOGY.md #9-18.
That document is frozen and this module implements it; it does not
re-decide anything settled there.

Arms:
  A  baseline similarity retrieval, no abstention mechanism at all.
  B  same retrieval as A, plus the best-possible in-sample distance
     cutoff (Youden's J).
  C1 LLM judge only, no routing gate, judges A/B's own retrieved chunks.
  C2 full path: LLM router gates retrieval, then the judge sees only
     what that routed retrieval returned.

Cell (single, deliberate, no sweep -- METHODOLOGY.md #10):
`cleaned_pdf__format_aware__cs800_ov100__similarity`, k=5. Same cell the
deployed app runs.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AgenticConfig, RetrieverConfig  # noqa: E402

from . import corpus, metrics, retrievers  # noqa: E402
from .llm_cache import CachedStructuredLLM  # noqa: E402
from .query_set import load_query_set  # noqa: E402

PDF_VARIANT = "cleaned"
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100
INGESTION_MODE = "format_aware"
RETRIEVER_STRATEGY = "similarity"
K = 5
CELL_ID = f"{PDF_VARIANT}_pdf__{INGESTION_MODE}__cs{CHUNK_SIZE}_ov{CHUNK_OVERLAP}__{RETRIEVER_STRATEGY}"

N_RUNS = 3

# eval/METHODOLOGY.md #11: frozen ground truth for arms A/B/C1. Do not
# recompute or modify -- verified against the representative cell at
# harness startup (see _verify_frozen_ground_truth) as a drift check,
# not as a source of truth.
SHOULD_ABSTAIN_IDS = frozenset({
    "neither-01", "neither-02", "neither-03", "neither-04", "neither-05",
    "pol-08", "pol-09", "pol-10", "faq-01", "faq-05", "faq-14", "faq-16",
})

# eval/METHODOLOGY.md #12: of the 4 retrievable_but_incomplete queries,
# these 3 are hits (in the 39 "should not abstain"), so a judge abstaining
# on them is the honest-judgment-vs-hit-rate-credit tension, not a bug.
# pol-10 is excluded from this set: it's already in SHOULD_ABSTAIN_IDS
# for an unrelated reason (its target chunk was missed at k=5, not merely
# incomplete).
RETRIEVABLE_BUT_INCOMPLETE_HITS = frozenset({"pol-01", "pol-05", "either-08"})

# eval/METHODOLOGY.md #16: approved cost envelope. A hard multiple of it
# to stop spending against, not just warn after the fact.
APPROVED_CALL_ESTIMATE = 459
HARD_STOP_CALLS = int(APPROVED_CALL_ESTIMATE * 1.5)


@dataclass
class ConfusionMatrix:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def specificity(self) -> float | None:
        denom = self.tn + self.fp
        return self.tn / denom if denom else None

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "recall": self.recall,
            "precision": self.precision if (self.tp + self.fp) else "undefined, no abstain predictions made",
        }


def confusion_matrix(predictions: dict[str, bool], ground_truth: dict[str, bool]) -> ConfusionMatrix:
    """predictions/ground_truth: query_id -> "should abstain" (True) /
    "should not abstain" (False). Positive class = should abstain, per
    METHODOLOGY.md #13.
    """
    tp = fp = fn = tn = 0
    for qid, truth in ground_truth.items():
        pred = predictions[qid]
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif truth and not pred:
            fn += 1
        else:
            tn += 1
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def _filtered(d: dict[str, bool], exclude_ids: frozenset[str]) -> dict[str, bool]:
    return {qid: v for qid, v in d.items() if qid not in exclude_ids}


def build_representative_cell(embeddings, work_dir: Path):
    """Builds the single cell's FAQ + policy sub-indexes, reusing
    eval/corpus.py and eval/retrievers.py unchanged -- the same
    deterministic, real-embeddings retrieval phase 2's sweep already
    exercises, just one cell instead of 24 (METHODOLOGY.md #10).
    """
    policy_raw_pages = corpus.build_policy_raw_pages(PDF_VARIANT)
    faq_docs = corpus.build_faq_documents(INGESTION_MODE, CHUNK_SIZE, CHUNK_OVERLAP)
    policy_chunks = corpus.build_policy_chunks(CHUNK_SIZE, CHUNK_OVERLAP, PDF_VARIANT)

    retriever_config = RetrieverConfig(strategy=RETRIEVER_STRATEGY, k=K)
    faq_store = retrievers.build_faq_index(faq_docs, embeddings, work_dir / "faq")
    policy_index = retrievers.build_policy_index(
        RETRIEVER_STRATEGY, policy_chunks, policy_raw_pages, embeddings, work_dir / "policy", retriever_config,
    )
    return faq_store, policy_index


def should_hit_ground_truth(query: dict, hits: list) -> bool:
    """True = should abstain, for a single arm's own retrieved hits
    (METHODOLOGY.md #11). `neither` queries are arm-invariant (always
    True); should-hit queries abstain iff their target chunk is absent
    from `hits`, reusing eval.metrics.hit_at_k unchanged (its existing
    OR-logic already handles `either` queries' multiple valid refs).
    """
    if query["expected_source"] == "neither":
        return True
    return not metrics.hit_at_k(hits, query["ground_truth"])


def _verify_frozen_ground_truth(queries: list[dict], baseline_hits: dict[str, list]) -> None:
    """Drift check, not a source of truth: the frozen SHOULD_ABSTAIN_IDS
    (METHODOLOGY.md #11) must match what the representative cell's
    deterministic retrieval actually produces right now. A mismatch means
    the corpus, chunking, or embedding model has changed since the labels
    were frozen, and must be investigated, not silently overridden.
    """
    mismatches = []
    for q in queries:
        expected = q["id"] in SHOULD_ABSTAIN_IDS
        actual = should_hit_ground_truth(q, baseline_hits[q["id"]])
        if expected != actual:
            mismatches.append((q["id"], expected, actual))
    if mismatches:
        raise RuntimeError(
            f"Frozen ground truth (METHODOLOGY.md #11) no longer matches the representative "
            f"cell's actual retrieval: {mismatches}. Do not silently adjust SHOULD_ABSTAIN_IDS -- "
            f"investigate whether the corpus/embeddings/chunking changed since the labels were frozen."
        )


def arm_a_matrix(ground_truth: dict[str, bool]) -> ConfusionMatrix:
    """Never abstains, by construction (METHODOLOGY.md #13): this IS
    Finding #1 restated as a confusion matrix, not a new result.
    """
    predictions = {qid: False for qid in ground_truth}
    return confusion_matrix(predictions, ground_truth)


def best_threshold_by_youdens_j(
    scores: dict[str, float], ground_truth: dict[str, bool]
) -> tuple[float, ConfusionMatrix]:
    """METHODOLOGY.md #14: maximize Youden's J (sensitivity + specificity
    - 1) over every distinct observed top-1 distance as a candidate
    cutoff, in-sample. Rule: predict "should abstain" if
    top1_distance >= cutoff. Not accuracy: a never-abstain rule already
    clears 76.5% accuracy here, which would bias cutoff selection toward
    ignoring the minority ("should abstain") class entirely.
    """
    candidates = sorted(set(scores.values()))
    best_j = None
    best_cutoff = None
    best_cm = None
    for cutoff in candidates:
        predictions = {qid: (score >= cutoff) for qid, score in scores.items()}
        cm = confusion_matrix(predictions, ground_truth)
        sensitivity = cm.recall or 0.0
        specificity = cm.specificity or 0.0
        j = sensitivity + specificity - 1
        if best_j is None or j > best_j:
            best_j, best_cutoff, best_cm = j, cutoff, cm
    return best_cutoff, best_cm


def run_arm_c(
    graph, route_llm, judge_llm, queries: list[dict], route_enabled: bool, n_runs: int = N_RUNS,
) -> list[dict]:
    """Runs one of C1 (route_enabled=False) or C2 (route_enabled=True)
    n_runs times over all 51 queries. Ground truth is NOT precomputed
    here: for C2 it must reflect what THAT run actually retrieved
    (METHODOLOGY.md #11), so it's derived per query per run below, not
    passed in.

    route_llm/judge_llm are the CachedStructuredLLM wrappers actually
    wired into `graph` (via build_agentic_graph's closures) -- passed in
    here too so their `.run_index` can be advanced once per run. Without
    this, every run after the first would be a guaranteed cache hit on
    run 1 (identical query, identical prompt), collapsing the 3-run
    range this whole method exists to measure down to zero by
    construction, not because the API is actually stable.
    """
    from adapters.agentic import run_agentic_query

    runs = []
    for run_index in range(n_runs):
        route_llm.run_index = run_index
        judge_llm.run_index = run_index
        per_query = []
        for q in queries:
            result = run_agentic_query(graph, q["text"], route_enabled=route_enabled)
            ground_truth = (
                q["id"] in SHOULD_ABSTAIN_IDS if not route_enabled
                else should_hit_ground_truth(q, result.hits)
            )
            per_query.append({
                "id": q["id"],
                "expected_source": q["expected_source"],
                "route": result.route,
                "abstained": result.abstained,
                "should_abstain": ground_truth,
                "route_call_failed": result.route_call_failed,
                "judge_call_failed": result.judge_call_failed,
            })
        runs.append(per_query)
    return runs


def fail_open_ids(per_query_run: list[dict]) -> frozenset[str]:
    """Queries where the route or judge call failed and fell back to its
    safe default (src/adapters/agentic.py's fail-open behavior). This is
    correct production behavior but must never be folded silently into
    "the judge chose to answer": a fail-open on any of the 12
    should-abstain queries would look like a false negative caused by
    judge quality when it was actually caused by an infra failure.
    Excluded from the scored confusion matrix (see matrices_for_run's
    exclude_ids) and always reported as its own category, per the
    pre-run review's blocking fix.
    """
    return frozenset(
        pq["id"] for pq in per_query_run if pq.get("route_call_failed") or pq.get("judge_call_failed")
    )


def routing_accuracy(per_query_run: list[dict]) -> float | None:
    """METHODOLOGY.md #17: C2 only, faq/policy_pdf queries only (either
    has no single correct route, neither has no correct source at all --
    same exclusion as phase 2's source_routing_correct, METHODOLOGY.md
    #3). Measures route's own explicit decision, not the emergent
    top-1-hit-source phase 2 measured; the two are related, not the same
    metric.
    """
    scored = [pq for pq in per_query_run if pq["expected_source"] in ("faq", "policy_pdf")]
    if not scored:
        return None
    correct = sum(1 for pq in scored if pq["route"] == pq["expected_source"])
    return correct / len(scored)


def matrices_for_run(per_query_run: list[dict], exclude_ids: frozenset[str] = frozenset()) -> ConfusionMatrix:
    predictions = {pq["id"]: pq["abstained"] for pq in per_query_run if pq["id"] not in exclude_ids}
    ground_truth = {pq["id"]: pq["should_abstain"] for pq in per_query_run if pq["id"] not in exclude_ids}
    return confusion_matrix(predictions, ground_truth)


def range_across_runs(values: list[float | None]) -> float | None:
    """METHODOLOGY.md #15: the variance bar's committed functional form
    is the full 3-run range (max - min), not a standard deviation --
    n=3 is too small to estimate a standard deviation reliably, and range
    is the more conservative (wider) of the two.
    """
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return max(clean) - min(clean)
