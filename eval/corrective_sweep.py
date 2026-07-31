"""Phase 5 driver: the corrective/self-reflective LangGraph loop
(src/adapters/corrective_rag.py) run across the same 51-query set and
representative cell phase 4 used (eval/agentic_sweep.py), scored against
that same module's arm A (plain retrieval, never abstains) and frozen
SHOULD_ABSTAIN_IDS ground truth -- reused here, not re-derived, so the two
phases are directly comparable.

Ground-truth caveat (eval/METHODOLOGY.md #19, read before interpreting any
result this module produces): SHOULD_ABSTAIN_IDS was frozen against a
SINGLE-SHOT retrieval that never changes the query. This graph can rewrite
the query and re-retrieve, so a "should abstain" label built from one fixed
retrieval doesn't mean the same thing for both systems. 5 of the 12 ids
(the `neither` queries) are unaffected -- the corpus has no answer no
matter how the query is phrased, so correctly abstaining on those is a
clean, comparable signal. The other 7 are queries the *original* phrasing's
retrieval missed but the corpus can actually answer -- exactly the case
this graph's rewrite loop is built to try to rescue. A non-abstain on one
of those 7 is not a false negative to be scored down; it may be the
corrective loop doing its job. matrices_for_run below still reports the
full 12-id confusion matrix for direct comparability with phase 4's arms,
but rescue_rate_for_run (over just the 7) must be reported alongside it,
never folded silently into the same number.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from . import agentic_sweep as sweep  # noqa: E402
from . import metrics  # noqa: E402

CELL_ID = sweep.CELL_ID
K = sweep.K
N_RUNS = sweep.N_RUNS
SHOULD_ABSTAIN_IDS = sweep.SHOULD_ABSTAIN_IDS
build_representative_cell = sweep.build_representative_cell

# eval/METHODOLOGY.md #19: the 7 of the 12 SHOULD_ABSTAIN_IDS that are
# genuinely answerable from the corpus (single-shot retrieval just missed
# them) -- the corrective loop's actual target, distinct from the 5
# `neither` ids where no rephrasing can help.
RESCUE_TARGET_IDS = SHOULD_ABSTAIN_IDS - {"neither-01", "neither-02", "neither-03", "neither-04", "neither-05"}

# Worst case per query: 1 route call + max_iterations x (grade_documents +
# generate + grade_generation) + (max_iterations - 1) rewrite calls. At
# max_iterations=3: 1 + 9 + 2 = 12. Most queries resolve on pass 1 (4
# calls: route + grade_documents + generate + grade_generation). Budgeted
# at an average of 6 calls/query x 51 queries x 3 runs (eval/METHODOLOGY.md
# #19's estimate) -- a hard multiple of it, not just a warning after the
# fact, mirroring phase 4's HARD_STOP_CALLS.
APPROVED_CALL_ESTIMATE = 918
HARD_STOP_CALLS = int(APPROVED_CALL_ESTIMATE * 1.5)


def fail_open_ids(per_query_run: list[dict]) -> frozenset[str]:
    """Analogous to eval/agentic_sweep.py's fail_open_ids, extended to this
    graph's five LLM call sites instead of phase 4's two. Excluded from the
    scored confusion matrix, always reported as its own category -- same
    discipline as phase 4 (never fold an infra failure into a content
    judgment)."""
    return frozenset(
        pq["id"] for pq in per_query_run
        if pq.get("route_call_failed") or pq.get("doc_grade_call_failed")
        or pq.get("rewrite_call_failed") or pq.get("generation_call_failed")
        or pq.get("faithfulness_call_failed")
    )


def run_corrective_arm(
    graph, llms: list, queries: list[dict], route_enabled: bool, max_iterations: int = 3, n_runs: int = N_RUNS,
) -> list[list[dict]]:
    """llms: the five CachedStructuredLLM wrappers (route, judge, rewrite,
    generate, faithfulness) actually wired into `graph`, so their
    run_index can be advanced once per run -- same reasoning as phase 4's
    run_arm_c (eval/agentic_sweep.py): without this, every run after the
    first would be a guaranteed cache hit on run 0, collapsing the 3-run
    range to zero by construction, not because the API is actually stable.
    """
    runs = []
    for run_index in range(n_runs):
        for llm in llms:
            llm.run_index = run_index
        runs.append(run_single_pass(graph, queries, route_enabled=route_enabled, max_iterations=max_iterations))
    return runs


def run_single_pass(graph, queries: list[dict], route_enabled: bool, max_iterations: int = 3) -> list[dict]:
    """One pass over all queries, at whatever run_index the wrapped LLMs
    are currently set to -- factored out of run_corrective_arm so a caller
    that needs to snapshot call counts between runs (e.g. eval/run_corrective_eval.py's
    per-run cost accounting) can drive run_index itself, one run at a time,
    without going through run_corrective_arm's own internal n_runs loop
    (which always starts counting run_index at 0 -- calling it repeatedly
    with n_runs=1 would silently re-run run_index=0 every time, not 0/1/2).
    """
    from adapters.corrective_rag import run_corrective_query

    per_query = []
    for q in queries:
        result = run_corrective_query(
            graph, q["text"], route_enabled=route_enabled, max_iterations=max_iterations,
        )
        per_query.append({
            "id": q["id"],
            "expected_source": q["expected_source"],
            "text": q["text"],
            "final_query": result.final_query,
            "route": result.route,
            "loops": result.loops,
            "corrective_fired": result.corrective_fired,
            "abstained": result.abstained,
            "abstain_reason": result.abstain_reason,
            "answer": result.answer,
            "grounded": result.grounded,
            "should_abstain": q["id"] in SHOULD_ABSTAIN_IDS,
            "route_call_failed": result.route_call_failed,
            "doc_grade_call_failed": result.doc_grade_call_failed,
            "rewrite_call_failed": result.rewrite_call_failed,
            "generation_call_failed": result.generation_call_failed,
            "faithfulness_call_failed": result.faithfulness_call_failed,
            # Observability (eval/METHODOLOGY.md #19, README's Phase 5 section):
            # the actual node path taken, so a reader can audit *why* this
            # query looped or abstained, not just its final answer.
            "trace": result.trace,
        })
    return per_query


def matrices_for_run(per_query_run: list[dict], exclude_ids: frozenset[str] = frozenset()) -> sweep.ConfusionMatrix:
    predictions = {pq["id"]: pq["abstained"] for pq in per_query_run if pq["id"] not in exclude_ids}
    ground_truth = {pq["id"]: pq["should_abstain"] for pq in per_query_run if pq["id"] not in exclude_ids}
    return sweep.confusion_matrix(predictions, ground_truth)


def rescue_rate_for_run(per_query_run: list[dict], exclude_ids: frozenset[str] = frozenset()) -> dict:
    """Fraction of RESCUE_TARGET_IDS (the 7 genuinely-answerable misses)
    the corrective loop did NOT abstain on -- reported separately from
    matrices_for_run's confusion matrix, per this module's docstring:
    a non-abstain here is the intended win condition, not a false negative.
    """
    scored = [pq for pq in per_query_run if pq["id"] in RESCUE_TARGET_IDS and pq["id"] not in exclude_ids]
    if not scored:
        return {"n": 0, "rescued": 0, "rate": None}
    rescued = sum(1 for pq in scored if not pq["abstained"])
    return {"n": len(scored), "rescued": rescued, "rate": rescued / len(scored)}


def faithfulness_rate_for_run(per_query_run: list[dict], exclude_ids: frozenset[str] = frozenset()) -> float | None:
    grounded_flags = [
        pq["grounded"] for pq in per_query_run
        if pq["id"] not in exclude_ids and pq["grounded"] is not None
    ]
    return metrics.faithfulness_rate(grounded_flags)


def mean_loops_for_run(per_query_run: list[dict], exclude_ids: frozenset[str] = frozenset()) -> float:
    return metrics.mean_loops([pq["loops"] for pq in per_query_run if pq["id"] not in exclude_ids])


def corrective_fire_rate_for_run(per_query_run: list[dict], exclude_ids: frozenset[str] = frozenset()) -> float:
    return metrics.corrective_fire_rate(
        [pq["corrective_fired"] for pq in per_query_run if pq["id"] not in exclude_ids]
    )
