"""Entry point for the phase 4 evaluation (eval/METHODOLOGY.md #9-18).

    OPENAI_API_KEY=... python -m eval.run_agentic_eval

Requires OPENAI_API_KEY (route + judge calls) and the
sentence-transformers/all-MiniLM-L6-v2 model available locally
(HF_HUB_OFFLINE=1 once cached -- see README). Not run in CI: this makes
real, paid LLM calls, gated on a real API key exactly like the practical
constraints in the phase 4 brief require. Estimated cost: ~$0.06 for
~459 calls (route + judge, 3 runs, C1 + C2) -- see the printed summary
for the actual count and a hard stop at 1.5x that if something is
calling more than expected.
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from adapters.agentic import JudgeDecision, RouteDecision, build_agentic_graph, get_structured_llms  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

from eval import agentic_sweep as sweep  # noqa: E402
from eval import metrics  # noqa: E402
from eval.llm_cache import CachedStructuredLLM, load_cache, save_cache  # noqa: E402
from eval.query_set import load_query_set  # noqa: E402
from eval.retrievers import query_combined  # noqa: E402


def main() -> None:
    queries = load_query_set()
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        faq_store, policy_index = sweep.build_representative_cell(embeddings, Path(tmp))

        # Arms A/B share this retrieval; C1 reproduces it deterministically
        # via route_enabled=False (verified below, not assumed).
        baseline_hits = {q["id"]: query_combined(q["text"], sweep.K, faq_store, policy_index) for q in queries}
        sweep._verify_frozen_ground_truth(queries, baseline_hits)

        ground_truth = {q["id"]: (q["id"] in sweep.SHOULD_ABSTAIN_IDS) for q in queries}

        # Arm A
        arm_a = sweep.arm_a_matrix(ground_truth)

        # Arm B
        top1_scores = {q["id"]: metrics.top1_score(baseline_hits[q["id"]]) for q in queries}
        top1_scores = {qid: s for qid, s in top1_scores.items() if s is not None}
        cutoff, arm_b = sweep.best_threshold_by_youdens_j(top1_scores, ground_truth)

        # Arms C1/C2
        route_config = DEFAULT_CONFIG.agentic
        base_route_llm, base_judge_llm = get_structured_llms(route_config)
        cache = load_cache()
        route_llm = CachedStructuredLLM(base_route_llm, RouteDecision, "route", route_config.llm.model_name, cache)
        judge_llm = CachedStructuredLLM(base_judge_llm, JudgeDecision, "judge", route_config.llm.model_name, cache)
        graph = build_agentic_graph(route_llm, judge_llm, faq_store, policy_index, k=sweep.K)

        c1_runs = sweep.run_arm_c(graph, route_llm, judge_llm, queries, route_enabled=False)
        _check_call_budget(route_llm, judge_llm)
        c2_runs = sweep.run_arm_c(graph, route_llm, judge_llm, queries, route_enabled=True)
        _check_call_budget(route_llm, judge_llm)

        save_cache(cache)

    results = {
        "cell_id": sweep.CELL_ID,
        "k": sweep.K,
        "arm_a": arm_a.to_dict(),
        "arm_b": {"cutoff": cutoff, **arm_b.to_dict()},
        "arm_c1_runs": [_run_summary(run) for run in c1_runs],
        "arm_c2_runs": [
            {**_run_summary(run), "routing_accuracy": sweep.routing_accuracy(run)}
            for run in c2_runs
        ],
        "real_llm_calls_made": route_llm.call_count + judge_llm.call_count,
    }

    # Ordinary (non-should-abstain) fail-opens: already excluded from the
    # scored matrix by _run_summary, printed here only for visibility.
    for label, runs in (("C1", c1_runs), ("C2", c2_runs)):
        for run_index, run in enumerate(runs):
            other = sweep.fail_open_ids(run) - sweep.SHOULD_ABSTAIN_IDS
            if other:
                print(f"WARNING: {label} run {run_index}: fail-open on {sorted(other)}, excluded from the scored matrix")

    # eval/METHODOLOGY.md #15a's acceptance bar, enforced at the process
    # exit code and in the results file itself, not merely logged: a run
    # that fails this check must not be mistakable for a clean one by
    # anything reading the exit code or the JSON downstream.
    invalid_reasons = _acceptance_check(c1_runs, c2_runs, results["real_llm_calls_made"])
    for reason in invalid_reasons:
        print(reason)

    results["run_invalid"] = bool(invalid_reasons)
    results["run_invalid_reasons"] = invalid_reasons

    out_path = REPO_ROOT / "results" / "agentic_eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Real (non-cached) LLM calls made this run: {results['real_llm_calls_made']}")

    if invalid_reasons:
        print(
            f"\nRUN INVALID ({len(invalid_reasons)} reason(s)): results/agentic_eval_results.json is "
            f"flagged run_invalid=true and must not be treated as a valid deliverable."
        )
        sys.exit(1)


def _acceptance_check(c1_runs: list, c2_runs: list, real_calls_made: int) -> list[str]:
    """eval/METHODOLOGY.md #15a's acceptance bar as a pure decision
    function: given the raw per-query-run data and the real call count,
    return the list of reasons the run is invalid (empty = clean).
    Kept separate from main()'s side effects (printing, writing the
    results file, exiting) so it's directly testable with synthetic
    per-query-run data, no embeddings/graph/API required.
    """
    reasons = []
    if real_calls_made == 0:
        reasons.append("zero real (non-cached) LLM calls were made -- nothing was actually evaluated")

    for label, runs in (("C1", c1_runs), ("C2", c2_runs)):
        for run_index, run in enumerate(runs):
            on_should_abstain = sweep.fail_open_ids(run) & sweep.SHOULD_ABSTAIN_IDS
            if on_should_abstain:
                reasons.append(
                    f"RE-RUN REQUIRED (METHODOLOGY.md #15a): {label} run {run_index}: fail-open on "
                    f"should-abstain quer{'y' if len(on_should_abstain) == 1 else 'ies'} "
                    f"{sorted(on_should_abstain)} -- effective n dropped below 12 for this run."
                )
    return reasons


def _run_summary(run: list[dict]) -> dict:
    """Fail-open calls (src/adapters/agentic.py's fail-open behavior on a
    route/judge call failure) are recorded as their own explicit
    category and excluded from the scored decision matrix, never folded
    silently into "the judge chose to answer" -- pre-run review's
    blocking fix. `all_including_fail_opens` is kept only for audit/
    transparency; the decision rule (METHODOLOGY.md #16) reads `scored`.
    """
    fail_opens = sweep.fail_open_ids(run)
    return {
        "fail_open_ids": sorted(fail_opens),
        "fail_open_count": len(fail_opens),
        "all_including_fail_opens": sweep.matrices_for_run(run).to_dict(),
        "scored": sweep.matrices_for_run(run, exclude_ids=fail_opens).to_dict(),
        "scored_excluding_retrievable_but_incomplete_hits": sweep.matrices_for_run(
            run, exclude_ids=fail_opens | sweep.RETRIEVABLE_BUT_INCOMPLETE_HITS
        ).to_dict(),
    }


def _check_call_budget(route_llm, judge_llm) -> None:
    total = route_llm.call_count + judge_llm.call_count
    if total > sweep.HARD_STOP_CALLS:
        raise RuntimeError(
            f"{total} real LLM calls made, past the hard stop of {sweep.HARD_STOP_CALLS} "
            f"(1.5x the approved ~{sweep.APPROVED_CALL_ESTIMATE}-call estimate). Stopping to "
            f"avoid uncontrolled spend -- investigate before re-running."
        )


if __name__ == "__main__":
    main()
