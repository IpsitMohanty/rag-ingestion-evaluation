"""Entry point for the phase 5 evaluation (eval/METHODOLOGY.md #19).

    OPENAI_API_KEY=... python -m eval.run_corrective_eval

Requires OPENAI_API_KEY (route/grade_documents/rewrite_query/generate/
grade_generation calls) and the sentence-transformers/all-MiniLM-L6-v2
model available locally (HF_HUB_OFFLINE=1 once cached -- see README). Not
run in CI: this makes real, paid LLM calls, gated on a real API key exactly
like phase 4's harness. Estimated cost: see
eval.corrective_sweep.APPROVED_CALL_ESTIMATE calls (~918, budgeted at an
average of 6 calls/query -- most queries resolve in one pass at 4 calls;
some correct once or twice at up to 12) -- see the printed summary for the
actual count and a hard stop at 1.5x that if something is calling more
than expected.

Compares against phase 4's arm A (eval.agentic_sweep.arm_a_matrix, plain
retrieval, never abstains) on the same representative cell and 51-query
set -- not re-run here, reused from eval.agentic_sweep unchanged.
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from adapters.corrective_rag import (  # noqa: E402
    FaithfulnessGrade,
    GeneratedAnswer,
    JudgeDecision,
    RewrittenQuery,
    RouteDecision,
    build_corrective_graph,
    get_corrective_structured_llms,
)
from config import DEFAULT_CONFIG  # noqa: E402

from eval import agentic_sweep as sweep  # noqa: E402
from eval import corrective_sweep as csweep  # noqa: E402
from eval.llm_cache import CachedStructuredLLM, load_cache, save_cache  # noqa: E402
from eval.query_set import load_query_set  # noqa: E402
from eval.retrievers import query_combined  # noqa: E402


def main() -> None:
    queries = load_query_set()
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        faq_store, policy_index = csweep.build_representative_cell(embeddings, Path(tmp))

        # arm A, reused unchanged from phase 4 -- same cell, same queries.
        baseline_hits = {q["id"]: query_combined(q["text"], csweep.K, faq_store, policy_index) for q in queries}
        sweep._verify_frozen_ground_truth(queries, baseline_hits)
        ground_truth = {q["id"]: (q["id"] in csweep.SHOULD_ABSTAIN_IDS) for q in queries}
        arm_a = sweep.arm_a_matrix(ground_truth)

        config = DEFAULT_CONFIG.corrective_agentic
        base_route, base_judge, base_rewrite, base_generate, base_faithfulness = get_corrective_structured_llms(config)
        cache = load_cache()
        model_name = config.llm.model_name
        route_llm = CachedStructuredLLM(base_route, RouteDecision, "route", model_name, cache)
        judge_llm = CachedStructuredLLM(base_judge, JudgeDecision, "grade_documents", model_name, cache)
        rewrite_llm = CachedStructuredLLM(base_rewrite, RewrittenQuery, "rewrite_query", model_name, cache)
        generate_llm = CachedStructuredLLM(base_generate, GeneratedAnswer, "generate", model_name, cache)
        faithfulness_llm = CachedStructuredLLM(base_faithfulness, FaithfulnessGrade, "grade_generation", model_name, cache)
        llms = [route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm]

        graph = build_corrective_graph(
            route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm,
            faq_store, policy_index, k=csweep.K,
        )

        # Drive run_index ourselves, one run at a time, so the real
        # (non-cached) call count can be attributed per run, not just as
        # one cumulative total -- csweep.run_single_pass runs exactly one
        # pass at whatever run_index the wrapped LLMs are currently set to.
        runs = []
        calls_per_run = []
        for run_index in range(csweep.N_RUNS):
            for llm in llms:
                llm.run_index = run_index
            before = sum(llm.call_count for llm in llms)
            runs.append(csweep.run_single_pass(graph, queries, route_enabled=True, max_iterations=config.max_iterations))
            calls_per_run.append(sum(llm.call_count for llm in llms) - before)
        _check_call_budget(llms)

        save_cache(cache)

    results = {
        "cell_id": csweep.CELL_ID,
        "k": csweep.K,
        "max_iterations": config.max_iterations,
        "arm_a": arm_a.to_dict(),
        "corrective_runs": [_run_summary(run) for run in runs],
        "real_llm_calls_made": sum(llm.call_count for llm in llms),
        "real_llm_calls_per_run": calls_per_run,
        "estimated_cost_usd": _estimate_cost_usd(calls_per_run, model_name),
    }

    for run_index, run in enumerate(runs):
        other = csweep.fail_open_ids(run) - csweep.SHOULD_ABSTAIN_IDS
        if other:
            print(f"WARNING: run {run_index}: fail-open on {sorted(other)}, excluded from the scored matrix")

    invalid_reasons = _acceptance_check(runs, results["real_llm_calls_made"])
    for reason in invalid_reasons:
        print(reason)

    results["run_invalid"] = bool(invalid_reasons)
    results["run_invalid_reasons"] = invalid_reasons

    out_path = REPO_ROOT / "results" / "corrective_eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Real (non-cached) LLM calls made this run: {results['real_llm_calls_made']}")

    if invalid_reasons:
        print(
            f"\nRUN INVALID ({len(invalid_reasons)} reason(s)): results/corrective_eval_results.json is "
            f"flagged run_invalid=true and must not be treated as a valid deliverable."
        )
        sys.exit(1)


def _acceptance_check(runs: list, real_calls_made: int) -> list[str]:
    """Mirrors eval/run_agentic_eval.py's _acceptance_check (eval/METHODOLOGY.md
    #15a's acceptance bar), extended to this graph's five call sites."""
    reasons = []
    if real_calls_made == 0:
        reasons.append("zero real (non-cached) LLM calls were made -- nothing was actually evaluated")

    for run_index, run in enumerate(runs):
        on_should_abstain = csweep.fail_open_ids(run) & csweep.SHOULD_ABSTAIN_IDS
        if on_should_abstain:
            reasons.append(
                f"RE-RUN REQUIRED (METHODOLOGY.md #15a): run {run_index}: fail-open on "
                f"should-abstain quer{'y' if len(on_should_abstain) == 1 else 'ies'} "
                f"{sorted(on_should_abstain)} -- effective n dropped below 12 for this run."
            )
    return reasons


def _run_summary(run: list[dict]) -> dict:
    fail_opens = csweep.fail_open_ids(run)
    return {
        "fail_open_ids": sorted(fail_opens),
        "fail_open_count": len(fail_opens),
        "all_including_fail_opens": csweep.matrices_for_run(run).to_dict(),
        "scored": csweep.matrices_for_run(run, exclude_ids=fail_opens).to_dict(),
        "scored_excluding_retrievable_but_incomplete_hits": csweep.matrices_for_run(
            run, exclude_ids=fail_opens | sweep.RETRIEVABLE_BUT_INCOMPLETE_HITS
        ).to_dict(),
        "rescue_rate": csweep.rescue_rate_for_run(run, exclude_ids=fail_opens),
        "faithfulness_rate": csweep.faithfulness_rate_for_run(run, exclude_ids=fail_opens),
        "mean_loops": csweep.mean_loops_for_run(run, exclude_ids=fail_opens),
        "corrective_fire_rate": csweep.corrective_fire_rate_for_run(run, exclude_ids=fail_opens),
        # Per-query detail + trace, so a reader can audit *why* a specific
        # query looped or abstained, not just the aggregate matrix -- the
        # observability requirement this phase's brief called for.
        "per_query": run,
    }


def _estimate_cost_usd(calls_per_run: list[int], model_name: str) -> dict:
    """Coarse, clearly-approximate cost estimate -- NOT reconciled against
    actual OpenAI usage/billing telemetry (CachedStructuredLLM discards
    token-usage metadata when caching a call's parsed Pydantic result --
    see eval/llm_cache.py). Assumes gpt-4o-mini list pricing ($0.15/1M
    input, $0.60/1M output tokens, as of this writing) and a rough blended
    ~800 input / ~120 output tokens per call (grade_documents/generate/
    grade_generation prompts carry the retrieved excerpts and are much
    longer than route/rewrite_query's short prompts; this is one blended
    average across all five call sites, not a per-node breakdown). Order-
    of-magnitude only -- the actual OpenAI usage dashboard is authoritative,
    this is not.
    """
    ASSUMED_INPUT_TOKENS_PER_CALL = 800
    ASSUMED_OUTPUT_TOKENS_PER_CALL = 120
    INPUT_PRICE_PER_1M = 0.15
    OUTPUT_PRICE_PER_1M = 0.60

    total_calls = sum(calls_per_run)
    per_call_cost = (
        ASSUMED_INPUT_TOKENS_PER_CALL / 1_000_000 * INPUT_PRICE_PER_1M
        + ASSUMED_OUTPUT_TOKENS_PER_CALL / 1_000_000 * OUTPUT_PRICE_PER_1M
    )
    return {
        "model": model_name,
        "total_real_calls": total_calls,
        "note": "rough order-of-magnitude estimate only, not reconciled against actual OpenAI billing/usage telemetry",
        "assumed_input_tokens_per_call": ASSUMED_INPUT_TOKENS_PER_CALL,
        "assumed_output_tokens_per_call": ASSUMED_OUTPUT_TOKENS_PER_CALL,
        "estimated_usd": round(total_calls * per_call_cost, 4),
    }


def _check_call_budget(llms: list) -> None:
    total = sum(llm.call_count for llm in llms)
    if total > csweep.HARD_STOP_CALLS:
        raise RuntimeError(
            f"{total} real LLM calls made, past the hard stop of {csweep.HARD_STOP_CALLS} "
            f"(1.5x the approved ~{csweep.APPROVED_CALL_ESTIMATE}-call estimate). Stopping to "
            f"avoid uncontrolled spend -- investigate before re-running."
        )


if __name__ == "__main__":
    main()
