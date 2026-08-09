"""Entry point for phase 6 Stage 4 (docs/phase6_preregistration.md #6a).

    OPENAI_API_KEY=... python -m eval.run_multiagent_critic_eval

Runs phase 5's own monolithic graph (`corrective_rag.build_corrective_graph`,
imported unmodified) and this phase's multi-agent graph
(`multiagent_critic_rag.build_multiagent_critic_graph`, `critic="baseline"`)
back-to-back, fresh/uncached (a cache hit would not reflect real latency --
#6a), over the same 51-query set and representative cell, timing each with
wall-clock `time.perf_counter()`. Not run in CI: real, paid LLM calls,
gated on a real API key, same as every other real-run harness in this
repo.

Approved estimate: 612 calls (51 queries x 2 graphs x ~6 calls/query,
#6a), hard stop 918 (1.5x). One pass per graph, not 3 -- #6a: this is a
structural/timing comparison, not a re-litigation of the API-variance
question phase 5 already settled.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from adapters.corrective_rag import build_corrective_graph, get_corrective_structured_llms  # noqa: E402
from adapters.multiagent_critic_rag import (  # noqa: E402
    build_multiagent_critic_graph,
    get_multiagent_structured_llms,
)
from config import DEFAULT_CONFIG  # noqa: E402

from eval import corrective_sweep as csweep  # noqa: E402
from eval import multiagent_critic_sweep as masweep  # noqa: E402
from eval.query_set import load_query_set  # noqa: E402

APPROVED_CALL_ESTIMATE = 612  # 51 queries x 2 graphs x ~6 calls/query (#6a)
HARD_STOP_CALLS = int(APPROVED_CALL_ESTIMATE * 1.5)


class TimingLLM:
    """Deliberately NOT cached (unlike eval.llm_cache.CachedStructuredLLM):
    #6a's latency comparison needs real, fresh calls -- a cache hit
    returns near-instantly and would not reflect real latency. Tracks
    call_count and a per-call wall-clock latency list instead of a disk
    cache.
    """

    def __init__(self, runnable):
        self._runnable = runnable
        self.call_count = 0
        self.call_latencies_s: list[float] = []

    def invoke(self, prompt: str):
        start = time.perf_counter()
        result = self._runnable.invoke(prompt)
        self.call_latencies_s.append(time.perf_counter() - start)
        self.call_count += 1
        return result


def _timed_llms(raw_llms) -> list[TimingLLM]:
    return [TimingLLM(llm) for llm in raw_llms]


def _run_and_time(graph, run_fn, queries, route_enabled=True, max_iterations=3) -> tuple[list[dict], float]:
    start = time.perf_counter()
    per_query = run_fn(graph, queries, route_enabled=route_enabled, max_iterations=max_iterations)
    wall_s = time.perf_counter() - start
    return per_query, wall_s


def _summary_for(label: str, per_query: list[dict], llms: list[TimingLLM], wall_s: float) -> dict:
    fail_opens = masweep.fail_open_ids(per_query)
    all_latencies = [lat for llm in llms for lat in llm.call_latencies_s]
    return {
        "label": label,
        "wall_clock_s": wall_s,
        "real_llm_calls": sum(llm.call_count for llm in llms),
        "mean_call_latency_s": sum(all_latencies) / len(all_latencies) if all_latencies else None,
        "fail_open_ids": sorted(fail_opens),
        "fail_open_count": len(fail_opens),
        "all_including_fail_opens": masweep.matrices_for_run(per_query).to_dict(),
        "scored": masweep.matrices_for_run(per_query, exclude_ids=fail_opens).to_dict(),
        "rescue_rate": masweep.rescue_rate_for_run(per_query, exclude_ids=fail_opens),
        "faithfulness_rate": masweep.faithfulness_rate_for_run(per_query, exclude_ids=fail_opens),
        "mean_loops": masweep.mean_loops_for_run(per_query, exclude_ids=fail_opens),
        "corrective_fire_rate": masweep.corrective_fire_rate_for_run(per_query, exclude_ids=fail_opens),
        "per_query": per_query,
    }


def _decision_agreement(old_run: list[dict], new_run: list[dict]) -> dict:
    """#6a prediction 1: per-query decisions should match up to LLM
    API-level nondeterminism, not necessarily byte-identical."""
    by_id_old = {pq["id"]: pq for pq in old_run}
    by_id_new = {pq["id"]: pq for pq in new_run}
    mismatched_abstain, mismatched_grounded = [], []
    for qid in by_id_old:
        old_pq, new_pq = by_id_old[qid], by_id_new[qid]
        if old_pq["abstained"] != new_pq["abstained"]:
            mismatched_abstain.append(qid)
        if old_pq["grounded"] != new_pq["grounded"]:
            mismatched_grounded.append(qid)
    n = len(by_id_old)
    return {
        "n_queries": n,
        "abstain_decision_mismatches": sorted(mismatched_abstain),
        "abstain_agreement_rate": (n - len(mismatched_abstain)) / n,
        "grounded_decision_mismatches": sorted(mismatched_grounded),
        "grounded_agreement_rate": (n - len(mismatched_grounded)) / n,
    }


def run_comparison_once(faq_store, policy_index, queries: list[dict], order: str = "old_first") -> dict:
    """Builds fresh (uncached) instances of both graphs against the given,
    already-built cell and runs each once over `queries`, in the order
    requested -- `order` controls only which graph experiences "ran first
    in this process" effects (connection warm-up, etc.), not which graph
    the returned dict calls "old"/"new". Factored out of main() so
    eval/run_paired_timing_runs.py can call this 4x with alternating
    order without rebuilding the retrieval indices each time (results
    doc's paired-timing follow-up, docs/phase6_results.md).
    """
    if order not in ("old_first", "new_first"):
        raise ValueError(f"order must be 'old_first' or 'new_first', got {order!r}")

    old_config = DEFAULT_CONFIG.corrective_agentic
    new_config = DEFAULT_CONFIG.multiagent_critic

    def _run_old():
        old_llms = _timed_llms(get_corrective_structured_llms(old_config))
        old_graph = build_corrective_graph(*old_llms, faq_store, policy_index, k=masweep.K)
        old_run, old_wall_s = _run_and_time(
            old_graph, csweep.run_single_pass, queries, route_enabled=True, max_iterations=old_config.max_iterations,
        )
        _check_call_budget(old_llms, "old (phase 5)")
        return old_run, old_wall_s, old_llms

    def _run_new():
        new_llms = _timed_llms(get_multiagent_structured_llms(new_config))
        new_graph = build_multiagent_critic_graph(
            *new_llms, faq_store, policy_index, k=masweep.K, critic=new_config.critic,
        )
        new_run, new_wall_s = _run_and_time(
            new_graph, masweep.run_single_pass, queries, route_enabled=True, max_iterations=new_config.max_iterations,
        )
        _check_call_budget(new_llms, "new (phase 6)")
        return new_run, new_wall_s, new_llms

    if order == "old_first":
        old_run, old_wall_s, old_llms = _run_old()
        new_run, new_wall_s, new_llms = _run_new()
    else:
        new_run, new_wall_s, new_llms = _run_new()
        old_run, old_wall_s, old_llms = _run_old()

    old_summary = _summary_for("phase5_monolith", old_run, old_llms, old_wall_s)
    new_summary = _summary_for("phase6_multiagent_baseline", new_run, new_llms, new_wall_s)
    agreement = _decision_agreement(old_run, new_run)
    total_calls = old_summary["real_llm_calls"] + new_summary["real_llm_calls"]

    return {
        "order": order,
        "cell_id": masweep.CELL_ID,
        "k": masweep.K,
        "old_graph": old_summary,
        "new_graph": new_summary,
        "decision_agreement": agreement,
        "call_count_delta": new_summary["real_llm_calls"] - old_summary["real_llm_calls"],
        "wall_clock_delta_s": new_summary["wall_clock_s"] - old_summary["wall_clock_s"],
        "total_real_llm_calls": total_calls,
        "approved_call_estimate": APPROVED_CALL_ESTIMATE,
        "hard_stop_calls": HARD_STOP_CALLS,
    }


def main() -> None:
    queries = load_query_set()
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        faq_store, policy_index = masweep.build_representative_cell(embeddings, Path(tmp))
        results = run_comparison_once(faq_store, policy_index, queries, order="old_first")

    out_path = REPO_ROOT / "results" / "multiagent_critic_eval_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    old_summary, new_summary, agreement = results["old_graph"], results["new_graph"], results["decision_agreement"]
    print(f"Wrote {out_path}")
    print(f"Old graph (phase 5 monolith): {old_summary['real_llm_calls']} calls, {old_summary['wall_clock_s']:.1f}s wall-clock")
    print(f"New graph (phase 6 multi-agent, baseline critic): {new_summary['real_llm_calls']} calls, {new_summary['wall_clock_s']:.1f}s wall-clock")
    print(f"Call-count delta (new - old): {results['call_count_delta']:+d}")
    print(f"Wall-clock delta (new - old): {results['wall_clock_delta_s']:+.1f}s")
    print(f"Abstain-decision agreement: {agreement['abstain_agreement_rate']:.3f} "
          f"({len(agreement['abstain_decision_mismatches'])} mismatches: {agreement['abstain_decision_mismatches']})")
    print(f"Total real (non-cached) LLM calls this run: {results['total_real_llm_calls']}")


def _check_call_budget(llms: list[TimingLLM], label: str) -> None:
    total = sum(llm.call_count for llm in llms)
    if total > HARD_STOP_CALLS:
        raise RuntimeError(
            f"{total} real LLM calls made ({label}), past the hard stop of {HARD_STOP_CALLS} "
            f"(1.5x the approved ~{APPROVED_CALL_ESTIMATE}-call estimate). Stopping to avoid "
            f"uncontrolled spend -- investigate before re-running."
        )


if __name__ == "__main__":
    main()
