"""Phase 6 results follow-up (docs/phase6_results.md): the single
comparison run in `results/multiagent_critic_eval_results.json` found the
multi-agent graph 15.3% faster, but that run had the old graph go first
in the process -- confounded with ordinary warm-up/connection-reuse
effects, not just a structural claim about the graphs. This script runs
4 paired comparisons, alternating which graph goes first each time
(old/new/old/new), to check whether "faster" tracks graph *identity*
(a real effect) or run *position* (an order artifact) -- not a new
benchmark harness, reuses eval.run_multiagent_critic_eval's
run_comparison_once unchanged, just calls it 4x.

    OPENAI_API_KEY=... python -m eval.run_paired_timing_runs

Real, paid calls: ~4x a single comparison's volume. Approved estimate
2448 (612 x 4, eval.run_multiagent_critic_eval.APPROVED_CALL_ESTIMATE),
hard stop 3672 (1.5x), checked across all 4 runs combined.
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

from eval import multiagent_critic_sweep as masweep  # noqa: E402
from eval.query_set import load_query_set  # noqa: E402
from eval.run_multiagent_critic_eval import APPROVED_CALL_ESTIMATE, run_comparison_once  # noqa: E402

ORDERS = ["old_first", "new_first", "old_first", "new_first"]
PAIRED_APPROVED_ESTIMATE = APPROVED_CALL_ESTIMATE * len(ORDERS)
PAIRED_HARD_STOP = int(PAIRED_APPROVED_ESTIMATE * 1.5)


def _aggregate(runs: list[dict]) -> dict:
    deltas_new_minus_old = [r["wall_clock_delta_s"] for r in runs]
    # Position-based: wall time of whichever graph ran first minus whichever ran second,
    # regardless of identity -- isolates an order/warm-up effect from a graph-identity effect.
    position_deltas = []
    for r in runs:
        old_s, new_s = r["old_graph"]["wall_clock_s"], r["new_graph"]["wall_clock_s"]
        first_s, second_s = (old_s, new_s) if r["order"] == "old_first" else (new_s, old_s)
        position_deltas.append(first_s - second_s)

    n = len(runs)
    mean_identity = sum(deltas_new_minus_old) / n
    mean_position = sum(position_deltas) / n
    return {
        "n_runs": n,
        "per_run": [
            {"run_index": i, "order": r["order"], "old_wall_s": r["old_graph"]["wall_clock_s"],
             "new_wall_s": r["new_graph"]["wall_clock_s"], "wall_clock_delta_new_minus_old_s": r["wall_clock_delta_s"]}
            for i, r in enumerate(runs)
        ],
        "identity_effect": {
            "description": "new_graph wall_s - old_graph wall_s, regardless of run order -- "
                            "consistently negative across orders would support a real (new-graph) effect",
            "deltas_s": deltas_new_minus_old,
            "mean_s": mean_identity,
        },
        "position_effect": {
            "description": "first-run wall_s - second-run wall_s, regardless of graph identity -- "
                            "consistently positive would support an order/warm-up artifact instead",
            "deltas_s": position_deltas,
            "mean_s": mean_position,
        },
    }


def main() -> None:
    queries = load_query_set()
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    runs = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        faq_store, policy_index = masweep.build_representative_cell(embeddings, Path(tmp))
        for i, order in enumerate(ORDERS):
            print(f"--- paired run {i} (order={order}) ---")
            result = run_comparison_once(faq_store, policy_index, queries, order=order)
            runs.append(result)
            print(f"  old={result['old_graph']['wall_clock_s']:.1f}s new={result['new_graph']['wall_clock_s']:.1f}s "
                  f"delta(new-old)={result['wall_clock_delta_s']:+.1f}s calls={result['total_real_llm_calls']}")
            total_so_far = sum(r["total_real_llm_calls"] for r in runs)
            if total_so_far > PAIRED_HARD_STOP:
                raise RuntimeError(
                    f"{total_so_far} real LLM calls made across {len(runs)} paired runs, past the hard "
                    f"stop of {PAIRED_HARD_STOP} (1.5x the approved ~{PAIRED_APPROVED_ESTIMATE}-call "
                    f"estimate). Stopping to avoid uncontrolled spend."
                )

    aggregate = _aggregate(runs)
    total_calls = sum(r["total_real_llm_calls"] for r in runs)
    output = {"runs": runs, "aggregate": aggregate, "total_real_llm_calls": total_calls}

    out_path = REPO_ROOT / "results" / "multiagent_critic_paired_timing_runs.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path}")
    print(f"Identity effect (new - old), per run: {[f'{d:+.1f}s' for d in aggregate['identity_effect']['deltas_s']]}, "
          f"mean {aggregate['identity_effect']['mean_s']:+.1f}s")
    print(f"Position effect (first - second), per run: {[f'{d:+.1f}s' for d in aggregate['position_effect']['deltas_s']]}, "
          f"mean {aggregate['position_effect']['mean_s']:+.1f}s")
    print(f"Total real (non-cached) LLM calls across all {len(runs)} paired runs: {total_calls}")


if __name__ == "__main__":
    main()
