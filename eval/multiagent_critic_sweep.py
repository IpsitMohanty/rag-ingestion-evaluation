"""Phase 6 Stage 4 driver (docs/phase6_preregistration.md #6/#6a): runs
`src/adapters/multiagent_critic_rag.py`'s graph over the same 51-query
set and representative cell phase 4/5 use, producing per-query records in
the *exact same dict shape* `eval/corrective_sweep.py::run_single_pass`
does -- so this module reuses that module's scoring functions
(`matrices_for_run`, `rescue_rate_for_run`, `faithfulness_rate_for_run`,
`mean_loops_for_run`, `corrective_fire_rate_for_run`, `fail_open_ids`)
directly, unchanged, rather than reimplementing them. Only `run_single_pass`
below is genuinely new -- everything else needed to score its output
already exists.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from . import agentic_sweep as sweep  # noqa: E402
from . import corrective_sweep as csweep  # noqa: E402

CELL_ID = csweep.CELL_ID
K = csweep.K
SHOULD_ABSTAIN_IDS = csweep.SHOULD_ABSTAIN_IDS
RESCUE_TARGET_IDS = csweep.RESCUE_TARGET_IDS
build_representative_cell = csweep.build_representative_cell

# Reused unchanged (docs/phase6_preregistration.md #6a: only run_single_pass
# below is new -- these depend only on the per-query dict shape, not on
# which graph produced it).
fail_open_ids = csweep.fail_open_ids
matrices_for_run = csweep.matrices_for_run
rescue_rate_for_run = csweep.rescue_rate_for_run
faithfulness_rate_for_run = csweep.faithfulness_rate_for_run
mean_loops_for_run = csweep.mean_loops_for_run
corrective_fire_rate_for_run = csweep.corrective_fire_rate_for_run


def run_single_pass(graph, queries: list[dict], route_enabled: bool, max_iterations: int = 3) -> list[dict]:
    """Mirrors eval.corrective_sweep.run_single_pass exactly, field for
    field, using adapters.multiagent_critic_rag.run_multiagent_critic_query
    instead of adapters.corrective_rag.run_corrective_query -- the two
    result dataclasses carry the same fields (docs/phase6_preregistration.md
    #6a's quality prediction depends on this: comparing the two runs must
    compare like fields, not fields that merely happen to have the same name).
    """
    from adapters.multiagent_critic_rag import run_multiagent_critic_query

    per_query = []
    for q in queries:
        result = run_multiagent_critic_query(
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
            "trace": result.trace,
        })
    return per_query
