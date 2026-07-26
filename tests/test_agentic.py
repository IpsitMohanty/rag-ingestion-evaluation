"""Coverage: phase 4's LangGraph adapter (src/adapters/agentic.py) and
the pure scoring/statistics functions in eval/agentic_sweep.py and
eval/llm_cache.py.

Entirely mocked-LLM: no OPENAI_API_KEY, no network, no real API spend.
The one test that calls the real OpenAI API is gated behind
`requires_llm` (conftest.py), same pattern as test_onnx_parity.py's
`requires_network` -- skipped in CI on purpose.
"""
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from adapters import retriever as retriever_adapter
from adapters.agentic import (
    AgenticResult,
    JudgeDecision,
    RouteDecision,
    build_agentic_graph,
    get_structured_llms,
    run_agentic_query,
)
from adapters import vectorstore as vectorstore_adapter
from config import AgenticConfig, RetrieverConfig, VectorStoreConfig
from conftest import requires_llm

from eval import agentic_sweep as sweep
from eval.llm_cache import CachedStructuredLLM, load_cache, save_cache
from eval.query_set import load_query_set
from eval.retrievers import build_faq_index, build_policy_index
from eval.run_agentic_eval import _acceptance_check

FAQ_DOCS = [
    Document(page_content="Question: How do I register? Answer: Open the app.", metadata={"faq_index": 0}),
]
POLICY_CHUNKS = [
    Document(page_content="Anganwadi Workers must have a minimum qualification.", metadata={"page": 5}),
]


class _FixedRouteLLM:
    def __init__(self, source: str = "both"):
        self._source = source

    def invoke(self, prompt: str) -> RouteDecision:
        return RouteDecision(source=self._source, reasoning="fixed for test")


class _FixedJudgeLLM:
    def __init__(self, answerable: bool = True):
        self._answerable = answerable

    def invoke(self, prompt: str) -> JudgeDecision:
        return JudgeDecision(answerable=self._answerable, reasoning="fixed for test")


class _RaisingLLM:
    """Stands in for a malformed/unparseable structured-output response
    (what with_structured_output raises on) and equally for a missing/
    invalid API key (what a real ChatOpenAI client raises on .invoke()
    with no credentials) -- both surface identically to the graph as an
    exception from .invoke(), so one fake covers both scenarios.
    """

    def __init__(self, exc: Exception):
        self._exc = exc

    def invoke(self, prompt: str):
        raise self._exc


def _real_faq_and_policy(tmp_path, fake_embeddings, faq_docs=FAQ_DOCS, policy_chunks=POLICY_CHUNKS):
    faq_store = build_faq_index(faq_docs, fake_embeddings, tmp_path / "faq")
    retriever_config = RetrieverConfig(strategy="similarity", k=5)
    policy_index = build_policy_index(
        "similarity", policy_chunks, policy_chunks, fake_embeddings, tmp_path / "policy", retriever_config,
    )
    return faq_store, policy_index


# --- graph routing ---------------------------------------------------

def test_route_disabled_never_calls_the_route_llm(tmp_path, fake_embeddings):
    """C1 (eval/METHODOLOGY.md #9): routing disabled means no LLM call at
    all for routing, not just a call that gets ignored."""
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    poison_route_llm = _RaisingLLM(AssertionError("route LLM must not be called when route_enabled=False"))
    graph = build_agentic_graph(poison_route_llm, _FixedJudgeLLM(), faq_store, policy_index, k=5)

    result = run_agentic_query(graph, "any query", route_enabled=False)

    assert result.route == "both"
    assert result.route_call_failed is False


def test_route_enabled_gates_retrieval_to_the_chosen_source(tmp_path, fake_embeddings):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    graph = build_agentic_graph(_FixedRouteLLM(source="faq"), _FixedJudgeLLM(), faq_store, policy_index, k=5)

    result = run_agentic_query(graph, "How do I register?", route_enabled=True)

    assert result.route == "faq"
    assert all(h.source_type == "faq" for h in result.hits)


def test_route_both_merges_faq_and_policy_hits(tmp_path, fake_embeddings):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    graph = build_agentic_graph(_FixedRouteLLM(source="both"), _FixedJudgeLLM(), faq_store, policy_index, k=5)

    result = run_agentic_query(graph, "some query", route_enabled=True)

    source_types = {h.source_type for h in result.hits}
    assert source_types == {"faq", "policy_pdf"}


# --- fail-open behavior: malformed output, missing/invalid credentials ---

def test_judge_call_failure_does_not_crash_and_defaults_to_answerable(tmp_path, fake_embeddings):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    graph = build_agentic_graph(
        _FixedRouteLLM(), _RaisingLLM(ValueError("malformed structured output")), faq_store, policy_index, k=5,
    )

    result = run_agentic_query(graph, "any query", route_enabled=True)

    assert result.judgment == "answerable"
    assert result.abstained is False
    assert result.judge_call_failed is True


def test_route_call_failure_does_not_crash_and_defaults_to_both(tmp_path, fake_embeddings):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    graph = build_agentic_graph(
        _RaisingLLM(ValueError("malformed structured output")), _FixedJudgeLLM(), faq_store, policy_index, k=5,
    )

    result = run_agentic_query(graph, "any query", route_enabled=True)

    assert result.route == "both"
    assert result.route_call_failed is True
    # retrieval still ran against both sources despite the route failure
    assert {h.source_type for h in result.hits} == {"faq", "policy_pdf"}


def test_missing_or_invalid_credentials_degrades_to_plain_retrieval(tmp_path, fake_embeddings):
    """Simulates what a real ChatOpenAI client raises when no/invalid API
    key is configured: the same .invoke()-raises path as malformed
    output, so the graph degrades to plain retrieval (never abstains due
    to an infra problem) rather than crashing or silently abstaining."""
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    auth_error = RuntimeError("Incorrect API key provided")
    graph = build_agentic_graph(_RaisingLLM(auth_error), _RaisingLLM(auth_error), faq_store, policy_index, k=5)

    result = run_agentic_query(graph, "any query", route_enabled=True)

    assert result.abstained is False
    assert result.route_call_failed is True
    assert result.judge_call_failed is True
    assert result.message is None


def test_empty_hits_do_not_crash_the_judge(tmp_path, fake_embeddings):
    empty_faq_store, empty_policy_index = _real_faq_and_policy(tmp_path, fake_embeddings, faq_docs=[], policy_chunks=[])
    graph = build_agentic_graph(_FixedRouteLLM(), _FixedJudgeLLM(answerable=False), empty_faq_store, empty_policy_index, k=5)

    result = run_agentic_query(graph, "anything", route_enabled=True)

    assert result.hits == []
    assert result.abstained is True
    assert result.message


# --- config ------------------------------------------------------------

def test_agentic_config_defaults_to_openai_gpt4o_mini():
    config = AgenticConfig()
    assert config.llm.backend == "openai"
    assert config.llm.model_name == "gpt-4o-mini"


def test_agentic_config_pins_temperature_zero_and_a_fixed_seed():
    """eval/METHODOLOGY.md #15's commitment, made explicit and reviewable
    in config.py rather than buried in the adapter (pre-run review)."""
    config = AgenticConfig()
    assert config.llm.temperature == 0.0
    assert config.llm.seed == 42


def test_llm_config_temperature_and_seed_default_to_none_for_other_callers():
    """Other, non-phase-4 callers of the shared LLMConfig seam are
    unaffected unless they opt in explicitly."""
    from config import LLMConfig

    plain = LLMConfig(backend="openai", model_name="gpt-4o-mini")
    assert plain.temperature is None
    assert plain.seed is None


# --- llm_cache -----------------------------------------------------------

class _CountingRunnable:
    def __init__(self, decision):
        self._decision = decision
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return self._decision


def test_cached_structured_llm_skips_the_real_call_on_a_cache_hit():
    underlying = _CountingRunnable(JudgeDecision(answerable=True, reasoning="r"))
    cache = {}
    wrapped = CachedStructuredLLM(underlying, JudgeDecision, "judge", "gpt-4o-mini", cache)

    first = wrapped.invoke("same prompt")
    second = wrapped.invoke("same prompt")

    assert underlying.calls == 1
    assert wrapped.call_count == 1
    assert first == second


def test_cached_structured_llm_distinguishes_different_prompts():
    underlying = _CountingRunnable(JudgeDecision(answerable=True, reasoning="r"))
    cache = {}
    wrapped = CachedStructuredLLM(underlying, JudgeDecision, "judge", "gpt-4o-mini", cache)

    wrapped.invoke("prompt one")
    wrapped.invoke("prompt two")

    assert underlying.calls == 2


def test_cache_key_includes_run_index_so_repeat_runs_are_not_cache_hits():
    """The pre-run review's blocking fix: without run_index in the key,
    retrieval determinism + fixed queries would make every run after the
    first a guaranteed cache hit on run 1, collapsing the 3-run range
    (METHODOLOGY.md #15) to zero by construction, not because the API is
    actually stable."""
    underlying = _CountingRunnable(JudgeDecision(answerable=True, reasoning="r"))
    cache = {}
    wrapped = CachedStructuredLLM(underlying, JudgeDecision, "judge", "gpt-4o-mini", cache)

    wrapped.run_index = 0
    wrapped.invoke("identical prompt")
    wrapped.run_index = 1
    wrapped.invoke("identical prompt")
    wrapped.run_index = 2
    wrapped.invoke("identical prompt")

    assert underlying.calls == 3  # one real call per run, not one total


def test_cache_key_still_hits_across_separate_harness_executions_for_the_same_run_index():
    """The other half of the same requirement: caching still works
    *within* a fixed run_index, across separate script invocations (e.g.
    re-running after a scoring bug fix) -- it's specifically repeat RUNS
    that must not collapse, not all caching."""
    underlying = _CountingRunnable(JudgeDecision(answerable=True, reasoning="r"))
    cache = {}
    first_invocation = CachedStructuredLLM(underlying, JudgeDecision, "judge", "gpt-4o-mini", cache, run_index=0)
    first_invocation.invoke("identical prompt")

    second_invocation = CachedStructuredLLM(underlying, JudgeDecision, "judge", "gpt-4o-mini", cache, run_index=0)
    second_invocation.invoke("identical prompt")

    assert underlying.calls == 1  # same run_index, same prompt: cache hit, no re-spend


def test_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "cache.json"
    underlying = _CountingRunnable(RouteDecision(source="faq", reasoning="r"))
    cache = load_cache(path)
    assert cache == {}
    wrapped = CachedStructuredLLM(underlying, RouteDecision, "route", "gpt-4o-mini", cache)
    wrapped.invoke("a query")
    save_cache(cache, path)

    reloaded = load_cache(path)
    wrapped2 = CachedStructuredLLM(underlying, RouteDecision, "route", "gpt-4o-mini", reloaded)
    wrapped2.invoke("a query")

    assert underlying.calls == 1  # second invoke was a cache hit from disk


# --- eval/agentic_sweep.py: pure confusion-matrix / statistics functions ---

def test_confusion_matrix_counts_all_four_quadrants():
    ground_truth = {"a": True, "b": True, "c": False, "d": False}
    predictions = {"a": True, "b": False, "c": True, "d": False}
    cm = sweep.confusion_matrix(predictions, ground_truth)
    assert (cm.tp, cm.fn, cm.fp, cm.tn) == (1, 1, 1, 1)


def test_arm_a_never_abstains_and_has_undefined_precision():
    ground_truth = {"a": True, "b": True, "c": False}
    cm = sweep.arm_a_matrix(ground_truth)
    assert cm.tp == 0 and cm.fp == 0
    assert cm.recall == 0.0
    assert cm.precision is None
    assert cm.to_dict()["precision"] == "undefined, no abstain predictions made"


def test_best_threshold_by_youdens_j_beats_never_abstaining_when_separable():
    scores = {"a": 0.9, "b": 0.95, "c": 0.1, "d": 0.2}
    ground_truth = {"a": True, "b": True, "c": False, "d": False}
    cutoff, cm = sweep.best_threshold_by_youdens_j(scores, ground_truth)
    assert cm.tp == 2 and cm.fp == 0  # perfectly separable data recovers a perfect cutoff


def test_best_threshold_does_not_optimize_for_plain_accuracy():
    """A never-abstain rule would score 3/4 = 0.75 accuracy here (only
    one true 'should abstain' case); Youden's J must not settle for that
    when a cutoff exists that catches the minority class too."""
    scores = {"a": 0.9, "b": 0.1, "c": 0.2, "d": 0.3}
    ground_truth = {"a": True, "b": False, "c": False, "d": False}
    cutoff, cm = sweep.best_threshold_by_youdens_j(scores, ground_truth)
    assert cm.tp == 1  # catches the one true "should abstain" case, not just tying accuracy


def test_range_across_runs_is_max_minus_min_not_stddev():
    assert sweep.range_across_runs([0.5, 0.6, 0.7]) == pytest.approx(0.2)
    assert sweep.range_across_runs([0.5]) is None  # need at least 2 values


def test_routing_accuracy_excludes_either_and_neither():
    per_query_run = [
        {"id": "faq-01", "expected_source": "faq", "route": "faq"},
        {"id": "pol-01", "expected_source": "policy_pdf", "route": "faq"},
        {"id": "either-01", "expected_source": "either", "route": "policy_pdf"},
        {"id": "neither-01", "expected_source": "neither", "route": "both"},
    ]
    acc = sweep.routing_accuracy(per_query_run)
    assert acc == 0.5  # 1 correct (faq-01) out of 2 scored (faq-01, pol-01)


def test_fail_open_ids_flags_route_or_judge_failures():
    per_query_run = [
        {"id": "a", "route_call_failed": False, "judge_call_failed": False},
        {"id": "b", "route_call_failed": True, "judge_call_failed": False},
        {"id": "c", "route_call_failed": False, "judge_call_failed": True},
    ]
    assert sweep.fail_open_ids(per_query_run) == frozenset({"b", "c"})


def test_fail_open_queries_are_excluded_from_the_scored_matrix():
    """Pre-run review's blocking fix: a fail-open on a should-abstain
    query must not be silently scored as "the judge chose to answer"."""
    per_query_run = [
        {"id": "a", "abstained": False, "should_abstain": True,
         "route_call_failed": False, "judge_call_failed": True},  # fail-open on a should-abstain case
        {"id": "b", "abstained": True, "should_abstain": True,
         "route_call_failed": False, "judge_call_failed": False},
    ]
    fail_opens = sweep.fail_open_ids(per_query_run)
    all_including = sweep.matrices_for_run(per_query_run)
    scored = sweep.matrices_for_run(per_query_run, exclude_ids=fail_opens)

    assert all_including.fn == 1  # silently counted as a missed abstention if not excluded
    assert scored.fn == 0  # excluded: the failure was infra, not a judge miss
    assert scored.tp == 1  # only query "b", the genuine judge decision, is scored


def test_acceptance_check_flags_fail_open_on_a_should_abstain_id():
    """The scored run that actually happened: exit-code enforcement, not
    just a log line, is the point of this review's fix. A fail-open on
    one of the 12 (here, faq-01, a real member of SHOULD_ABSTAIN_IDS)
    must produce a non-empty reasons list containing the §15a line."""
    assert "faq-01" in sweep.SHOULD_ABSTAIN_IDS
    run = [
        {"id": "faq-01", "route_call_failed": False, "judge_call_failed": True},
        {"id": "either-01", "route_call_failed": False, "judge_call_failed": False},
    ]
    reasons = _acceptance_check(c1_runs=[run], c2_runs=[], real_calls_made=1)
    assert reasons
    assert any("RE-RUN REQUIRED (METHODOLOGY.md #15a)" in r for r in reasons)
    assert any("faq-01" in r for r in reasons)


def test_acceptance_check_flags_zero_real_calls():
    reasons = _acceptance_check(c1_runs=[], c2_runs=[], real_calls_made=0)
    assert reasons
    assert any("zero real" in r for r in reasons)


def test_acceptance_check_clean_run_has_no_reasons():
    run = [
        {"id": "either-01", "route_call_failed": False, "judge_call_failed": False},
        {"id": "faq-02", "route_call_failed": False, "judge_call_failed": True},  # fail-open, but NOT one of the 12
    ]
    reasons = _acceptance_check(c1_runs=[run], c2_runs=[run], real_calls_made=459)
    assert reasons == []


def test_run_arm_c_uses_a_fresh_sample_per_run_not_a_cached_replay(tmp_path, fake_embeddings):
    """Integration-level version of the cache/run_index fix: driving
    run_arm_c through the real graph must actually invoke the underlying
    judge runnable once per run, not once total."""
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    underlying_judge = _CountingRunnable(JudgeDecision(answerable=True, reasoning="r"))
    cache = {}
    route_llm = CachedStructuredLLM(_FixedRouteLLM(), RouteDecision, "route", "gpt-4o-mini", cache)
    judge_llm = CachedStructuredLLM(underlying_judge, JudgeDecision, "judge", "gpt-4o-mini", cache)
    graph = build_agentic_graph(route_llm, judge_llm, faq_store, policy_index, k=5)

    queries = [{"id": "q1", "text": "How do I register?", "expected_source": "faq",
                "ground_truth": [{"source": "faq", "faq_index": 0}]}]
    sweep.run_arm_c(graph, route_llm, judge_llm, queries, route_enabled=False, n_runs=3)

    assert underlying_judge.calls == 3  # one real call per run, not a single cached replay


def test_matrices_for_run_can_exclude_specific_query_ids():
    per_query_run = [
        {"id": "a", "abstained": True, "should_abstain": True},
        {"id": "b", "abstained": False, "should_abstain": False},
    ]
    full = sweep.matrices_for_run(per_query_run)
    excluding_a = sweep.matrices_for_run(per_query_run, exclude_ids=frozenset({"a"}))
    assert full.tp == 1
    assert excluding_a.tp == 0 and excluding_a.tn == 1


def test_should_abstain_ids_are_real_query_ids_in_the_labeled_set():
    """Drift/typo guard on the frozen METHODOLOGY.md #11 constant, not a
    re-derivation of it."""
    all_ids = {q["id"] for q in load_query_set()}
    assert sweep.SHOULD_ABSTAIN_IDS <= all_ids


def test_retrievable_but_incomplete_hits_are_actually_flagged_incomplete():
    queries = {q["id"]: q for q in load_query_set()}
    for qid in sweep.RETRIEVABLE_BUT_INCOMPLETE_HITS:
        assert queries[qid].get("retrievable_but_incomplete") is True
    # pol-10 is retrievable_but_incomplete too, but sits in SHOULD_ABSTAIN_IDS
    # instead (METHODOLOGY.md #12), not in this set.
    assert "pol-10" not in sweep.RETRIEVABLE_BUT_INCOMPLETE_HITS
    assert "pol-10" in sweep.SHOULD_ABSTAIN_IDS


# --- real API, gated, skipped in CI --------------------------------------

@requires_llm
def test_real_openai_structured_output_returns_valid_schema():
    """Mirrors test_onnx_parity.py's role: verifies the real integration
    actually works (structured output against the true API), not just
    the mocked contract every other test in this file exercises. Skipped
    without RUN_LLM_TESTS=1 and OPENAI_API_KEY."""
    config = AgenticConfig()
    route_llm, judge_llm = get_structured_llms(config)

    route_result = route_llm.invoke("Decide: faq, policy_pdf, or both? Question: How do I log in to the app?")
    assert isinstance(route_result, RouteDecision)
    assert route_result.source in ("faq", "policy_pdf", "both")

    judge_result = judge_llm.invoke(
        "Question: What color is the sky?\nExcerpts:\n1. [faq] The app has a blue icon."
    )
    assert isinstance(judge_result, JudgeDecision)
    assert isinstance(judge_result.answerable, bool)
