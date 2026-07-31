"""Coverage: phase 5's corrective/self-reflective LangGraph adapter
(src/adapters/corrective_rag.py) and the pure scoring functions in
eval/corrective_sweep.py and eval/metrics.py's phase-5 additions.

Entirely mocked-LLM: no OPENAI_API_KEY, no network, no real API spend.
The one test that calls the real OpenAI API is gated behind `requires_llm`
(conftest.py), same pattern as tests/test_agentic.py.
"""
from langchain_core.documents import Document

from adapters.corrective_rag import (
    FaithfulnessGrade,
    GeneratedAnswer,
    RewrittenQuery,
    build_corrective_graph,
    get_corrective_structured_llms,
    run_corrective_query,
)
from adapters.agentic import JudgeDecision, RouteDecision
from config import CorrectiveAgenticConfig, RetrieverConfig
from conftest import requires_llm

from eval import corrective_sweep as csweep
from eval import metrics
from eval.llm_cache import CachedStructuredLLM
from eval.retrievers import build_faq_index, build_policy_index
from eval.run_corrective_eval import _acceptance_check

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
    """Stands in for grade_documents's judge_llm. `sequence`, if given, is
    consumed one bool per call (insufficient-then-sufficient, etc.);
    falls back to `answerable` once exhausted or if not given at all."""

    def __init__(self, sequence: list[bool] | None = None, answerable: bool = True):
        self._sequence = list(sequence) if sequence is not None else None
        self._answerable = answerable

    def invoke(self, prompt: str) -> JudgeDecision:
        value = self._sequence.pop(0) if self._sequence else self._answerable
        return JudgeDecision(answerable=value, reasoning="fixed for test")


class _FixedRewriteLLM:
    def __init__(self, query: str = "a rewritten query"):
        self._query = query

    def invoke(self, prompt: str) -> RewrittenQuery:
        return RewrittenQuery(rewritten_query=self._query, reasoning="fixed for test")


class _FixedGenerateLLM:
    def __init__(self, answer: str = "a grounded answer"):
        self._answer = answer

    def invoke(self, prompt: str) -> GeneratedAnswer:
        return GeneratedAnswer(answer=self._answer)


class _FixedFaithfulnessLLM:
    def __init__(self, sequence: list[bool] | None = None, grounded: bool = True):
        self._sequence = list(sequence) if sequence is not None else None
        self._grounded = grounded

    def invoke(self, prompt: str) -> FaithfulnessGrade:
        value = self._sequence.pop(0) if self._sequence else self._grounded
        return FaithfulnessGrade(grounded=value, reasoning="fixed for test")


class _RaisingLLM:
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


def _build_graph(
    tmp_path, fake_embeddings,
    route_llm=None, judge_llm=None, rewrite_llm=None, generate_llm=None, faithfulness_llm=None,
    faq_docs=FAQ_DOCS, policy_chunks=POLICY_CHUNKS,
):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings, faq_docs, policy_chunks)
    return build_corrective_graph(
        route_llm or _FixedRouteLLM(),
        judge_llm or _FixedJudgeLLM(),
        rewrite_llm or _FixedRewriteLLM(),
        generate_llm or _FixedGenerateLLM(),
        faithfulness_llm or _FixedFaithfulnessLLM(),
        faq_store, policy_index, k=5,
    )


# --- happy paths: no correction needed, one correction, two corrections ---

def test_sufficient_documents_and_grounded_generation_skip_the_loop_entirely(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings)

    result = run_corrective_query(graph, "How do I register?")

    assert result.loops == 1
    assert result.corrective_fired is False
    assert result.abstained is False
    assert result.answer == "a grounded answer"
    assert result.grounded is True


def test_insufficient_documents_trigger_one_rewrite_then_generate(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(sequence=[False, True])  # insufficient, then sufficient after rewrite
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_corrective_query(graph, "some vague query")

    assert result.loops == 2
    assert result.corrective_fired is True
    assert result.abstained is False
    assert result.final_query == "a rewritten query"


def test_ungrounded_generation_triggers_one_rewrite_then_succeeds(tmp_path, fake_embeddings):
    faithfulness_llm = _FixedFaithfulnessLLM(sequence=[False, True])
    graph = _build_graph(tmp_path, fake_embeddings, faithfulness_llm=faithfulness_llm)

    result = run_corrective_query(graph, "some query")

    assert result.loops == 2
    assert result.corrective_fired is True
    assert result.abstained is False
    assert result.grounded is True


# --- budget guard: bounded cycles, graceful abstention, never a hallucination ---

def test_persistently_insufficient_documents_abstain_at_the_budget_and_never_generate(tmp_path, fake_embeddings):
    poison_generate = _RaisingLLM(AssertionError("generate must not be called if documents never became sufficient"))
    judge_llm = _FixedJudgeLLM(answerable=False)  # never sufficient
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm, generate_llm=poison_generate)

    result = run_corrective_query(graph, "an unanswerable query", max_iterations=3)

    assert result.loops == 3
    assert result.abstained is True
    assert result.abstain_reason == "budget_exhausted"
    assert result.answer is None
    assert result.grounded is None


def test_persistently_ungrounded_generation_abstains_at_the_budget(tmp_path, fake_embeddings):
    faithfulness_llm = _FixedFaithfulnessLLM(grounded=False)  # never grounded
    graph = _build_graph(tmp_path, fake_embeddings, faithfulness_llm=faithfulness_llm)

    result = run_corrective_query(graph, "some query", max_iterations=3)

    assert result.loops == 3
    assert result.abstained is True
    assert result.abstain_reason == "budget_exhausted"
    assert result.answer is None


def test_budget_guard_respects_a_custom_max_iterations(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(answerable=False)
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_corrective_query(graph, "an unanswerable query", max_iterations=1)

    assert result.loops == 1  # no budget left for even one rewrite
    assert result.abstained is True


# --- fail-open behavior: each of the five call sites, on its own ---

def test_route_disabled_never_calls_the_route_llm(tmp_path, fake_embeddings):
    poison_route = _RaisingLLM(AssertionError("route LLM must not be called when route_enabled=False"))
    graph = _build_graph(tmp_path, fake_embeddings, route_llm=poison_route)

    result = run_corrective_query(graph, "any query", route_enabled=False)

    assert result.route == "both"
    assert result.route_call_failed is False
    assert result.abstained is False


def test_route_call_failure_defaults_to_both_and_does_not_block_the_rest_of_the_graph(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, route_llm=_RaisingLLM(RuntimeError("bad key")))

    result = run_corrective_query(graph, "any query")

    assert result.route == "both"
    assert result.route_call_failed is True
    assert result.abstained is False  # rest of the graph still ran normally


def test_grade_documents_call_failure_fails_open_to_sufficient_and_skips_the_rewrite_loop(tmp_path, fake_embeddings):
    poison_rewrite = _RaisingLLM(AssertionError("rewrite must not be called: fail-open goes straight to generate"))
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=_RaisingLLM(ValueError("malformed")), rewrite_llm=poison_rewrite)

    result = run_corrective_query(graph, "any query")

    assert result.doc_grade_call_failed is True
    assert result.loops == 1  # fail-open to sufficient, no rewrite consumed
    assert result.abstained is False


def test_rewrite_call_failure_reuses_the_previous_query_and_still_consumes_budget(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(answerable=False)  # forces the rewrite path every pass
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm, rewrite_llm=_RaisingLLM(ValueError("malformed")))

    result = run_corrective_query(graph, "the original query", max_iterations=3)

    assert result.rewrite_call_failed is True
    assert result.final_query == "the original query"  # unchanged: rewrite failed, no fabricated query
    assert result.loops == 3  # still bounded -- a broken rewrite LLM can't spin forever
    assert result.abstained is True


def test_generation_call_failure_is_graded_not_grounded_and_never_silently_answers(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, generate_llm=_RaisingLLM(RuntimeError("timeout")))

    result = run_corrective_query(graph, "any query", max_iterations=3)

    assert result.generation_call_failed is True
    assert result.answer is None
    assert result.abstained is True  # generation never recovers across retries in this test
    assert result.loops == 3


def test_faithfulness_call_failure_fails_open_to_not_grounded_not_grounded(tmp_path, fake_embeddings):
    """Asymmetric with grade_documents' fail-open: an unverifiable answer
    must never be presented as grounded just because the checker broke."""
    graph = _build_graph(tmp_path, fake_embeddings, faithfulness_llm=_RaisingLLM(ValueError("malformed")))

    result = run_corrective_query(graph, "any query", max_iterations=3)

    assert result.faithfulness_call_failed is True
    assert result.abstained is True  # never resolves to "grounded" on a broken checker
    assert result.answer is None


def test_empty_hits_do_not_crash_the_pipeline(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, faq_docs=[], policy_chunks=[])

    result = run_corrective_query(graph, "anything", max_iterations=2)

    assert result.hits == [] or result.abstained is True  # never crashes either way


# --- observability: the trace records the actual path taken ---

def test_trace_records_the_actual_node_path_including_the_correction(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(sequence=[False, True])
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_corrective_query(graph, "some query")

    node_path = [entry["node"] for entry in result.trace]
    assert node_path == ["grade_documents", "rewrite_query", "grade_documents", "generate", "grade_generation"]


def test_trace_records_the_abstain_node_on_budget_exhaustion(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(answerable=False)
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_corrective_query(graph, "an unanswerable query", max_iterations=2)

    assert result.trace[-1]["node"] == "abstain"


# --- config -------------------------------------------------------------

def test_corrective_agentic_config_defaults_to_openai_gpt4o_mini():
    config = CorrectiveAgenticConfig()
    assert config.llm.backend == "openai"
    assert config.llm.model_name == "gpt-4o-mini"


def test_corrective_agentic_config_pins_temperature_zero_and_a_fixed_seed():
    config = CorrectiveAgenticConfig()
    assert config.llm.temperature == 0.0
    assert config.llm.seed == 42


def test_corrective_agentic_config_default_max_iterations_is_three():
    assert CorrectiveAgenticConfig().max_iterations == 3


# --- eval/metrics.py phase-5 additions -----------------------------------

def test_faithfulness_rate_computes_fraction_grounded():
    assert metrics.faithfulness_rate([True, True, False, True]) == 0.75


def test_faithfulness_rate_is_none_when_nothing_was_generated():
    assert metrics.faithfulness_rate([]) is None


def test_mean_loops_averages_loop_counts():
    assert metrics.mean_loops([1, 2, 3]) == 2.0


def test_mean_loops_is_zero_for_no_data():
    assert metrics.mean_loops([]) == 0.0


def test_corrective_fire_rate_computes_fraction_that_looped():
    assert metrics.corrective_fire_rate([True, False, False, True]) == 0.5


# --- eval/corrective_sweep.py pure functions -----------------------------

def test_rescue_target_ids_are_the_should_abstain_ids_minus_the_neither_queries():
    assert csweep.RESCUE_TARGET_IDS <= csweep.SHOULD_ABSTAIN_IDS
    assert len(csweep.RESCUE_TARGET_IDS) == 7
    assert not any(qid.startswith("neither-") for qid in csweep.RESCUE_TARGET_IDS)


def test_fail_open_ids_flags_any_of_the_five_call_sites():
    per_query_run = [
        {"id": "a", "route_call_failed": False, "doc_grade_call_failed": False,
         "rewrite_call_failed": False, "generation_call_failed": False, "faithfulness_call_failed": False},
        {"id": "b", "route_call_failed": False, "doc_grade_call_failed": True,
         "rewrite_call_failed": False, "generation_call_failed": False, "faithfulness_call_failed": False},
        {"id": "c", "route_call_failed": False, "doc_grade_call_failed": False,
         "rewrite_call_failed": False, "generation_call_failed": False, "faithfulness_call_failed": True},
    ]
    assert csweep.fail_open_ids(per_query_run) == frozenset({"b", "c"})


def test_matrices_for_run_scores_abstention_against_should_abstain():
    per_query_run = [
        {"id": "a", "abstained": True, "should_abstain": True},
        {"id": "b", "abstained": False, "should_abstain": False},
        {"id": "c", "abstained": False, "should_abstain": True},  # a "rescue" -- see rescue_rate test below
    ]
    cm = csweep.matrices_for_run(per_query_run)
    assert (cm.tp, cm.tn, cm.fn) == (1, 1, 1)


def test_rescue_rate_counts_non_abstains_on_the_seven_target_ids_as_wins():
    rescue_id = next(iter(csweep.RESCUE_TARGET_IDS))
    per_query_run = [
        {"id": rescue_id, "abstained": False},
        {"id": "neither-01", "abstained": True},  # not a rescue target, excluded from this metric
    ]
    result = csweep.rescue_rate_for_run(per_query_run)
    assert result == {"n": 1, "rescued": 1, "rate": 1.0}


def test_rescue_rate_is_none_when_no_target_ids_are_present():
    assert "either-01" not in csweep.RESCUE_TARGET_IDS
    result = csweep.rescue_rate_for_run([{"id": "either-01", "abstained": False}])
    assert result == {"n": 0, "rescued": 0, "rate": None}


class _CountingJudgeRunnable:
    """Stands in for the real judge_llm base runnable, wrapped by
    CachedStructuredLLM below -- counts real (non-cached) invocations."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt: str) -> JudgeDecision:
        self.calls += 1
        return JudgeDecision(answerable=True, reasoning="fixed for test")


def test_run_corrective_arm_advances_run_index_on_all_five_wrappers_not_just_some(tmp_path, fake_embeddings):
    """Integration-level version of the cache/run_index fix eval/agentic_sweep.py's
    run_arm_c already had a test for (test_run_arm_c_uses_a_fresh_sample_per_run_not_a_cached_replay)
    -- this graph wraps FIVE LLMs, not two, so a partial fix (e.g. only
    updating route_llm/judge_llm's run_index and forgetting rewrite/
    generate/faithfulness) would silently collapse those three nodes'
    3-run range to zero by construction, exactly like the bug phase 4's
    pre-run review caught. Driving run_corrective_arm through the real
    graph must invoke EVERY wrapped LLM once per run, not once total.
    """
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    cache = {}
    underlying_judge = _CountingJudgeRunnable()

    route_llm = CachedStructuredLLM(_FixedRouteLLM(), RouteDecision, "route", "gpt-4o-mini", cache)
    judge_llm = CachedStructuredLLM(underlying_judge, JudgeDecision, "grade_documents", "gpt-4o-mini", cache)
    rewrite_llm = CachedStructuredLLM(_FixedRewriteLLM(), RewrittenQuery, "rewrite_query", "gpt-4o-mini", cache)
    generate_llm = CachedStructuredLLM(_FixedGenerateLLM(), GeneratedAnswer, "generate", "gpt-4o-mini", cache)
    faithfulness_llm = CachedStructuredLLM(
        _FixedFaithfulnessLLM(), FaithfulnessGrade, "grade_generation", "gpt-4o-mini", cache,
    )
    llms = [route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm]
    graph = build_corrective_graph(route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm, faq_store, policy_index, k=5)

    queries = [{"id": "q1", "text": "How do I register?", "expected_source": "faq"}]
    csweep.run_corrective_arm(graph, llms, queries, route_enabled=True, max_iterations=3, n_runs=3)

    assert underlying_judge.calls == 3  # one real call per run, not a single cached replay


def test_run_corrective_arm_is_a_cache_hit_on_a_second_execution_with_the_same_run_indices(tmp_path, fake_embeddings):
    """The other half of the same requirement (mirrors
    test_agentic.py's cache tests): re-running with the SAME run indices
    (e.g. re-invoking the harness after a scoring bug fix, not a fresh
    3-run sample) must be a cache hit, not a re-spend."""
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    cache = {}
    underlying_judge = _CountingJudgeRunnable()

    route_llm = CachedStructuredLLM(_FixedRouteLLM(), RouteDecision, "route", "gpt-4o-mini", cache)
    judge_llm = CachedStructuredLLM(underlying_judge, JudgeDecision, "grade_documents", "gpt-4o-mini", cache)
    rewrite_llm = CachedStructuredLLM(_FixedRewriteLLM(), RewrittenQuery, "rewrite_query", "gpt-4o-mini", cache)
    generate_llm = CachedStructuredLLM(_FixedGenerateLLM(), GeneratedAnswer, "generate", "gpt-4o-mini", cache)
    faithfulness_llm = CachedStructuredLLM(
        _FixedFaithfulnessLLM(), FaithfulnessGrade, "grade_generation", "gpt-4o-mini", cache,
    )
    llms = [route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm]
    graph = build_corrective_graph(route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm, faq_store, policy_index, k=5)

    queries = [{"id": "q1", "text": "How do I register?", "expected_source": "faq"}]
    csweep.run_corrective_arm(graph, llms, queries, route_enabled=True, max_iterations=3, n_runs=3)
    csweep.run_corrective_arm(graph, llms, queries, route_enabled=True, max_iterations=3, n_runs=3)

    assert underlying_judge.calls == 3  # still 3, not 6: second pass over the same 3 run indices was all cache hits


def test_faithfulness_rate_for_run_excludes_abstained_queries_with_no_grounded_verdict():
    per_query_run = [
        {"id": "a", "grounded": True},
        {"id": "b", "grounded": None},  # abstained -- nothing generated
        {"id": "c", "grounded": False},
    ]
    assert csweep.faithfulness_rate_for_run(per_query_run) == 0.5


def test_mean_loops_and_corrective_fire_rate_for_run():
    per_query_run = [
        {"id": "a", "loops": 1, "corrective_fired": False},
        {"id": "b", "loops": 3, "corrective_fired": True},
    ]
    assert csweep.mean_loops_for_run(per_query_run) == 2.0
    assert csweep.corrective_fire_rate_for_run(per_query_run) == 0.5


# --- eval/run_corrective_eval.py acceptance check ------------------------

def test_acceptance_check_flags_fail_open_on_a_should_abstain_id():
    should_abstain_id = next(iter(csweep.SHOULD_ABSTAIN_IDS))
    run = [
        {"id": should_abstain_id, "route_call_failed": False, "doc_grade_call_failed": True,
         "rewrite_call_failed": False, "generation_call_failed": False, "faithfulness_call_failed": False},
        {"id": "either-01", "route_call_failed": False, "doc_grade_call_failed": False,
         "rewrite_call_failed": False, "generation_call_failed": False, "faithfulness_call_failed": False},
    ]
    reasons = _acceptance_check(runs=[run], real_calls_made=1)
    assert reasons
    assert any("RE-RUN REQUIRED (METHODOLOGY.md #15a)" in r for r in reasons)
    assert any(should_abstain_id in r for r in reasons)


def test_acceptance_check_flags_zero_real_calls():
    reasons = _acceptance_check(runs=[], real_calls_made=0)
    assert reasons
    assert any("zero real" in r for r in reasons)


def test_acceptance_check_clean_run_has_no_reasons():
    run = [
        {"id": "either-01", "route_call_failed": False, "doc_grade_call_failed": False,
         "rewrite_call_failed": False, "generation_call_failed": False, "faithfulness_call_failed": False},
        {"id": "faq-02", "route_call_failed": False, "doc_grade_call_failed": False,
         "rewrite_call_failed": False, "generation_call_failed": True, "faithfulness_call_failed": False},  # not one of the 12
    ]
    reasons = _acceptance_check(runs=[run], real_calls_made=918)
    assert reasons == []


# --- real API, gated, skipped in CI --------------------------------------

@requires_llm
def test_real_openai_structured_output_returns_valid_schema_for_all_five_calls():
    config = CorrectiveAgenticConfig()
    route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm = get_corrective_structured_llms(config)

    route_result = route_llm.invoke("Decide: faq, policy_pdf, or both? Question: How do I log in to the app?")
    assert isinstance(route_result, RouteDecision)

    judge_result = judge_llm.invoke(
        "Question: What color is the sky?\nRetrieved excerpts:\n1. [faq] The app has a blue icon."
    )
    assert isinstance(judge_result, JudgeDecision)

    rewrite_result = rewrite_llm.invoke("Original question: how do I log in\n\nThis is attempt 2 of 3.\n")
    assert isinstance(rewrite_result, RewrittenQuery)
    assert rewrite_result.rewritten_query

    generate_result = generate_llm.invoke(
        "Question: What color is the sky?\n\nRetrieved excerpts:\n1. [faq] The sky is blue."
    )
    assert isinstance(generate_result, GeneratedAnswer)
    assert generate_result.answer

    faithfulness_result = faithfulness_llm.invoke(
        "Question: What color is the sky?\n\nRetrieved excerpts:\n1. [faq] The sky is blue.\n\n"
        "Generated answer:\nThe sky is blue."
    )
    assert isinstance(faithfulness_result, FaithfulnessGrade)
    assert isinstance(faithfulness_result.grounded, bool)
