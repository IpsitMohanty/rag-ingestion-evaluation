"""Coverage: phase 6's multi-agent graph (src/adapters/multiagent_critic_rag.py)
-- the same corrective loop tests/test_corrective_rag.py covers,
reorganized into three named agents. Mirrors that file's fixtures/fake-LLM
pattern (docs/phase6_preregistration.md #6a's predicted behavioral
parity is exactly what these tests check), plus coverage specific to
this module's new seams: the retrieval agent's route-once-never-again
branching, and the ce/lora NotImplementedError gate.

Entirely mocked-LLM: no OPENAI_API_KEY, no network, no real API spend.
"""
from langchain_core.documents import Document

from adapters.agentic import JudgeDecision, RouteDecision
from adapters.corrective_rag import (
    FaithfulnessGrade,
    GeneratedAnswer,
    RewrittenQuery,
    build_corrective_graph,
    run_corrective_query,
)
from adapters.multiagent_critic_rag import (
    build_multiagent_critic_graph,
    get_multiagent_structured_llms,
    run_multiagent_critic_query,
)
from config import MultiAgentCriticConfig, RetrieverConfig
from conftest import requires_llm

from eval.retrievers import build_faq_index, build_policy_index

FAQ_DOCS = [
    Document(page_content="Question: How do I register? Answer: Open the app.", metadata={"faq_index": 0}),
]
POLICY_CHUNKS = [
    Document(page_content="Anganwadi Workers must have a minimum qualification.", metadata={"page": 5}),
]


class _FixedRouteLLM:
    def __init__(self, source: str = "both"):
        self._source = source
        self.call_count = 0

    def invoke(self, prompt: str) -> RouteDecision:
        self.call_count += 1
        return RouteDecision(source=self._source, reasoning="fixed for test")


class _FixedJudgeLLM:
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
    faq_docs=FAQ_DOCS, policy_chunks=POLICY_CHUNKS, critic="baseline",
):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings, faq_docs, policy_chunks)
    return build_multiagent_critic_graph(
        route_llm or _FixedRouteLLM(),
        judge_llm or _FixedJudgeLLM(),
        rewrite_llm or _FixedRewriteLLM(),
        generate_llm or _FixedGenerateLLM(),
        faithfulness_llm or _FixedFaithfulnessLLM(),
        faq_store, policy_index, k=5, critic=critic,
    )


# --- happy paths: no correction needed, one correction, two corrections ---

def test_sufficient_documents_and_grounded_generation_skip_the_loop_entirely(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings)

    result = run_multiagent_critic_query(graph, "How do I register?")

    assert result.loops == 1
    assert result.corrective_fired is False
    assert result.abstained is False
    assert result.answer == "a grounded answer"
    assert result.grounded is True


def test_insufficient_documents_trigger_one_rewrite_then_generate(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(sequence=[False, True])
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_multiagent_critic_query(graph, "some vague query")

    assert result.loops == 2
    assert result.corrective_fired is True
    assert result.abstained is False
    assert result.final_query == "a rewritten query"


def test_ungrounded_generation_triggers_one_rewrite_then_succeeds(tmp_path, fake_embeddings):
    faithfulness_llm = _FixedFaithfulnessLLM(sequence=[False, True])
    graph = _build_graph(tmp_path, fake_embeddings, faithfulness_llm=faithfulness_llm)

    result = run_multiagent_critic_query(graph, "some query")

    assert result.loops == 2
    assert result.corrective_fired is True
    assert result.abstained is False
    assert result.grounded is True


# --- budget guard: bounded cycles, graceful abstention, never a hallucination ---

def test_persistently_insufficient_documents_abstain_at_the_budget_and_never_generate(tmp_path, fake_embeddings):
    poison_generate = _RaisingLLM(AssertionError("generate must not be called if documents never became sufficient"))
    judge_llm = _FixedJudgeLLM(answerable=False)
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm, generate_llm=poison_generate)

    result = run_multiagent_critic_query(graph, "an unanswerable query", max_iterations=3)

    assert result.loops == 3
    assert result.abstained is True
    assert result.abstain_reason == "budget_exhausted"
    assert result.answer is None
    assert result.grounded is None


def test_persistently_ungrounded_generation_abstains_at_the_budget(tmp_path, fake_embeddings):
    faithfulness_llm = _FixedFaithfulnessLLM(grounded=False)
    graph = _build_graph(tmp_path, fake_embeddings, faithfulness_llm=faithfulness_llm)

    result = run_multiagent_critic_query(graph, "some query", max_iterations=3)

    assert result.loops == 3
    assert result.abstained is True
    assert result.abstain_reason == "budget_exhausted"
    assert result.answer is None


def test_budget_guard_respects_a_custom_max_iterations(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(answerable=False)
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_multiagent_critic_query(graph, "an unanswerable query", max_iterations=1)

    assert result.loops == 1
    assert result.abstained is True


# --- fail-open behavior: each call site, on its own ---

def test_route_disabled_never_calls_the_route_llm(tmp_path, fake_embeddings):
    poison_route = _RaisingLLM(AssertionError("route LLM must not be called when route_enabled=False"))
    graph = _build_graph(tmp_path, fake_embeddings, route_llm=poison_route)

    result = run_multiagent_critic_query(graph, "any query", route_enabled=False)

    assert result.route == "both"
    assert result.route_call_failed is False
    assert result.abstained is False


def test_route_call_failure_defaults_to_both_and_does_not_block_the_rest_of_the_graph(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, route_llm=_RaisingLLM(RuntimeError("bad key")))

    result = run_multiagent_critic_query(graph, "any query")

    assert result.route == "both"
    assert result.route_call_failed is True
    assert result.abstained is False


def test_critic_agent_call_failure_fails_open_to_sufficient_and_skips_the_rewrite_loop(tmp_path, fake_embeddings):
    poison_rewrite = _RaisingLLM(AssertionError("rewrite must not be called: fail-open goes straight to generate"))
    graph = _build_graph(
        tmp_path, fake_embeddings, judge_llm=_RaisingLLM(ValueError("malformed")), rewrite_llm=poison_rewrite,
    )

    result = run_multiagent_critic_query(graph, "any query")

    assert result.doc_grade_call_failed is True
    assert result.loops == 1
    assert result.abstained is False


def test_rewrite_call_failure_reuses_the_previous_query_and_still_consumes_budget(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(answerable=False)
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm, rewrite_llm=_RaisingLLM(ValueError("malformed")))

    result = run_multiagent_critic_query(graph, "the original query", max_iterations=3)

    assert result.rewrite_call_failed is True
    assert result.final_query == "the original query"
    assert result.loops == 3
    assert result.abstained is True


def test_generation_call_failure_is_graded_not_grounded_and_never_silently_answers(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, generate_llm=_RaisingLLM(RuntimeError("timeout")))

    result = run_multiagent_critic_query(graph, "any query", max_iterations=3)

    assert result.generation_call_failed is True
    assert result.answer is None
    assert result.abstained is True
    assert result.loops == 3


def test_faithfulness_call_failure_fails_open_to_not_grounded(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, faithfulness_llm=_RaisingLLM(ValueError("malformed")))

    result = run_multiagent_critic_query(graph, "any query", max_iterations=3)

    assert result.faithfulness_call_failed is True
    assert result.abstained is True
    assert result.answer is None


def test_empty_hits_do_not_crash_the_pipeline(tmp_path, fake_embeddings):
    graph = _build_graph(tmp_path, fake_embeddings, faq_docs=[], policy_chunks=[])

    result = run_multiagent_critic_query(graph, "anything", max_iterations=2)

    assert result.hits == [] or result.abstained is True


# --- this module's own seams: route-once-never-again, ce/lora deferred ---

def test_route_llm_is_called_exactly_once_even_across_multiple_retries(tmp_path, fake_embeddings):
    """#6a's implementation note: the retrieval agent never re-routes on a
    retry entry, only rewrite_query+retrieve -- unlike route, which only
    phase 5's fixed START edge (not the loop) ever reaches."""
    route_llm = _FixedRouteLLM()
    judge_llm = _FixedJudgeLLM(sequence=[False, False, True])  # forces two retries
    graph = _build_graph(tmp_path, fake_embeddings, route_llm=route_llm, judge_llm=judge_llm)

    result = run_multiagent_critic_query(graph, "some vague query", max_iterations=3)

    assert result.loops == 3
    assert route_llm.call_count == 1  # not 3


def test_critic_ce_raises_not_implemented_before_any_node_runs(tmp_path, fake_embeddings):
    poison_everything = _RaisingLLM(AssertionError("no LLM should be invoked -- the graph must not even build"))
    try:
        _build_graph(
            tmp_path, fake_embeddings, critic="ce",
            route_llm=poison_everything, judge_llm=poison_everything,
            rewrite_llm=poison_everything, generate_llm=poison_everything, faithfulness_llm=poison_everything,
        )
        assert False, "expected NotImplementedError for critic='ce'"
    except NotImplementedError as exc:
        assert "docs/phase6_preregistration.md #5b" in str(exc)


def test_critic_lora_raises_not_implemented_before_any_node_runs(tmp_path, fake_embeddings):
    try:
        _build_graph(tmp_path, fake_embeddings, critic="lora")
        assert False, "expected NotImplementedError for critic='lora'"
    except NotImplementedError as exc:
        assert "#5b" in str(exc)


def test_unknown_critic_value_raises_value_error(tmp_path, fake_embeddings):
    try:
        _build_graph(tmp_path, fake_embeddings, critic="not-a-real-critic")
        assert False, "expected ValueError for an unrecognized critic value"
    except ValueError:
        pass


# --- observability: the trace records the actual agent path taken ---

def test_trace_records_the_actual_agent_path_including_the_correction(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(sequence=[False, True])
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_multiagent_critic_query(graph, "some query")

    node_path = [entry["node"] for entry in result.trace]
    assert node_path == ["retrieval_agent", "critic_agent", "retrieval_agent", "critic_agent", "generator_agent"]


def test_trace_records_the_abstain_node_on_budget_exhaustion(tmp_path, fake_embeddings):
    judge_llm = _FixedJudgeLLM(answerable=False)
    graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm)

    result = run_multiagent_critic_query(graph, "an unanswerable query", max_iterations=2)

    assert result.trace[-1]["node"] == "abstain"


# --- behavioral parity with phase 5's monolithic graph (docs/phase6_preregistration.md #6a's prediction 1) ---

def _build_corrective_graph_for_parity(tmp_path, fake_embeddings, judge_llm=None, faithfulness_llm=None):
    faq_store, policy_index = _real_faq_and_policy(tmp_path, fake_embeddings)
    return build_corrective_graph(
        _FixedRouteLLM(), judge_llm or _FixedJudgeLLM(), _FixedRewriteLLM(), _FixedGenerateLLM(),
        faithfulness_llm or _FixedFaithfulnessLLM(), faq_store, policy_index, k=5,
    )


def test_baseline_critic_reproduces_phase_5s_decision_on_a_clean_pass(tmp_path, fake_embeddings):
    old_graph = _build_corrective_graph_for_parity(tmp_path, fake_embeddings)
    new_graph = _build_graph(tmp_path, fake_embeddings)

    old_result = run_corrective_query(old_graph, "How do I register?")
    new_result = run_multiagent_critic_query(new_graph, "How do I register?")

    assert (old_result.loops, old_result.abstained, old_result.answer, old_result.grounded) == \
        (new_result.loops, new_result.abstained, new_result.answer, new_result.grounded)


def test_baseline_critic_reproduces_phase_5s_decision_on_a_corrected_pass(tmp_path, fake_embeddings):
    judge_llm_old = _FixedJudgeLLM(sequence=[False, True])
    judge_llm_new = _FixedJudgeLLM(sequence=[False, True])  # same fixed sequence, independent instance
    old_graph = _build_corrective_graph_for_parity(tmp_path, fake_embeddings, judge_llm=judge_llm_old)
    new_graph = _build_graph(tmp_path, fake_embeddings, judge_llm=judge_llm_new)

    old_result = run_corrective_query(old_graph, "some vague query")
    new_result = run_multiagent_critic_query(new_graph, "some vague query")

    assert (old_result.loops, old_result.corrective_fired, old_result.abstained, old_result.answer) == \
        (new_result.loops, new_result.corrective_fired, new_result.abstained, new_result.answer)


# --- config ---------------------------------------------------------------

def test_multiagent_critic_config_defaults_to_openai_gpt4o_mini():
    config = MultiAgentCriticConfig()
    assert config.llm.backend == "openai"
    assert config.llm.model_name == "gpt-4o-mini"


def test_multiagent_critic_config_pins_temperature_zero_and_a_fixed_seed():
    config = MultiAgentCriticConfig()
    assert config.llm.temperature == 0.0
    assert config.llm.seed == 42


def test_multiagent_critic_config_default_max_iterations_is_three():
    assert MultiAgentCriticConfig().max_iterations == 3


def test_multiagent_critic_config_default_critic_is_baseline():
    assert MultiAgentCriticConfig().critic == "baseline"


# --- real API, gated, skipped in CI --------------------------------------

@requires_llm
def test_real_openai_structured_output_returns_valid_schema_for_all_five_calls():
    config = MultiAgentCriticConfig()
    route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm = get_multiagent_structured_llms(config)

    route_result = route_llm.invoke("Decide: faq, policy_pdf, or both? Question: How do I log in to the app?")
    assert isinstance(route_result, RouteDecision)

    judge_result = judge_llm.invoke(
        "Question: What color is the sky?\nRetrieved excerpts:\n1. [faq] The app has a blue icon."
    )
    assert isinstance(judge_result, JudgeDecision)

    rewrite_result = rewrite_llm.invoke("Original question: how do I log in\n\nThis is attempt 2 of 3.\n")
    assert isinstance(rewrite_result, RewrittenQuery)
    assert rewrite_result.rewritten_query

    generate_result = generate_llm.invoke("Question: How do I log in?\nRetrieved excerpts:\n1. [faq] Open the app.")
    assert isinstance(generate_result, GeneratedAnswer)

    faithfulness_result = faithfulness_llm.invoke(
        "Question: How do I log in?\nRetrieved excerpts:\n1. [faq] Open the app.\nGenerated answer:\nOpen the app."
    )
    assert isinstance(faithfulness_result, FaithfulnessGrade)
