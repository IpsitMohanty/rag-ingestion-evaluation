"""Phase 6 (docs/phase6_preregistration.md #6/#6a): the same corrective,
self-reflective loop `src/adapters/corrective_rag.py` builds, reorganized
into three named agents plus a thin, conditional-edge router, with a
swappable critic. `corrective_rag.py` is left completely unmodified --
this module imports its prompts/schemas, never its logic, and is an
additional, separate graph, not a replacement (same relationship
`corrective_rag.py` itself has to `agentic.py`).

Three agents, five graph nodes total (three agents plus the existing
`respond`/`abstain` terminals -- not counted as agents, same as phase 5):

    retrieval_agent -> critic_agent -> generator_agent -> (thin router)

- **retrieval_agent** does BOTH of phase 5's retrieval-side jobs,
  branching on whether `route` is already set in state (#6a's
  implementation note, since #6's diagram doesn't place `rewrite_query`):
  first entry -> `route` + `retrieve` (`ROUTE_PROMPT`/`RouteDecision`,
  `query_routed`, all reused from `adapters.agentic`/`eval.retrievers`
  unchanged); retry entry -> `rewrite_query` + `retrieve`
  (`REWRITE_PROMPT`/`RewrittenQuery`, reused unchanged from
  `corrective_rag`). Never re-routes on a retry, exactly like phase 5's
  fixed `rewrite_query -> retrieve` edge.
- **critic_agent** is the swappable component (`critic:
  "baseline" | "ce" | "lora"`, `MultiAgentCriticConfig`). `baseline`
  invokes the identical `JUDGE_PROMPT`/`JudgeDecision` call
  `corrective_rag.py`'s `grade_documents_node` makes -- same prompt,
  same schema, same fail-open-to-`sufficient` behavior, byte-for-byte.
  `ce`/`lora` raise `NotImplementedError`: docs/phase6_preregistration.md
  #5b -- the training pool measured below a workable size on this
  corpus twice (0/92 verbatim, 6/92 LLM-paraphrased, both under the
  precommitted floor of 10), so neither arm was built this phase. This
  is a deliberate, documented gap, not an oversight -- the config seam
  exists for a future, non-saturated corpus.
- **generator_agent** does `generate` + `grade_generation` together,
  reusing `GENERATE_PROMPT`/`FAITHFULNESS_PROMPT`/`GeneratedAnswer`/
  `FaithfulnessGrade` unchanged from `corrective_rag.py`. Same fail-open
  asymmetry: a failed generation is never graded grounded; a failed
  faithfulness check fails open to `not_grounded`, never `grounded`.
- **Thin router**: `route_after_critic`/`route_after_generator`
  conditional edges, the same shape as phase 5's `decide_to_generate`/
  `decide_after_generation` -- not promoted to a fifth named node (#6:
  "kept as conditional edges... unless Stage 4 finds a concrete reason
  to split it out"; it didn't).

Every cycle is bounded exactly like phase 5's: `loop_count` vs
`max_iterations`, checked in both conditional-edge functions, budget
exhaustion always routes to `abstain_node`, never back to `generator_agent`.
"""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from adapters.agentic import JUDGE_PROMPT, ROUTE_PROMPT, JudgeDecision, RouteDecision, _format_excerpts
from adapters.corrective_rag import (
    FAITHFULNESS_PROMPT,
    GENERATE_PROMPT,
    REWRITE_PROMPT,
    FaithfulnessGrade,
    GeneratedAnswer,
    RewrittenQuery,
)
from config import MultiAgentCriticConfig


class MultiAgentCriticState(TypedDict, total=False):
    query: str
    original_query: str
    route_enabled: bool
    max_iterations: int
    loop_count: int
    route: Literal["faq", "policy_pdf", "both"]
    route_reasoning: str
    route_call_failed: bool
    hits: list  # list[eval.metrics.ScoredHit]
    doc_grade: Literal["sufficient", "insufficient"]
    doc_grade_reasoning: str
    doc_grade_call_failed: bool
    rewrite_reasoning: str
    rewrite_call_failed: bool
    query_history: Annotated[list[str], operator.add]
    answer: str | None
    generation_call_failed: bool
    faithfulness: Literal["grounded", "not_grounded"]
    faithfulness_reasoning: str
    faithfulness_call_failed: bool
    response: dict
    trace: Annotated[list[dict], operator.add]


@dataclass
class MultiAgentCriticResult:
    query: str
    final_query: str
    route: str
    loops: int
    corrective_fired: bool
    abstained: bool
    abstain_reason: str | None
    answer: str | None
    grounded: bool | None
    hits: list = field(repr=False)
    trace: list = field(repr=False)
    route_call_failed: bool = False
    doc_grade_call_failed: bool = False
    rewrite_call_failed: bool = False
    generation_call_failed: bool = False
    faithfulness_call_failed: bool = False


def build_multiagent_critic_graph(
    route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm,
    faq_store, policy_index, k: int, critic: str = "baseline",
):
    """route_llm/judge_llm/rewrite_llm/generate_llm/faithfulness_llm: same
    five structured-output runnables corrective_rag.get_corrective_structured_llms
    builds -- this function's production construction path is
    get_multiagent_structured_llms below, same pattern. `critic` selects
    critic_agent's implementation; only "baseline" is built this phase
    (#5b) -- "ce"/"lora" raise NotImplementedError immediately, before
    any node runs, not lazily on first use.
    """
    from eval.retrievers import query_routed  # eval/ is a sibling package, not a src/ dependency

    if critic not in ("baseline", "ce", "lora"):
        raise ValueError(f"Unknown critic {critic!r}, expected 'baseline', 'ce', or 'lora'")
    if critic in ("ce", "lora"):
        raise NotImplementedError(
            f"critic={critic!r} is deferred, not built this phase -- "
            f"docs/phase6_preregistration.md #5b: the training pool measured below a workable "
            f"size on this corpus twice (0/92 verbatim, 6/92 LLM-paraphrased, both under the "
            f"precommitted floor of 10). The seam exists for a future, non-saturated corpus."
        )

    def retrieval_agent(state: MultiAgentCriticState) -> dict:
        is_retry = state.get("route") is not None
        if not is_retry:
            if not state.get("route_enabled", True):
                route, route_reasoning, route_failed = "both", "routing disabled for this arm", False
            else:
                try:
                    decision = route_llm.invoke(ROUTE_PROMPT.format(query=state["query"]))
                    route, route_reasoning, route_failed = decision.source, decision.reasoning, False
                except Exception as exc:
                    route = "both"
                    route_reasoning = f"route call failed ({exc.__class__.__name__}), defaulting to both"
                    route_failed = True
            hits = query_routed(state["query"], k, faq_store, policy_index, route)
            return {
                "route": route, "route_reasoning": route_reasoning, "route_call_failed": route_failed,
                "hits": hits,
                "trace": [{"node": "retrieval_agent", "action": "route+retrieve", "loop": state["loop_count"]}],
            }

        # Retry entry: rewrite_query + retrieve, never re-route (phase 5's fixed edge shape, #6a).
        next_loop = state["loop_count"] + 1
        try:
            decision = rewrite_llm.invoke(
                REWRITE_PROMPT.format(
                    query=state["original_query"], attempt=next_loop, max_attempts=state["max_iterations"],
                )
            )
            new_query, rewrite_reasoning, rewrite_failed = decision.rewritten_query, decision.reasoning, False
        except Exception as exc:
            # Fail open to the previous query unchanged -- still consumes budget, never fabricates a query.
            new_query = state["query"]
            rewrite_reasoning = f"rewrite call failed ({exc.__class__.__name__}), reusing previous query"
            rewrite_failed = True
        hits = query_routed(new_query, k, faq_store, policy_index, state["route"])
        return {
            "query": new_query, "loop_count": next_loop,
            "rewrite_reasoning": rewrite_reasoning, "rewrite_call_failed": rewrite_failed,
            "query_history": [new_query], "hits": hits,
            "trace": [{"node": "retrieval_agent", "action": "rewrite_query+retrieve", "loop": next_loop}],
        }

    def critic_agent(state: MultiAgentCriticState) -> dict:
        # critic == "baseline" only, reached here -- ce/lora already raised above, before the
        # graph was ever built, so this branch never has to handle them.
        excerpts_text = _format_excerpts(state["hits"])
        try:
            decision = judge_llm.invoke(JUDGE_PROMPT.format(query=state["query"], excerpts=excerpts_text))
            grade = "sufficient" if decision.answerable else "insufficient"
            return {
                "doc_grade": grade, "doc_grade_reasoning": decision.reasoning,
                "trace": [{"node": "critic_agent", "critic": "baseline", "loop": state["loop_count"], "grade": grade}],
            }
        except Exception as exc:
            # Fail open to "sufficient" -- same asymmetric reasoning as corrective_rag.py's
            # grade_documents_node (#6's hard constraint): an infra failure is not a content judgment.
            return {
                "doc_grade": "sufficient",
                "doc_grade_reasoning": f"grading call failed ({exc.__class__.__name__}), defaulting to sufficient",
                "doc_grade_call_failed": True,
                "trace": [{
                    "node": "critic_agent", "critic": "baseline", "loop": state["loop_count"],
                    "grade": "sufficient (fail-open)",
                }],
            }

    def route_after_critic(state: MultiAgentCriticState) -> str:
        if state["doc_grade"] == "sufficient":
            return "generate"
        if state["loop_count"] < state["max_iterations"]:
            return "retry"
        return "abstain"

    def generator_agent(state: MultiAgentCriticState) -> dict:
        excerpts_text = _format_excerpts(state["hits"])
        try:
            decision = generate_llm.invoke(
                GENERATE_PROMPT.format(query=state["original_query"], excerpts=excerpts_text)
            )
            answer, generation_failed = decision.answer, False
        except Exception as exc:
            answer, generation_failed = None, True
            gen_fail_trace = {
                "node": "generator_agent", "loop": state["loop_count"],
                "generation_failed": exc.__class__.__name__,
            }

        if generation_failed:
            # Nothing to grade -- a failed generation is never "grounded" (same as corrective_rag.py).
            return {
                "answer": None, "generation_call_failed": True,
                "faithfulness": "not_grounded", "faithfulness_reasoning": "generation failed, nothing to grade",
                "trace": [gen_fail_trace],
            }

        try:
            grade_decision = faithfulness_llm.invoke(
                FAITHFULNESS_PROMPT.format(query=state["original_query"], excerpts=excerpts_text, answer=answer)
            )
            faithfulness = "grounded" if grade_decision.grounded else "not_grounded"
            return {
                "answer": answer, "generation_call_failed": False,
                "faithfulness": faithfulness, "faithfulness_reasoning": grade_decision.reasoning,
                "trace": [{
                    "node": "generator_agent", "loop": state["loop_count"], "grounded": grade_decision.grounded,
                }],
            }
        except Exception as exc:
            # Fail open to "not_grounded", NOT "grounded" -- same asymmetric reasoning as
            # corrective_rag.py's grade_generation_node (#6's hard constraint).
            return {
                "answer": answer, "generation_call_failed": False,
                "faithfulness": "not_grounded",
                "faithfulness_reasoning": f"faithfulness check failed ({exc.__class__.__name__}), defaulting to not_grounded",
                "faithfulness_call_failed": True,
                "trace": [{
                    "node": "generator_agent", "loop": state["loop_count"], "grounded": False, "check_failed": True,
                }],
            }

    def route_after_generator(state: MultiAgentCriticState) -> str:
        if state["faithfulness"] == "grounded":
            return "respond"
        if state["loop_count"] < state["max_iterations"]:
            return "retry"
        return "abstain"

    def respond_node(state: MultiAgentCriticState) -> dict:
        return {
            "response": {
                "abstained": False, "answer": state["answer"], "hits": state["hits"],
                "message": None, "abstain_reason": None,
            }
        }

    def abstain_node(state: MultiAgentCriticState) -> dict:
        return {
            "response": {
                "abstained": True, "answer": None, "hits": [],
                "message": "Could not produce a sufficiently grounded answer within the retry budget.",
                "abstain_reason": "budget_exhausted",
            },
            "trace": [{"node": "abstain", "loop": state["loop_count"]}],
        }

    graph = StateGraph(MultiAgentCriticState)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node("generator_agent", generator_agent)
    graph.add_node("respond", respond_node)
    graph.add_node("abstain", abstain_node)

    graph.add_edge(START, "retrieval_agent")
    graph.add_edge("retrieval_agent", "critic_agent")
    graph.add_conditional_edges(
        "critic_agent", route_after_critic,
        {"generate": "generator_agent", "retry": "retrieval_agent", "abstain": "abstain"},
    )
    graph.add_conditional_edges(
        "generator_agent", route_after_generator,
        {"respond": "respond", "retry": "retrieval_agent", "abstain": "abstain"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def run_multiagent_critic_query(
    graph, query: str, route_enabled: bool = True, max_iterations: int = 3,
) -> MultiAgentCriticResult:
    initial_state = {
        "query": query,
        "original_query": query,
        "route_enabled": route_enabled,
        "max_iterations": max_iterations,
        "loop_count": 1,
        "query_history": [query],
        "trace": [],
    }
    final_state = graph.invoke(initial_state)
    response = final_state["response"]
    abstained = response["abstained"]
    return MultiAgentCriticResult(
        query=query,
        final_query=final_state["query"],
        route=final_state["route"],
        loops=final_state["loop_count"],
        corrective_fired=final_state["loop_count"] > 1,
        abstained=abstained,
        abstain_reason=response.get("abstain_reason"),
        answer=response.get("answer"),
        grounded=None if abstained else (final_state.get("faithfulness") == "grounded"),
        hits=final_state.get("hits", []),
        trace=final_state.get("trace", []),
        route_call_failed=final_state.get("route_call_failed", False),
        doc_grade_call_failed=final_state.get("doc_grade_call_failed", False),
        rewrite_call_failed=final_state.get("rewrite_call_failed", False),
        generation_call_failed=final_state.get("generation_call_failed", False),
        faithfulness_call_failed=final_state.get("faithfulness_call_failed", False),
    )


def get_multiagent_structured_llms(config: MultiAgentCriticConfig):
    """Production construction path, mirroring
    corrective_rag.get_corrective_structured_llms. Only meaningful for
    config.critic == "baseline" this phase (#5b) -- build_multiagent_critic_graph
    raises NotImplementedError for "ce"/"lora" before these would ever be used.
    """
    from adapters.llm import get_llm

    base_llm = get_llm(config.llm)
    return (
        base_llm.with_structured_output(RouteDecision),
        base_llm.with_structured_output(JudgeDecision),
        base_llm.with_structured_output(RewrittenQuery),
        base_llm.with_structured_output(GeneratedAnswer),
        base_llm.with_structured_output(FaithfulnessGrade),
    )
