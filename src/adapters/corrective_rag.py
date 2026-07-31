"""LangGraph adapter for the corrective, self-reflective RAG loop: CRAG-style
retrieval grading + query rewriting around retrieval, Self-RAG-style
groundedness grading around a new generation step. Second module in this
repo importing langgraph directly -- src/adapters/agentic.py is the first,
and is left entirely unmodified by this module. Its linear
route/retrieve/judge/respond graph remains a documented baseline and
comparison point, not a design this module replaces or supersedes.

Naming: agentic.py's graph is linear, no loops, no conditional edges, and is
explicitly documented there as "not an agent". THIS graph has genuine
conditional branching (decide_to_generate, decide_after_generation) and a
bounded corrective loop (grade_documents -> rewrite_query -> retrieve, and
grade_generation -> rewrite_query -> retrieve), so "single-agent" is an
accurate label here, not overclaiming. See eval/METHODOLOGY.md #19 for the
methodology this graph exists to test, and README's Phase 5 section for the
architecture/results summary.

Every cycle is bounded: state's loop_count vs max_iterations (default 3
total retrieve+generate passes, see CorrectiveAgenticConfig). decide_to_generate
and decide_after_generation are the only two places a cycle can continue,
and both check the budget before looping -- on exhaustion, abstain_node
runs, never generate_node again. Same "never hallucinate" ethos as
agentic.py's fail-open convention, extended here to "ran out of retry
budget" as well as "a call failed".

Reuses, not reimplements: RouteDecision/JudgeDecision/ROUTE_PROMPT/
JUDGE_PROMPT/_format_excerpts from adapters.agentic -- grade_documents IS
the same answerability judge phase 4 built, just interpreted as a
retrieval-sufficiency grade instead of a final abstention decision. Also
reuses eval.retrievers.query_routed for retrieval itself: no retrieval
logic is duplicated here, same as agentic.py.
"""
import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from adapters.agentic import JUDGE_PROMPT, ROUTE_PROMPT, JudgeDecision, RouteDecision, _format_excerpts
from config import CorrectiveAgenticConfig


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description="A reformulated version of the question, more likely to retrieve the relevant excerpts"
    )
    reasoning: str = Field(description="One sentence explaining what was changed and why")


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="A grounded answer to the question, using only the provided excerpts")


class FaithfulnessGrade(BaseModel):
    grounded: bool = Field(
        description="True if every factual claim in the answer is directly supported by the excerpts, "
        "false if it states anything the excerpts do not"
    )
    reasoning: str = Field(description="One sentence explaining the decision")


REWRITE_PROMPT = """The excerpts retrieved for the question below were judged \
insufficient to answer it, or an answer generated from them was judged not \
grounded in them. Reformulate the question to be more likely to retrieve \
the relevant content -- for example by making an implicit term explicit, \
trying alternate phrasing, or broadening an overly narrow term. Do not \
change what is being asked, only how it is phrased.

Original question: {query}

This is attempt {attempt} of {max_attempts}.
"""

GENERATE_PROMPT = """Answer the question using ONLY the excerpts below. If \
the excerpts do not contain enough information to answer, say so plainly \
rather than guessing or using outside knowledge.

Question: {query}

Retrieved excerpts:
{excerpts}
"""

FAITHFULNESS_PROMPT = """You are checking whether a generated answer is \
fully grounded in the excerpts it was supposed to be based on -- not \
whether the answer is correct in some absolute sense, and not whether the \
excerpts are sufficient (that was already decided separately). Judge ONLY \
faithfulness: does every factual claim in the answer trace back to \
something actually stated in the excerpts?

Answer "grounded" only if every specific claim in the answer -- a fact, a \
number, a name -- is directly supported by the excerpts. Answer \
"not grounded" if the answer states anything the excerpts do not, even if \
the claim sounds plausible or is likely true in general.

Question: {query}

Retrieved excerpts:
{excerpts}

Generated answer:
{answer}
"""


class CorrectiveAgenticState(TypedDict, total=False):
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
class CorrectiveAgenticResult:
    query: str
    final_query: str
    route: str
    loops: int
    corrective_fired: bool  # loops > 1: at least one rewrite+re-retrieve happened
    abstained: bool
    abstain_reason: str | None
    answer: str | None
    grounded: bool | None  # None when abstained -- nothing was generated to grade
    hits: list = field(repr=False)
    trace: list = field(repr=False)
    route_call_failed: bool = False
    doc_grade_call_failed: bool = False
    rewrite_call_failed: bool = False
    generation_call_failed: bool = False
    faithfulness_call_failed: bool = False


def build_corrective_graph(
    route_llm, judge_llm, rewrite_llm, generate_llm, faithfulness_llm, faq_store, policy_index, k: int,
):
    """route_llm/judge_llm/rewrite_llm/generate_llm/faithfulness_llm:
    structured-output runnables (production: get_corrective_structured_llms
    below; tests: fakes exposing the same .invoke() contract, same pattern
    as tests/test_agentic.py). faq_store/policy_index: the same objects
    eval/retrievers.py's build_faq_index/build_policy_index produce.
    """
    from eval.retrievers import query_routed  # eval/ is a sibling package, not a src/ dependency

    def route_node(state: CorrectiveAgenticState) -> dict:
        if not state.get("route_enabled", True):
            return {"route": "both", "route_reasoning": "routing disabled for this arm"}
        try:
            decision = route_llm.invoke(ROUTE_PROMPT.format(query=state["query"]))
            return {"route": decision.source, "route_reasoning": decision.reasoning}
        except Exception as exc:
            return {
                "route": "both",
                "route_reasoning": f"route call failed ({exc.__class__.__name__}), defaulting to both",
                "route_call_failed": True,
            }

    def retrieve_node(state: CorrectiveAgenticState) -> dict:
        hits = query_routed(state["query"], k, faq_store, policy_index, state["route"])
        return {"hits": hits}

    def grade_documents_node(state: CorrectiveAgenticState) -> dict:
        excerpts_text = _format_excerpts(state["hits"])
        try:
            decision = judge_llm.invoke(JUDGE_PROMPT.format(query=state["query"], excerpts=excerpts_text))
            grade = "sufficient" if decision.answerable else "insufficient"
            return {
                "doc_grade": grade,
                "doc_grade_reasoning": decision.reasoning,
                "trace": [{"node": "grade_documents", "loop": state["loop_count"], "grade": grade}],
            }
        except Exception as exc:
            # Fail open to "sufficient": a grading-call failure is an infra
            # problem, not a content judgment -- try to answer with what was
            # retrieved rather than burn retry budget correcting for a
            # broken grader. Asymmetric with grade_generation's fail-open
            # below on purpose (see that node's comment).
            return {
                "doc_grade": "sufficient",
                "doc_grade_reasoning": f"grading call failed ({exc.__class__.__name__}), defaulting to sufficient",
                "doc_grade_call_failed": True,
                "trace": [{"node": "grade_documents", "loop": state["loop_count"], "grade": "sufficient (fail-open)"}],
            }

    def decide_to_generate(state: CorrectiveAgenticState) -> str:
        if state["doc_grade"] == "sufficient":
            return "generate"
        if state["loop_count"] < state["max_iterations"]:
            return "rewrite_query"
        return "abstain"

    def rewrite_query_node(state: CorrectiveAgenticState) -> dict:
        next_loop = state["loop_count"] + 1
        try:
            decision = rewrite_llm.invoke(
                REWRITE_PROMPT.format(
                    query=state["original_query"], attempt=next_loop, max_attempts=state["max_iterations"],
                )
            )
            new_query, reasoning, failed = decision.rewritten_query, decision.reasoning, False
        except Exception as exc:
            # Fail open to the previous query unchanged: still consumes a
            # loop of budget (so a broken rewrite LLM can't spin forever),
            # but never fabricates a query.
            new_query = state["query"]
            reasoning = f"rewrite call failed ({exc.__class__.__name__}), reusing previous query"
            failed = True
        return {
            "query": new_query,
            "loop_count": next_loop,
            "rewrite_reasoning": reasoning,
            "rewrite_call_failed": failed,
            "query_history": [new_query],
            "trace": [{"node": "rewrite_query", "loop": next_loop, "query": new_query}],
        }

    def generate_node(state: CorrectiveAgenticState) -> dict:
        excerpts_text = _format_excerpts(state["hits"])
        try:
            decision = generate_llm.invoke(
                GENERATE_PROMPT.format(query=state["original_query"], excerpts=excerpts_text)
            )
            return {"answer": decision.answer, "trace": [{"node": "generate", "loop": state["loop_count"]}]}
        except Exception as exc:
            return {
                "answer": None,
                "generation_call_failed": True,
                "trace": [{"node": "generate", "loop": state["loop_count"], "failed": exc.__class__.__name__}],
            }

    def grade_generation_node(state: CorrectiveAgenticState) -> dict:
        if state.get("generation_call_failed") or state.get("answer") is None:
            # Nothing to grade -- a failed generation is never "grounded".
            return {
                "faithfulness": "not_grounded",
                "faithfulness_reasoning": "generation failed, nothing to grade",
                "trace": [{"node": "grade_generation", "loop": state["loop_count"], "grounded": False}],
            }
        excerpts_text = _format_excerpts(state["hits"])
        try:
            decision = faithfulness_llm.invoke(
                FAITHFULNESS_PROMPT.format(
                    query=state["original_query"], excerpts=excerpts_text, answer=state["answer"],
                )
            )
            grade = "grounded" if decision.grounded else "not_grounded"
            return {
                "faithfulness": grade,
                "faithfulness_reasoning": decision.reasoning,
                "trace": [{"node": "grade_generation", "loop": state["loop_count"], "grounded": decision.grounded}],
            }
        except Exception as exc:
            # Fail open to "not_grounded", NOT "grounded": unlike route/
            # grade_documents (where fail-open-to-proceed is safe because
            # worst case is plain retrieval), an unverifiable answer must
            # never be presented as grounded just because the checker
            # itself broke. This is the budget guard's "abstain, never
            # hallucinate" ethos applied to the faithfulness check itself.
            return {
                "faithfulness": "not_grounded",
                "faithfulness_reasoning": f"faithfulness check failed ({exc.__class__.__name__}), defaulting to not_grounded",
                "faithfulness_call_failed": True,
                "trace": [
                    {"node": "grade_generation", "loop": state["loop_count"], "grounded": False, "check_failed": True}
                ],
            }

    def decide_after_generation(state: CorrectiveAgenticState) -> str:
        if state["faithfulness"] == "grounded":
            return "respond"
        if state["loop_count"] < state["max_iterations"]:
            return "rewrite_query"
        return "abstain"

    def respond_node(state: CorrectiveAgenticState) -> dict:
        return {
            "response": {
                "abstained": False, "answer": state["answer"], "hits": state["hits"],
                "message": None, "abstain_reason": None,
            }
        }

    def abstain_node(state: CorrectiveAgenticState) -> dict:
        return {
            "response": {
                "abstained": True, "answer": None, "hits": [],
                "message": "Could not produce a sufficiently grounded answer within the retry budget.",
                "abstain_reason": "budget_exhausted",
            },
            "trace": [{"node": "abstain", "loop": state["loop_count"]}],
        }

    graph = StateGraph(CorrectiveAgenticState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade_generation", grade_generation_node)
    graph.add_node("respond", respond_node)
    graph.add_node("abstain", abstain_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents", decide_to_generate, {"generate": "generate", "rewrite_query": "rewrite_query", "abstain": "abstain"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "grade_generation")
    graph.add_conditional_edges(
        "grade_generation", decide_after_generation,
        {"respond": "respond", "rewrite_query": "rewrite_query", "abstain": "abstain"},
    )
    graph.add_edge("respond", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def run_corrective_query(
    graph, query: str, route_enabled: bool = True, max_iterations: int = 3,
) -> CorrectiveAgenticResult:
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
    return CorrectiveAgenticResult(
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


def get_corrective_structured_llms(config: CorrectiveAgenticConfig):
    """Production construction path, mirroring adapters.agentic.get_structured_llms:
    one real ChatOpenAI client (via adapters/llm.py) wrapped with
    with_structured_output for each of the five schemas this graph uses.
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
