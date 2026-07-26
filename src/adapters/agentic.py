"""LangGraph adapter: the only module in this repo importing langgraph
directly, same isolation discipline as adapters/retriever.py importing
langchain_classic. Builds a small, linear four-node graph -- route,
retrieve, judge, respond -- that tests whether an LLM judge can do what
phase 2's Finding #1 showed a similarity threshold cannot (see
eval/METHODOLOGY.md #9-18 for the evaluation this graph exists to run).

Naming, binding for README/ANALYSIS.md/commits: this is "LLM-routed
retrieval with an LLM abstention judge, built on LangGraph". A linear
pipeline, two LLM calls, no loops, no tool selection beyond retrieval, no
self-correction. Not an agent; never call it one.

route_llm/judge_llm are injected as already-built structured-output
runnables (an object exposing .invoke(prompt) -> RouteDecision /
JudgeDecision) rather than constructed inside this module. That keeps
the graph testable with a fake object standing in for
ChatOpenAI(...).with_structured_output(...), with no API key and no
langchain_openai import required for structural tests -- see
get_structured_llms below for the one place that builds the real thing.
"""
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import AgenticConfig


class RouteDecision(BaseModel):
    source: Literal["faq", "policy_pdf", "both"] = Field(
        description="Which corpus (or both) should be searched to answer the question"
    )
    reasoning: str = Field(description="One sentence explaining the decision")


class JudgeDecision(BaseModel):
    answerable: bool = Field(
        description="True if the excerpts state the specific answer the question asks for, false otherwise"
    )
    reasoning: str = Field(description="One sentence explaining the decision")


ROUTE_PROMPT = """You are a query router for a two-part document corpus:

1. "faq": short Q&A pairs about using the Poshan Tracker mobile app \
(registration, beneficiary types, screens/features, login, data entry).
2. "policy_pdf": a long-form government policy document (Mission Saksham \
Anganwadi & Poshan 2.0 scheme guidelines): staffing roles, eligibility \
criteria, scheme structure, nutrition programs, administrative procedure.

Decide which corpus (or both) should be searched to answer the question below.
Respond "both" only if the question could plausibly be answered from either, \
or requires combining both.

Question: {query}
"""

JUDGE_PROMPT = """You are evaluating retrieval quality for a question-answering \
system over a fixed, known corpus (Poshan Tracker app FAQs and Mission Saksham \
Anganwadi & Poshan 2.0 scheme guidelines). You will be shown a user's question \
and a list of text excerpts a retriever selected as its best guesses for \
relevant content.

Decide whether these SPECIFIC excerpts actually state enough information to \
answer the question. Do not use your own general knowledge, and do not credit \
an excerpt for being on-topic or adjacent to the subject if it doesn't state \
the actual answer.

Answer "answerable" only if at least one excerpt states the specific \
information the question asks for.
Answer "not_answerable" if the excerpts are off-topic, or merely reference \
that something exists or is relevant without stating the specifics asked for.

Question: {query}

Retrieved excerpts:
{excerpts}
"""


class AgenticState(TypedDict, total=False):
    query: str
    route_enabled: bool
    route: Literal["faq", "policy_pdf", "both"]
    route_reasoning: str
    hits: list  # list[eval.metrics.ScoredHit] -- not redefined here, eval/ owns that type
    judgment: Literal["answerable", "not_answerable"]
    judge_reasoning: str
    response: dict
    route_call_failed: bool
    judge_call_failed: bool


@dataclass
class AgenticResult:
    query: str
    route: str
    route_reasoning: str
    hits: list = field(repr=False)
    judgment: str
    judge_reasoning: str
    abstained: bool
    message: str | None
    route_call_failed: bool = False
    judge_call_failed: bool = False


def _format_excerpts(hits) -> str:
    if not hits:
        return "(no excerpts retrieved)"
    return "\n".join(f"{i}. [{h.source_type}] {h.document.page_content}" for i, h in enumerate(hits, start=1))


def build_agentic_graph(route_llm, judge_llm, faq_store, policy_index, k: int):
    """route_llm / judge_llm: structured-output runnables (production:
    ChatOpenAI(...).with_structured_output(RouteDecision/JudgeDecision),
    see get_structured_llms; tests: a fake exposing the same .invoke()
    contract). faq_store / policy_index: the SAME objects
    eval/retrievers.py's build_faq_index/build_policy_index already
    produce -- retrieval itself is never reimplemented here, only gated
    by the route node's decision via eval.retrievers.query_routed.
    """
    from eval.retrievers import query_routed  # eval/ is a sibling package, not a src/ dependency

    def route_node(state: AgenticState) -> dict:
        if not state.get("route_enabled", True):
            # C1 (eval/METHODOLOGY.md #9): routing disabled entirely, no
            # LLM call made. Always "both", identical to arms A/B's
            # retrieval, by construction.
            return {"route": "both", "route_reasoning": "routing disabled for this arm"}
        try:
            decision = route_llm.invoke(ROUTE_PROMPT.format(query=state["query"]))
            return {"route": decision.source, "route_reasoning": decision.reasoning}
        except Exception as exc:
            # Fail open to "both": if the router itself is unavailable,
            # searching everything is safer than guessing a narrower
            # source. Never crash the graph on a route-call failure.
            return {
                "route": "both",
                "route_reasoning": f"route call failed ({exc.__class__.__name__}), defaulting to both",
                "route_call_failed": True,
            }

    def retrieve_node(state: AgenticState) -> dict:
        hits = query_routed(state["query"], k, faq_store, policy_index, state["route"])
        return {"hits": hits}

    def judge_node(state: AgenticState) -> dict:
        excerpts_text = _format_excerpts(state["hits"])
        try:
            decision = judge_llm.invoke(JUDGE_PROMPT.format(query=state["query"], excerpts=excerpts_text))
            return {
                "judgment": "answerable" if decision.answerable else "not_answerable",
                "judge_reasoning": decision.reasoning,
            }
        except Exception as exc:
            # Fail open to "answerable": on a judge-call failure, behave
            # like plain retrieval (never abstain because of an infra
            # hiccup, not a real content judgment). Mirrors
            # app/app_logic.py::generate_answer's existing fail-open
            # convention for LLM-call failures.
            return {
                "judgment": "answerable",
                "judge_reasoning": f"judge call failed ({exc.__class__.__name__}), defaulting to answerable",
                "judge_call_failed": True,
            }

    def respond_node(state: AgenticState) -> dict:
        if state["judgment"] == "not_answerable":
            response = {
                "abstained": True,
                "hits": [],
                "message": "The corpus does not appear to contain an answer to this question.",
            }
        else:
            response = {"abstained": False, "hits": state["hits"], "message": None}
        return {"response": response}

    graph = StateGraph(AgenticState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("judge", judge_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "judge")
    graph.add_edge("judge", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def run_agentic_query(graph, query: str, route_enabled: bool = True) -> AgenticResult:
    final_state = graph.invoke({"query": query, "route_enabled": route_enabled})
    return AgenticResult(
        query=query,
        route=final_state["route"],
        route_reasoning=final_state["route_reasoning"],
        hits=final_state["hits"],
        judgment=final_state["judgment"],
        judge_reasoning=final_state["judge_reasoning"],
        abstained=final_state["response"]["abstained"],
        message=final_state["response"]["message"],
        route_call_failed=final_state.get("route_call_failed", False),
        judge_call_failed=final_state.get("judge_call_failed", False),
    )


def get_structured_llms(config: AgenticConfig):
    """Production construction path: a real ChatOpenAI client (via the
    existing adapters/llm.py seam) wrapped with with_structured_output
    for each schema. langchain_openai is not a base dependency of this
    repo (see requirements.txt) -- same deferred-import convention as
    adapters/llm.py's own openai branch, only exercised if you actually
    pick this backend.
    """
    from adapters.llm import get_llm

    base_llm = get_llm(config.llm)
    return base_llm.with_structured_output(RouteDecision), base_llm.with_structured_output(JudgeDecision)
