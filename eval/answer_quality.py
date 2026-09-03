"""Answer-quality evaluation for baseline and agentic RAG runs.

The evaluator deliberately does not construct a retrieval graph. Each pipeline
variant supplies a runner returning an answer, retrieved excerpts, and an
abstention decision. This keeps the measurement layer comparable across the
baseline, corrective, and multi-agent implementations.
"""
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable

from pydantic import BaseModel, Field


class CorrectnessGrade(BaseModel):
    correct: bool = Field(description="Whether the answer resolves the question correctly")
    reasoning: str = Field(description="One sentence explaining the decision")


class FaithfulnessGrade(BaseModel):
    grounded: bool = Field(description="Whether every factual claim is supported by the excerpts")
    reasoning: str = Field(description="One sentence explaining the decision")


CORRECTNESS_PROMPT = """Evaluate whether the candidate answer correctly answers the question.
Use the reference answer as the expected content, but allow concise paraphrases,
equivalent wording, and equivalent units (for example, calories and kcal).
Do not require wording or ordering to match. Mark incorrect if the answer
contradicts the reference, omits a central requested fact, or claims an answer
when the evidence is absent. Do not penalize a supported answer merely for
being shorter when it still resolves the question.

Question: {query}

Reference answer:
{reference_answer}

Candidate answer:
{answer}
"""

FAITHFULNESS_PROMPT = """Evaluate whether every factual claim in the candidate answer is
supported by the retrieved excerpts. Do not use outside knowledge. Mark grounded
only when all material claims are directly supported; an answer that says it cannot
answer is grounded if it makes no unsupported factual claims.

Question: {query}

Retrieved excerpts:
{excerpts}

Candidate answer:
{answer}
"""


@dataclass
class AnswerQualityResult:
    query_id: str
    query: str
    variant: str
    abstained: bool
    answer: str | None
    correct: bool | None
    grounded: bool | None
    correctness_reasoning: str | None
    faithfulness_reasoning: str | None
    elapsed_seconds: float
    token_usage: dict[str, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerQualitySummary:
    variant: str
    n: int
    answered: int
    abstained: int
    correctness_rate: float | None
    faithfulness_rate: float | None
    abstention_accuracy: float
    mean_latency_seconds: float
    total_tokens: int | None


def _format_excerpts(excerpts: Iterable[Any]) -> str:
    formatted = []
    for index, excerpt in enumerate(excerpts, start=1):
        if isinstance(excerpt, str):
            content = excerpt
        elif hasattr(excerpt, "document"):
            content = excerpt.document.page_content
        elif isinstance(excerpt, dict):
            content = excerpt.get("content") or excerpt.get("page_content") or ""
        else:
            content = getattr(excerpt, "page_content", str(excerpt))
        formatted.append(f"{index}. {content}")
    return "\n".join(formatted) or "(no excerpts retrieved)"


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def evaluate_one(
    query: dict[str, Any],
    variant: str,
    runner: Callable[[str], Any],
    correctness_llm: Any,
    faithfulness_llm: Any,
) -> AnswerQualityResult:
    """Run and grade one query.

    ``runner`` may return a dict or any result object exposing ``answer``,
    ``abstained``, and ``hits``. LLMs must expose ``invoke(prompt)`` and return
    the corresponding Pydantic grade.
    """
    started = time.perf_counter()
    output = runner(query["text"])
    elapsed = time.perf_counter() - started
    abstained = bool(_field(output, "abstained", False))
    answer = _field(output, "answer")
    hits = _field(output, "hits", [])

    correct = None
    grounded = None
    correctness_reasoning = None
    faithfulness_reasoning = None
    if not abstained and answer:
        correctness = correctness_llm.invoke(
            CORRECTNESS_PROMPT.format(
                query=query["text"],
                reference_answer=query.get("reference_answer", "(no reference answer supplied)"),
                answer=answer,
            )
        )
        faithfulness = faithfulness_llm.invoke(
            FAITHFULNESS_PROMPT.format(
                query=query["text"],
                excerpts=_format_excerpts(hits),
                answer=answer,
            )
        )
        correct = bool(correctness.correct)
        grounded = bool(faithfulness.grounded)
        correctness_reasoning = correctness.reasoning
        faithfulness_reasoning = faithfulness.reasoning

    return AnswerQualityResult(
        query_id=query["id"],
        query=query["text"],
        variant=variant,
        abstained=abstained,
        answer=answer,
        correct=correct,
        grounded=grounded,
        correctness_reasoning=correctness_reasoning,
        faithfulness_reasoning=faithfulness_reasoning,
        elapsed_seconds=elapsed,
        token_usage=_field(output, "token_usage"),
        metadata={
            "expected_source": query.get("expected_source"),
            "should_abstain": query.get("expected_source") == "neither",
            "abstention_correct": abstained == (query.get("expected_source") == "neither"),
        },
    )


def summarize(results: list[AnswerQualityResult]) -> AnswerQualitySummary:
    if not results:
        raise ValueError("Cannot summarize an empty answer-quality result set")
    variants = {result.variant for result in results}
    if len(variants) != 1:
        raise ValueError("summarize expects results from exactly one variant")
    answered = [result for result in results if not result.abstained]
    scored_correct = [result.correct for result in answered if result.correct is not None]
    scored_grounded = [result.grounded for result in answered if result.grounded is not None]
    token_values = [
        sum(result.token_usage.values())
        for result in results
        if result.token_usage is not None
    ]
    return AnswerQualitySummary(
        variant=results[0].variant,
        n=len(results),
        answered=len(answered),
        abstained=len(results) - len(answered),
        correctness_rate=(
            sum(scored_correct) / len(scored_correct) if scored_correct else None
        ),
        faithfulness_rate=(
            sum(scored_grounded) / len(scored_grounded) if scored_grounded else None
        ),
        abstention_accuracy=sum(
            result.metadata["abstention_correct"] for result in results
        ) / len(results),
        mean_latency_seconds=sum(r.elapsed_seconds for r in results) / len(results),
        total_tokens=sum(token_values) if token_values else None,
    )


def evaluate_variants(
    queries: list[dict[str, Any]],
    runners: dict[str, Callable[[str], Any]],
    correctness_llm: Any,
    faithfulness_llm: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate several pipeline runners against the same labeled queries.

    Runners are keyed by a stable variant name, such as ``baseline``,
    ``corrective``, or ``multiagent``. The returned objects contain both raw
    per-query results and an aggregate summary so they can be serialized by a
    benchmark command without losing audit detail.
    """
    if not runners:
        raise ValueError("At least one answer-quality runner is required")
    output: dict[str, dict[str, Any]] = {}
    for variant, runner in runners.items():
        results = [
            evaluate_one(query, variant, runner, correctness_llm, faithfulness_llm)
            for query in queries
        ]
        output[variant] = {
            "summary": summarize(results),
            "results": results,
        }
    return output
