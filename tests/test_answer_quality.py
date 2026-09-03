from types import SimpleNamespace

import pytest

from eval.answer_quality import evaluate_one, evaluate_variants, summarize


class _GradeLLM:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.result


def test_evaluate_one_grades_answer_and_excerpts():
    correctness = SimpleNamespace(correct=True, reasoning="Matches the reference.")
    faithfulness = SimpleNamespace(grounded=True, reasoning="Claims are supported.")
    result = evaluate_one(
        {
            "id": "faq-01",
            "text": "How do I register?",
            "reference_answer": "Register through the application.",
        },
        "corrective",
        lambda _: {
            "abstained": False,
            "answer": "Use the application to register.",
            "hits": [{"content": "Registration is completed in the application."}],
            "token_usage": {"input": 10, "output": 5},
        },
        _GradeLLM(correctness),
        _GradeLLM(faithfulness),
    )

    assert result.correct is True
    assert result.grounded is True
    assert result.token_usage == {"input": 10, "output": 5}
    assert result.elapsed_seconds >= 0


def test_abstained_answers_are_not_graded():
    correctness = _GradeLLM(SimpleNamespace(correct=True, reasoning="unused"))
    faithfulness = _GradeLLM(SimpleNamespace(grounded=True, reasoning="unused"))
    result = evaluate_one(
        {"id": "neither-01", "text": "Unknown question"},
        "baseline",
        lambda _: {"abstained": True, "answer": None, "hits": []},
        correctness,
        faithfulness,
    )

    assert result.correct is None
    assert result.grounded is None
    assert correctness.prompts == []
    assert faithfulness.prompts == []


def test_summarize_reports_answered_and_quality_rates():
    results = [
        SimpleNamespace(
            variant="baseline", abstained=False, correct=True, grounded=True,
            elapsed_seconds=1.0, token_usage={"input": 2, "output": 3},
            metadata={"abstention_correct": True},
        ),
        SimpleNamespace(
            variant="baseline", abstained=True, correct=None, grounded=None,
            elapsed_seconds=3.0, token_usage=None,
            metadata={"abstention_correct": True},
        ),
    ]

    summary = summarize(results)

    assert summary.n == 2
    assert summary.answered == 1
    assert summary.abstained == 1
    assert summary.correctness_rate == 1.0
    assert summary.faithfulness_rate == 1.0
    assert summary.abstention_accuracy == 1.0
    assert summary.mean_latency_seconds == 2.0
    assert summary.total_tokens == 5


def test_summarize_rejects_mixed_variants():
    with pytest.raises(ValueError, match="exactly one variant"):
        summarize([
            SimpleNamespace(variant="baseline", elapsed_seconds=0),
            SimpleNamespace(variant="corrective", elapsed_seconds=0),
        ])


def test_evaluate_variants_preserves_comparable_variant_results():
    grade = _GradeLLM(SimpleNamespace(correct=True, grounded=True, reasoning="ok"))
    evaluated = evaluate_variants(
        [{"id": "faq-01", "text": "Question", "expected_source": "faq", "reference_answer": "Answer"}],
        {
            "baseline": lambda _: {"answer": "Answer", "abstained": False, "hits": []},
            "corrective": lambda _: {"answer": "Answer", "abstained": False, "hits": []},
        },
        grade,
        grade,
    )

    assert set(evaluated) == {"baseline", "corrective"}
    assert evaluated["baseline"]["summary"].variant == "baseline"


def test_correctness_prompt_allows_equivalent_units():
    from eval.answer_quality import CORRECTNESS_PROMPT

    assert "equivalent units" in CORRECTNESS_PROMPT
