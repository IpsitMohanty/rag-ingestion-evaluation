from eval.report_answer_quality import render_report


def test_render_report_contains_comparable_summary_and_failures():
    report = render_report({
        "baseline": {
            "summary": {
                "correctness_rate": 0.5,
                "faithfulness_rate": 1.0,
                "abstention_accuracy": 0.75,
                "mean_latency_seconds": 1.25,
                "answered": 3,
                "n": 4,
            },
            "results": [
                {"query_id": "q1", "abstained": False, "correct": False},
                {"query_id": "q2", "abstained": True, "correct": None},
            ],
        },
    })

    assert "| baseline | 50.0% | 100.0% | 75.0% | 1.25s | 3/4 |" in report
    assert "**baseline:** q1" in report
    assert "**baseline:** q2" in report
