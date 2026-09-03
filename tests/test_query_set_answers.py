from eval.query_set import add_reference_answers


def test_add_reference_answers_uses_faq_row_answer():
    queries = [{
        "id": "faq",
        "text": "Question",
        "expected_source": "faq",
        "ground_truth": [{"source": "faq", "faq_index": 0}],
    }]

    enriched = add_reference_answers(queries)

    assert enriched[0]["reference_answer"]
    assert "reference_answer" not in queries[0]


def test_add_reference_answers_marks_neither_as_unanswerable():
    queries = [{
        "id": "neither",
        "text": "Unknown",
        "expected_source": "neither",
        "ground_truth": None,
    }]

    assert add_reference_answers(queries)[0]["reference_answer"] == (
        "The corpus does not contain an answer."
    )
