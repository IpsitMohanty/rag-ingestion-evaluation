"""Coverage: eval/queries.yaml schema validation (eval/query_set.py)."""
import pytest

from eval.query_set import QUERY_SET_PATH, by_bucket, load_query_set

BASE_ENTRY = {
    "id": "q-test",
    "text": "some query",
    "expected_source": "faq",
    "origin": "faq_reworded",
    "ground_truth": [{"source": "faq", "faq_index": 0}],
    "original_question": "the original",
}


def test_real_query_set_loads_and_validates():
    queries = load_query_set()
    assert len(queries) == 51


def test_real_query_set_bucket_counts():
    buckets = by_bucket(load_query_set())
    assert len(buckets["faq"]) == 26
    assert len(buckets["policy_pdf"]) == 11
    assert len(buckets["either"]) == 9
    assert len(buckets["neither"]) == 5


def test_real_query_set_ids_are_unique():
    queries = load_query_set()
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))


def _load_with(entries, tmp_path):
    import yaml

    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump({"queries": entries}), encoding="utf-8")
    return path


def test_missing_required_field_raises(tmp_path):
    bad = dict(BASE_ENTRY)
    del bad["origin"]
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="missing required field"):
        load_query_set(path)


def test_invalid_expected_source_raises(tmp_path):
    bad = {**BASE_ENTRY, "expected_source": "not-a-real-bucket"}
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="invalid expected_source"):
        load_query_set(path)


def test_invalid_origin_raises(tmp_path):
    bad = {**BASE_ENTRY, "origin": "not-a-real-origin"}
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="invalid origin"):
        load_query_set(path)


def test_neither_with_ground_truth_raises(tmp_path):
    bad = {**BASE_ENTRY, "expected_source": "neither", "ground_truth": [{"source": "faq", "faq_index": 0}]}
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="non-null ground_truth"):
        load_query_set(path)


def test_non_neither_without_ground_truth_raises(tmp_path):
    bad = {**BASE_ENTRY, "ground_truth": None}
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="no ground_truth"):
        load_query_set(path)


def test_out_of_range_faq_index_raises(tmp_path):
    bad = {**BASE_ENTRY, "ground_truth": [{"source": "faq", "faq_index": 99999}]}
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        load_query_set(path)


def test_out_of_range_page_raises(tmp_path):
    bad = {
        **BASE_ENTRY, "expected_source": "policy_pdf",
        "ground_truth": [{"source": "policy_pdf", "page": 9999}],
    }
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        load_query_set(path)


def test_faq_reworded_without_original_question_raises(tmp_path):
    bad = dict(BASE_ENTRY)
    del bad["original_question"]
    path = _load_with([bad], tmp_path)
    with pytest.raises(ValueError, match="original_question"):
        load_query_set(path)


def test_duplicate_ids_raise(tmp_path):
    path = _load_with([BASE_ENTRY, dict(BASE_ENTRY)], tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        load_query_set(path)


def test_query_set_path_points_at_real_file():
    assert QUERY_SET_PATH.exists()
