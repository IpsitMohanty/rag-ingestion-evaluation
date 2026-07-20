"""Load and validate eval/queries.yaml against the schema documented at
the top of that file. Raises on malformed entries rather than skipping
them -- a bad ground-truth ref should fail the run, not silently drop a
query from the sweep.
"""
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
QUERY_SET_PATH = REPO_ROOT / "eval" / "queries.yaml"
FAQ_JSON_PATH = REPO_ROOT / "data" / "faq" / "all_faq.json"

VALID_EXPECTED_SOURCES = {"faq", "policy_pdf", "either", "neither"}
VALID_ORIGINS = {"mowcd_page_given", "policy_derived", "faq_reworded", "neither_constructed"}
POLICY_PDF_PAGE_COUNT = 77


def _validate_entry(q: dict[str, Any], faq_row_count: int) -> None:
    missing = {"id", "text", "expected_source", "origin"} - set(q)
    if missing:
        raise ValueError(f"query {q.get('id', '?')!r} missing required field(s): {missing}")

    if q["expected_source"] not in VALID_EXPECTED_SOURCES:
        raise ValueError(f"query {q['id']!r} has invalid expected_source {q['expected_source']!r}")
    if q["origin"] not in VALID_ORIGINS:
        raise ValueError(f"query {q['id']!r} has invalid origin {q['origin']!r}")

    ground_truth = q.get("ground_truth")
    if q["expected_source"] == "neither":
        if ground_truth:
            raise ValueError(f"query {q['id']!r} is 'neither' but has a non-null ground_truth")
    else:
        if not ground_truth:
            raise ValueError(f"query {q['id']!r} is {q['expected_source']!r} but has no ground_truth")
        for ref in ground_truth:
            if ref["source"] == "faq":
                if not (0 <= ref["faq_index"] < faq_row_count):
                    raise ValueError(f"query {q['id']!r} faq_index {ref['faq_index']} out of range")
            elif ref["source"] == "policy_pdf":
                if not (1 <= ref["page"] <= POLICY_PDF_PAGE_COUNT):
                    raise ValueError(f"query {q['id']!r} page {ref['page']} out of range")
            else:
                raise ValueError(f"query {q['id']!r} has unknown ground_truth source {ref['source']!r}")

    if q["origin"] == "faq_reworded" and "original_question" not in q:
        raise ValueError(f"query {q['id']!r} is faq_reworded but has no original_question")


def load_query_set(path: Path = QUERY_SET_PATH, faq_json_path: Path = FAQ_JSON_PATH) -> list[dict]:
    import json

    faq_rows = json.loads(faq_json_path.read_text(encoding="utf-8"))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries = data["queries"]

    ids = [q["id"] for q in queries]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate query ids: {duplicates}")

    for q in queries:
        _validate_entry(q, faq_row_count=len(faq_rows))

    return queries


def by_bucket(queries: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"faq": [], "policy_pdf": [], "either": [], "neither": []}
    for q in queries:
        buckets[q["expected_source"]].append(q)
    return buckets
