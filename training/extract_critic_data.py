"""Stage 2 (docs/phase6_preregistration.md #5): builds the mechanically
auto-labeled critic training pool from the 92 FAQ rows never used as
ground truth by any of the 51 phase-4/5 eval queries.

    python -m training.extract_critic_data

Two phases, in order, matching the pre-registration:

1. **Retriever spot-check (#4's retriever-pinning requirement)**, run
   BEFORE any labeling. Reuses eval.agentic_sweep.build_representative_cell
   and eval.agentic_sweep._verify_frozen_ground_truth verbatim -- the same
   drift check eval/run_corrective_eval.py already runs before phase 5's
   real calls -- across all 51 eval queries, plus an explicit printed
   spot-check on one named eval-gold query (faq-05, a RESCUE_TARGET_ID
   whose original-phrasing single-shot retrieval is frozen as a MISS) so
   the check is legible as "this one query" evidence, not just a pass/fail
   count. If this fails, the corpus/embeddings/chunking has drifted since
   phase 5's labels were frozen, and labeling must not proceed on a
   retriever that no longer matches what the critic will be graded
   against at eval time.

2. **Mechanical labeling.** For each of the 92 unused FAQ rows: the row's
   own question text (verbatim) as the query, retrieved via
   eval.retrievers.query_combined at the pinned cell/k (the same
   single-shot, no-routing retrieval SHOULD_ABSTAIN_IDS was itself frozen
   against -- see eval/agentic_sweep.py's run_corrective_eval.py usage),
   labeled via eval.metrics.hit_at_k against that row's own faq_index.
   Zero LLM calls, zero hand-labeling, zero eval-set overlap by
   construction (the 92 rows are exactly the complement of the 34
   faq_index values any of the 51 eval queries reference as ground_truth).

Writes data/critic_training/faq_expansion.jsonl (one row per example,
`split` field "train"/"val") and data/critic_training/summary.json
(counts, class balance, seed, retriever config -- the numbers the
pre-registration's Stage 2 checkpoint is reviewed against). Deliberately
does NOT train anything -- CE/LoRA training are later, separately
reviewed stages.
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

from eval import agentic_sweep as sweep  # noqa: E402
from eval import metrics  # noqa: E402
from eval.query_set import load_query_set  # noqa: E402
from eval.retrievers import query_combined  # noqa: E402

FAQ_JSON_PATH = REPO_ROOT / "data" / "faq" / "all_faq.json"
OUT_DIR = REPO_ROOT / "data" / "critic_training"

SEED = 42  # same convention as CorrectiveAgenticConfig/AgenticConfig's fixed seed=42
VAL_FRACTION = 0.2
SPOT_CHECK_ID = "faq-05"  # a RESCUE_TARGET_ID: frozen as a single-shot MISS on its original phrasing


def spot_check_retriever(queries: list[dict], faq_store, policy_index) -> dict[str, list]:
    """Phase #4's retriever-pinning requirement, checked before any
    labeling happens. Returns baseline_hits so the caller isn't forced to
    re-retrieve for the printed spot-check line.
    """
    baseline_hits = {q["id"]: query_combined(q["text"], sweep.K, faq_store, policy_index) for q in queries}

    # Full 51-query drift check -- reused verbatim, not reimplemented.
    sweep._verify_frozen_ground_truth(queries, baseline_hits)

    # One named example, printed, so the check reads as "this query, this
    # outcome" evidence rather than only an aggregate pass/fail.
    spot_query = next(q for q in queries if q["id"] == SPOT_CHECK_ID)
    spot_hits = baseline_hits[SPOT_CHECK_ID]
    hit = metrics.hit_at_k(spot_hits, spot_query["ground_truth"])
    expected_should_abstain = SPOT_CHECK_ID in sweep.SHOULD_ABSTAIN_IDS
    print(
        f"Spot-check ({SPOT_CHECK_ID!r}, cell={sweep.CELL_ID!r}, k={sweep.K}): "
        f"hit_at_k={hit} (expected miss, since {SPOT_CHECK_ID!r} in SHOULD_ABSTAIN_IDS="
        f"{expected_should_abstain}) -- {'OK, reproduces phase 5' if hit is False else 'MISMATCH'}"
    )
    if hit is not False:
        raise RuntimeError(
            f"Retriever spot-check failed: {SPOT_CHECK_ID!r} was frozen (eval/METHODOLOGY.md #11, "
            f"eval.agentic_sweep.SHOULD_ABSTAIN_IDS) as a single-shot retrieval MISS, but the "
            f"representative cell just retrieved a hit for it. Do not proceed to labeling on a "
            f"retriever that no longer reproduces phase 5's own retrieval -- investigate drift first."
        )
    print(
        f"Full-set drift check passed: all 51 eval queries' single-shot retrieval still matches "
        f"the frozen SHOULD_ABSTAIN_IDS ground truth (eval.agentic_sweep._verify_frozen_ground_truth)."
    )
    return baseline_hits


def unused_faq_indices(queries: list[dict], faq_row_count: int) -> list[int]:
    used = {
        ref["faq_index"]
        for q in queries
        for ref in (q.get("ground_truth") or [])
        if ref["source"] == "faq"
    }
    unused = sorted(set(range(faq_row_count)) - used)
    return unused


def build_examples(
    unused_indices: list[int], faq_rows: list[dict], faq_store, policy_index,
    text_overrides: dict[int, str] | None = None,
) -> list[dict]:
    """text_overrides: idx -> query text, used in place of the row's own
    verbatim question when present (docs/phase6_preregistration.md #5a:
    verbatim question/answer text both measured 0/92 positive -- self-
    retrieval, not a difficulty test, on this 126-doc corpus). Default
    (no override) reproduces #5's original, now-documented-negative
    verbatim-text behavior unchanged -- kept, not deleted, since it's the
    finding #5a records.
    """
    examples = []
    for idx in unused_indices:
        row = faq_rows[idx]
        text = (text_overrides or {}).get(idx, row["question"])
        ground_truth = [{"source": "faq", "faq_index": idx}]
        hits = query_combined(text, sweep.K, faq_store, policy_index)
        sufficient = metrics.hit_at_k(hits, ground_truth)
        examples.append({
            "id": f"faq-expand-{idx:03d}",
            "faq_index": idx,
            "text": text,
            "original_question": row["question"] if text_overrides else None,
            "expected_source": "faq",
            "ground_truth": ground_truth,
            "cell_id": sweep.CELL_ID,
            "k": sweep.K,
            "needs_correction": 0 if sufficient else 1,
            "retrieved": [
                {
                    "source_type": h.source_type,
                    "score": h.score,
                    "faq_index": h.faq_index,
                    "page": h.page,
                    "snippet": h.document.page_content[:200],
                }
                for h in hits
            ],
        })
    return examples


def stratified_split(examples: list[dict], seed: int, val_fraction: float) -> None:
    """Assigns a `split` field in place ("train"/"val"), stratified by
    needs_correction so validation isn't left with zero positives -- the
    plain-random-split risk docs/phase6_preregistration.md #5 flags at
    this sample size.
    """
    import random

    rng = random.Random(seed)
    for label in (0, 1):
        bucket = [ex for ex in examples if ex["needs_correction"] == label]
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * val_fraction)) if bucket else 0
        val_ids = {ex["id"] for ex in bucket[:n_val]}
        for ex in examples:
            if ex["id"] in val_ids:
                ex["split"] = "val"
        for ex in bucket[n_val:]:
            ex["split"] = "train"


def main() -> None:
    queries = load_query_set()
    faq_rows = json.loads(FAQ_JSON_PATH.read_text(encoding="utf-8"))
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        faq_store, policy_index = sweep.build_representative_cell(embeddings, Path(tmp))

        spot_check_retriever(queries, faq_store, policy_index)

        unused = unused_faq_indices(queries, len(faq_rows))
        print(f"\n{len(faq_rows)} FAQ rows total, {len(faq_rows) - len(unused)} used as eval-query "
              f"ground truth, {len(unused)} unused (available for labeling).")

        examples = build_examples(unused, faq_rows, faq_store, policy_index)

    n_pos = sum(ex["needs_correction"] for ex in examples)
    n_neg = len(examples) - n_pos
    stratified_split(examples, SEED, VAL_FRACTION)
    n_val_pos = sum(1 for ex in examples if ex["split"] == "val" and ex["needs_correction"] == 1)
    n_val = sum(1 for ex in examples if ex["split"] == "val")
    n_train = len(examples) - n_val

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "faq_expansion.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    summary = {
        "source": "92 unused FAQ rows (data/faq/all_faq.json), verbatim question text, "
                  "mechanical hit_at_k labeling -- docs/phase6_preregistration.md #5",
        "cell_id": sweep.CELL_ID,
        "k": sweep.K,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "total_examples": len(examples),
        "positive_needs_correction_1": n_pos,
        "negative_needs_correction_0": n_neg,
        "positive_rate": n_pos / len(examples) if examples else None,
        "train_count": n_train,
        "val_count": n_val,
        "val_positive_count": n_val_pos,
        "preregistered_estimate": {
            "positive": 16, "negative": 76,
            "note": "docs/phase6_preregistration.md #5 -- projected from phase 2's hit@5=0.826, "
                    "not measured; this summary's positive_needs_correction_1 is the measured figure.",
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {jsonl_path} ({len(examples)} rows) and {summary_path}")
    print(f"Measured class balance: {n_pos} positive (needs_correction=1) / {n_neg} negative "
          f"out of {len(examples)} -- pre-registered estimate was ~16 / ~76.")
    print(f"Split: {n_train} train / {n_val} val ({n_val_pos} positive in val).")
    print("\nStage 2 checkpoint: labeling complete. Per instructions, pausing here for review "
          "before any CE/LoRA training -- not an auto-proceed.")


if __name__ == "__main__":
    main()
