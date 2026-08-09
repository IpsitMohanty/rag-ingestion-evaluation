"""Stage 2 amendment (docs/phase6_preregistration.md #5a, dated
2026-08-09): #5's verbatim-question-text labeling measured 0/92 positive
-- self-retrieval on a 126-document corpus, not a difficulty test,
confirmed structural by a same-session diagnostic. This script generates
one LLM paraphrase per unused FAQ row, checks the resulting lexical
overlap against the pre-registered gate, and -- only if the gate passes
-- immediately re-runs the mechanical hit@k relabeling using the
paraphrased text in place of the verbatim question.

    OPENAI_API_KEY=... python -m training.generate_faq_paraphrases

Real (non-cached) OpenAI calls: gpt-4o-mini, temperature=0.0, seed=42,
same convention as every other LLM call in this repo. Approved estimate:
92 calls (one per unused FAQ row), hard stop 138 (1.5x, matching
eval/agentic_sweep.py's own convention). Cached via eval.llm_cache the
same way phase 4/5's real calls are, so a second run of this script
(e.g. after a code fix) does not re-spend.

Gate (docs/phase6_preregistration.md #5a, stated there BEFORE this script
ever ran, not adjusted after seeing output):
  1. mean Jaccard overlap in [0.05, 0.25]
  2. <=14 of 92 rows (~15%) exceed 0.5 overlap
  3. resulting relabeling yields >=10 positive (needs_correction=1) rows
If any check fails, this script stops (non-zero exit) and does NOT write
faq_expansion.jsonl -- the amendment says the next step on a gate failure
is hand-authored paraphrasing, reviewed before proceeding, not a second
automated attempt with adjusted prompt wording.
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

from pydantic import BaseModel, Field  # noqa: E402

from adapters import embeddings as embeddings_adapter  # noqa: E402
from adapters.llm import get_llm  # noqa: E402
from config import DEFAULT_CONFIG, LLMConfig  # noqa: E402

from eval import agentic_sweep as sweep  # noqa: E402
from eval import metrics  # noqa: E402
from eval.llm_cache import CachedStructuredLLM, load_cache, save_cache  # noqa: E402
from eval.query_set import load_query_set  # noqa: E402

from training.extract_critic_data import (  # noqa: E402
    FAQ_JSON_PATH, OUT_DIR, SEED, VAL_FRACTION, build_examples, stratified_split, unused_faq_indices,
)

# Not part of src/config.py's Config/DEFAULT_CONFIG: this is offline
# training-data tooling, not a pipeline component the app or eval harness
# ever constructs at runtime. Same deterministic convention as
# AgenticConfig/CorrectiveAgenticConfig regardless.
PARAPHRASE_LLM_CONFIG = LLMConfig(backend="openai", model_name="gpt-4o-mini", temperature=0.0, seed=42)

APPROVED_CALL_ESTIMATE = 92  # one call per unused FAQ row
HARD_STOP_CALLS = int(APPROVED_CALL_ESTIMATE * 1.5)

# Gate, #5a -- stated before this script ran, not adjusted after seeing output.
JACCARD_MEAN_RANGE = (0.05, 0.25)
MAX_HIGH_OVERLAP_FRACTION_COUNT = 14  # of 92, ~15%, hand set's 3/26 rounded up
HIGH_OVERLAP_THRESHOLD = 0.5
MIN_POSITIVE_EXAMPLES = 10

PARAPHRASE_PROMPT = """Reword the FAQ question below so it asks the exact \
same thing, but using different vocabulary and phrasing -- substitute \
synonyms, restructure the sentence, describe a named feature/concept \
instead of naming it outright where that still reads naturally. Do not \
change what is being asked, add information, or remove information. Do \
not just add a prefix or suffix to the original wording -- substantively \
rephrase it, the way someone unfamiliar with this app's own terminology \
might ask the same question.

Original question: {question}
"""


class FaqParaphrase(BaseModel):
    paraphrased_question: str = Field(
        description="The same question, reworded with different vocabulary/phrasing, meaning unchanged"
    )
    reasoning: str = Field(description="One sentence explaining what was changed")


def generate_paraphrases(unused_indices: list[int], faq_rows: list[dict]) -> tuple[dict[int, dict], int]:
    """Returns (idx -> {paraphrase, reasoning, jaccard_overlap}, real_call_count)."""
    base_llm = get_llm(PARAPHRASE_LLM_CONFIG).with_structured_output(FaqParaphrase)
    cache = load_cache()
    llm = CachedStructuredLLM(base_llm, FaqParaphrase, "faq_paraphrase", PARAPHRASE_LLM_CONFIG.model_name, cache)

    results = {}
    for idx in unused_indices:
        question = faq_rows[idx]["question"]
        decision = llm.invoke(PARAPHRASE_PROMPT.format(question=question))
        overlap = metrics.jaccard_overlap(decision.paraphrased_question, question)
        results[idx] = {
            "paraphrase": decision.paraphrased_question,
            "reasoning": decision.reasoning,
            "jaccard_overlap": overlap,
        }

    if llm.call_count > HARD_STOP_CALLS:
        raise RuntimeError(
            f"{llm.call_count} real LLM calls made, past the hard stop of {HARD_STOP_CALLS} "
            f"(1.5x the approved ~{APPROVED_CALL_ESTIMATE}-call estimate). Stopping before saving the cache."
        )
    save_cache(cache)
    return results, llm.call_count


def check_gate(paraphrases: dict[int, dict], n_total: int) -> tuple[bool, dict]:
    overlaps = [v["jaccard_overlap"] for v in paraphrases.values()]
    mean_overlap = sum(overlaps) / len(overlaps)
    n_high = sum(1 for o in overlaps if o > HIGH_OVERLAP_THRESHOLD)
    mean_ok = JACCARD_MEAN_RANGE[0] <= mean_overlap <= JACCARD_MEAN_RANGE[1]
    high_ok = n_high <= MAX_HIGH_OVERLAP_FRACTION_COUNT
    report = {
        "n": n_total,
        "mean_jaccard_overlap": mean_overlap,
        "mean_in_range": mean_ok,
        "range_checked": list(JACCARD_MEAN_RANGE),
        "n_high_overlap_gt_0.5": n_high,
        "high_overlap_count_ok": high_ok,
        "high_overlap_max_allowed": MAX_HIGH_OVERLAP_FRACTION_COUNT,
        "hand_authored_26_reference": {"mean": 0.134, "median": 0.101, "high_overlap_count": 3, "n": 26},
    }
    return (mean_ok and high_ok), report


def main() -> None:
    queries = load_query_set()
    faq_rows = json.loads(FAQ_JSON_PATH.read_text(encoding="utf-8"))
    unused = unused_faq_indices(queries, len(faq_rows))
    print(f"Generating paraphrases for {len(unused)} unused FAQ rows "
          f"(approved estimate {APPROVED_CALL_ESTIMATE} calls, hard stop {HARD_STOP_CALLS}).")

    paraphrases, real_calls = generate_paraphrases(unused, faq_rows)
    print(f"Real (non-cached) LLM calls made: {real_calls}")

    passed, gate_report = check_gate(paraphrases, len(unused))
    print(f"\nJaccard-overlap gate: mean={gate_report['mean_jaccard_overlap']:.3f} "
          f"(range {gate_report['range_checked']}, ok={gate_report['mean_in_range']}), "
          f"high-overlap(>0.5) count={gate_report['n_high_overlap_gt_0.5']} "
          f"(max {gate_report['high_overlap_max_allowed']}, ok={gate_report['high_overlap_count_ok']})")

    paraphrases_path = OUT_DIR / "faq_paraphrases.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with paraphrases_path.open("w", encoding="utf-8") as f:
        for idx in unused:
            row = {"faq_index": idx, "original_question": faq_rows[idx]["question"], **paraphrases[idx]}
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {paraphrases_path} ({len(unused)} rows)")

    if not passed:
        gate_report["overlap_gate_passed"] = False
        gate_report["stopped_before_relabeling"] = True
        (OUT_DIR / "faq_paraphrases_summary.json").write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
        print(
            "\nGATE FAILED: overlap distribution did not land in the pre-registered range. "
            "Per docs/phase6_preregistration.md #5a: stopping here, NOT re-running with adjusted "
            "prompt wording, and NOT relabeling. Next step is hand-authored paraphrasing, reviewed "
            "before proceeding."
        )
        sys.exit(1)

    print("\nOverlap gate passed. Re-running mechanical hit@k relabeling with paraphrased query text...")
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)
    text_overrides = {idx: paraphrases[idx]["paraphrase"] for idx in unused}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        faq_store, policy_index = sweep.build_representative_cell(embeddings, Path(tmp))
        examples = build_examples(unused, faq_rows, faq_store, policy_index, text_overrides=text_overrides)

    n_pos = sum(ex["needs_correction"] for ex in examples)
    n_neg = len(examples) - n_pos
    gate_report["overlap_gate_passed"] = True
    gate_report["measured_positive_count"] = n_pos
    gate_report["measured_negative_count"] = n_neg
    gate_report["min_positive_required"] = MIN_POSITIVE_EXAMPLES
    positive_gate_passed = n_pos >= MIN_POSITIVE_EXAMPLES
    gate_report["positive_count_gate_passed"] = positive_gate_passed
    (OUT_DIR / "faq_paraphrases_summary.json").write_text(json.dumps(gate_report, indent=2), encoding="utf-8")

    print(f"Measured class balance after paraphrase-based relabeling: {n_pos} positive / {n_neg} negative "
          f"out of {len(examples)} (gate: >={MIN_POSITIVE_EXAMPLES} required, "
          f"{'PASSED' if positive_gate_passed else 'FAILED'}).")

    if not positive_gate_passed:
        print(
            "\nGATE FAILED: overlap distribution passed but relabeling still yielded too few positive "
            "examples. Per docs/phase6_preregistration.md #5a: stopping, NOT writing faq_expansion.jsonl, "
            "NOT proceeding to CE/LoRA training. Next step is hand-authored paraphrasing, reviewed before "
            "proceeding."
        )
        sys.exit(1)

    stratified_split(examples, SEED, VAL_FRACTION)
    n_val = sum(1 for ex in examples if ex["split"] == "val")
    n_val_pos = sum(1 for ex in examples if ex["split"] == "val" and ex["needs_correction"] == 1)
    n_train = len(examples) - n_val

    expansion_path = OUT_DIR / "faq_expansion.jsonl"
    with expansion_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    summary = {
        "source": "92 unused FAQ rows, LLM-paraphrased query text (docs/phase6_preregistration.md #5a, "
                  "amending #5's verbatim-text approach, which measured 0/92 positive)",
        "cell_id": sweep.CELL_ID, "k": sweep.K, "seed": SEED, "val_fraction": VAL_FRACTION,
        "total_examples": len(examples), "positive_needs_correction_1": n_pos, "negative_needs_correction_0": n_neg,
        "positive_rate": n_pos / len(examples) if examples else None,
        "train_count": n_train, "val_count": n_val, "val_positive_count": n_val_pos,
        "paraphrase_gate": gate_report,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {expansion_path} ({len(examples)} rows) and {OUT_DIR / 'summary.json'}")
    print(f"Split: {n_train} train / {n_val} val ({n_val_pos} positive in val).")
    print(
        "\nStage 2 (amended) checkpoint: paraphrase generation + relabeling complete, both gates passed. "
        "Pausing here for review before Stage 3 (CE/LoRA training) -- not an auto-proceed."
    )


if __name__ == "__main__":
    main()
