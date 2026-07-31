# Phase 2 evaluation: analysis

Source data: `results/sweep_results.json`, 24 cells (2 PDF variants x 2
ingestion modes x 3 chunk configs x 2 retriever strategies), 51 queries
per cell, n=1 per cell (see `eval/METHODOLOGY.md` #7 for why no
repeats). Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
Scoring rules follow `eval/METHODOLOGY.md` exactly; this document
doesn't re-derive them.

Findings #1-#5 below were established on the original 12-cell grid
(raw PDF only) and re-checked against all 24 cells once the cleaned-PDF
variant was added (see "Cleaned vs raw PDF" further down). Each says
explicitly which cell count it was verified against.

Every number below is per-bucket. There is no pooled/overall metric
anywhere in this document, per METHODOLOGY.md #1.

## Finding #1: the system cannot tell "found it" from "found nothing"

Representative cell (`raw_pdf__format_aware__cs800_ov100__similarity`); **the
pattern below holds in all 24 cells of the sweep without exception,
raw and cleaned PDF variants alike**. This is the strongest, most
load-bearing finding in this evaluation, not a footnote to the chunking
result below.

| | n | mean distance | median | min | max |
|---|---|---|---|---|---|
| neither (5 queries, no correct passage exists) | 5 | 0.829 | 0.815 | 0.622 | 0.990 |
| should-hit (faq+policy_pdf+either, pooled, 46 queries) | 46 | 0.738 | 0.729 | 0.337 | 1.308 |

(Chroma distance: lower = more similar/more confident.)

The means point the right way: `neither` queries score less confident
on average than queries with a real answer. **But the ranges overlap in
every single cell of the sweep.** Some should-hit queries score worse
(max 1.308) than every `neither` query. Some `neither` queries score
better (min 0.622) than the majority of should-hit queries.

**Consequence: there is no similarity-score threshold that reliably
separates "the system found the answer" from "the system found the
least-bad thing available," in any configuration tested.** A caller
reading only the top-1 distance cannot build a working "I don't know"
gate out of it here. That's a common design assumption in shipped
RAG systems (retrieve, check a similarity threshold, abstain below it).
This corpus and this embedding model say that gate wouldn't work as
built. It reproduces identically in direction across all 24 cells (only
the exact mean/median shift slightly), so it isn't an artifact of one
unlucky chunk_size, ingestion mode, or PDF variant. See the
**Limitations** section below for the scope this claim is and isn't
making.

## Finding #2: format-aware ingestion beats uniform chunking, conditionally

**Rule of thumb, not a universal winner: format-aware ingestion matters
once chunk_size exceeds the corpus's smallest atomic unit (~178 chars,
the FAQ's average answer length here); below that, the two are
indistinguishable.** That conditional statement is what the data
supports, not "format-aware wins," full stop.

| chunk_size | format_aware hit@1 | uniform hit@1 | format_aware MRR | uniform MRR | mean FAQ rows merged per uniform chunk |
|---|---|---|---|---|---|
| 200 | 0.462 | 0.462 | 0.596 | 0.571 | 1.00 |
| 500 | 0.500 | 0.385 | 0.631 | 0.549 | 1.52 |
| 800 | 0.500 | 0.385 | 0.642 | 0.525 | 2.51 |

At chunk_size 200 (smaller than the FAQ's ~178-char average answer),
format-aware and uniform ingestion are statistically indistinguishable
(hit@1 identical, MRR within 0.025) because almost nothing actually gets
merged at that size (`mean_rows_per_uniform_chunk` = 1.00: uniform
chunking "does nothing" here, the brief's first predicted failure mode).
At chunk_size 500 and 800, uniform chunking measurably merges unrelated
Q&A pairs into single chunks (1.52 and 2.51 rows/chunk) and FAQ retrieval
degrades accordingly: hit@1 drops 12-22 points, MRR drops 8-12 points.
This is the brief's second predicted failure mode ("merges unrelated
Q&As... pollutes retrieval"), directly measured via
`mean_rows_per_uniform_chunk`, not merely inferred from the hit-rate gap.

The direction (format-aware >= uniform) holds at every chunk_size tested,
and the *size* of the gap tracks the *size* of the measured merging
almost monotonically: a consistent direction plus an independently
measured mechanism that predicts the gap's magnitude is what makes this a
real finding. But it is conditional on chunk_size, and should be reported
that way: **"format-aware ingestion helps once your chunk size exceeds
your smallest atomic unit" is the transferable takeaway, not "always
split FAQ-shaped content separately regardless of size."**

## How much of that FAQ number is semantic vs. lexical match?

(Same representative cell as above.)

| | n | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| high_lexical_overlap (named-feature questions: RCH Profile, Home Visit, Poshan Tracker Dashboard) | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| low_lexical_overlap (everything else) | 23 | 0.435 | 0.783 | **0.826** | 0.826 | 0.596 |
| whole FAQ bucket (all 26, quoted above) | 26 | 0.500 | 0.808 | 0.846 | 0.846 | 0.642 |

**The honest semantic-only curve, quoted throughout this document and the
README, is the full low-lexical-overlap row: 0.435 / 0.783 / 0.826 / 0.826
at k=1/3/5/10. Neither endpoint should be quoted alone.** hit@5=0.826 is
the operationally relevant figure (real systems retrieve k=5, not k=1),
and it's also where the curve stops improving: hit@10 is identical to
hit@5, not higher. See the next section for why.

This matters for two reasons. First, it's the honest denominator: the
whole-bucket curve (0.500/0.808/0.846/0.846) is inflated at every k by the
3 high-overlap queries (perfect score throughout, effectively a
string-match win, not a semantic one). **The low-overlap curve above is
the more honest estimate of this system's *semantic* retrieval quality on
paraphrased FAQ-style queries**, and it's the curve the baseline-arm
comparison below uses. Second, it sets up that comparison: is a system
that plateaus at 82.6% recall by k=5, and goes no further by k=10, actually
beating the alternative of just not retrieving at all?

The 3 high-overlap queries were kept rather than reworded further
because they're genuine named-UI-feature questions (see METHODOLOGY.md
#4): forcing them lower would trade clarity for a cosmetically lower
score, not a more honest one.

## The k=5-to-k=10 plateau: a representation ceiling, not a retrieval-depth problem

hit@10 equals hit@5 *exactly* on the low-lexical-overlap FAQ subset: 0.826
at both (19 of 23 queries hit at k=5; the same 19 at k=10). **4 of the 23
queries (17%) never surface their target chunk at any k tested, up to
10.** This holds in every cell of the sweep, not just the representative
one. It is not an artifact of this particular chunk_size/ingestion
combination.

**"Just retrieve more chunks" is the obvious response to a hit-rate gap,
and this data shows it doesn't work here.** If the miss were a
retrieval-depth problem (the right chunk sitting just outside the top
5), hit@10 would climb past hit@5 as more candidates get a chance to
include it. It doesn't, anywhere in the sweep. That points to a ceiling
in how `all-MiniLM-L6-v2` represents those specific paraphrase/answer
pairs in vector space: the target chunk's embedding is apparently never
among even the 10 closest to the query's embedding, for any of these 4
queries, at any configuration tested. It is a representation-quality
limit, not a search-depth limit.

**This compounds Finding #1 rather than sitting beside it.** Finding #1
says the system can't flag when it's found nothing good. This finding
says that raising k, the standard mitigation for "maybe it's just
outside the top few," doesn't rescue these particular misses either.
Together: for the queries this system gets wrong, it gets them wrong in
a way that neither a confidence threshold nor a deeper retrieval budget
catches or fixes.

## Baseline arm: cost and recall, no LLM

Built and run (not dropped): deterministic, no LLM, per the phase 2
brief and `eval/METHODOLOGY.md` design intent. The optional LLM-judged
answer-quality arm was **not** built, per the brief; that one really is
a documented seam, not this one.

**Whole-corpus size** (both sources, concatenated, as a context-stuffing
prompt would need): 31,778 chars (FAQ) + 128,803 chars (policy PDF, all
77 pages) = **160,581 chars ≈ 40,145 tokens** (approximate, using the
standard ~4-characters-per-token heuristic for English text: not a
precise tokenizer count, but the right order of magnitude). That fits
comfortably inside a modern 128K+-token context window, which is exactly
why this baseline is a real competitor here and not a straw man: this
corpus is small enough that "just stuff the whole thing in" is a
genuinely available option, not a hypothetical.

**Recall = 1.0 by construction for stuffing.** The whole corpus is in the
prompt; nothing relevant can be missed. State this plainly rather than
letting retrieval look like it's "competing" on recall. It isn't;
stuffing wins recall trivially, always, by definition.

**Index build cost**: retrieval requires building an embeddings index
before the first query; stuffing requires none. Measured on this
machine, this embedding model, the representative cell's corpus (126 FAQ
docs + 213 policy chunks, 339 total): **~17.2s** to embed and index
(4.3s FAQ + 12.9s policy), plus a ~1.1s one-time model load. Stuffing's
build cost is zero: the whole corpus is just concatenated text,
assembled at query time.

**Per-query token cost**, chunk_size=800 (phase 1's shipped default),
against the low-lexical-overlap FAQ recall curve from the section above
(the honest, semantic-only number, not the lexically-inflated whole-bucket one):

| | tokens/query (approx) | vs. whole-corpus stuffing | recall (low-overlap FAQ) |
|---|---|---|---|
| whole-corpus stuffing | ~40,145 | 1x (baseline) | 1.0, by construction |
| retrieval, k=1 | ~200 | 200x cheaper | 0.435 |
| retrieval, k=3 | ~600 | 67x cheaper | 0.783 |
| retrieval, k=5 | ~1,000 | **40x cheaper** | **0.826: the practical figure** |
| retrieval, k=10 | ~2,000 | 20x cheaper | 0.826: identical to k=5, see the plateau finding above |

**This is a real trade-off, not a foregone conclusion either way.**
Retrieval at k=5 is 40x cheaper per query than stuffing the whole corpus,
but tops out at 82.6% recall on paraphrased FAQ questions and never
reaches 100% even at k=10 in this dataset. Whether that trade is worth
it depends entirely on what a wrong or missing answer costs in the
deployment context, something outside the scope of this repo to
decide. What this repo can say: at this corpus's size (~40K tokens),
stuffing is not prohibitively expensive in absolute terms (well within
one modern context window, and a single query's stuffing cost is still
cheap in isolation), so the choice is a genuine cost/completeness
trade-off, not something forced by scale. It would stop being a genuine
choice well before this if the corpus were meaningfully larger, or if per
context window costs were the binding constraint rather than latency;
that boundary is not measured here.

Also worth reading alongside Finding #1: since similarity scores can't
reliably flag "the retriever didn't find anything good," a retrieval-only
system has no cheap way to know when it should have stuffed the whole
document instead. That's a second, independent argument for why this
trade-off doesn't resolve in retrieval's favor by default.

## Policy bucket: hit-rate with and without retrievable_but_incomplete

(Representative cell: `raw_pdf__format_aware__cs800_ov100__similarity`,
phase 1's shipped default: raw PDF variant; see "Cleaned vs raw PDF"
below for whether the cleaning axis changes this. All 6 raw-PDF
similarity-strategy cells show the same pattern in direction, if not
magnitude. See `sweep_results.json` for the rest.)

| | n | hit@1 | hit@10 | MRR |
|---|---|---|---|---|
| all 11 policy_pdf queries | 11 | 0.636 | 0.909 | 0.691 |
| excluding the 3 incomplete-flagged (pol-01, pol-05, pol-10) | 8 | 0.750 | 1.000 | 0.783 |

The gap is real and in the direction METHODOLOGY.md #2 predicted: the
raw, all-queries hit-rate is ~11-15 points lower than what the system
achieves on questions it can actually fully answer. Both numbers are
"correct": the higher one describes retrieval quality on answerable
questions, the lower one describes what a user actually experiences
across the full range of things they might plausibly ask. Reporting
only the higher one would have been the misleading choice.

Source-routing accuracy (top-1 source type correct; faq/policy_pdf only,
per METHODOLOGY.md #3) at this cell: **policy_pdf 0.909, faq 0.885**.
Both high and comparable, meaning the combined FAQ+policy retriever
rarely confuses which corpus a question belongs to, independent of
whether it finds the exact right passage within that corpus.

## Either bucket (cross-source queries)

Hit-rate under OR-across-sources scoring (METHODOLOGY.md #3) ranges
0.33-0.67 at k=1 and 0.89-1.00 at k=10 across all 24 cells (raw and
cleaned PDF included), with no consistent pattern tied to ingestion
mode, chunk size, or PDF variant: expected, since these 9 queries are
answerable from either sub-index. n=9 is too small to support a
finer-grained claim than "these queries are reliably findable by k=10
in every configuration tested." Excluded from source-routing accuracy
throughout, per METHODOLOGY.md #3.

## Retriever strategy: similarity vs ParentDocumentRetriever (policy bucket only, raw PDF)

| cell (ingestion, chunk_size) | similarity hit@1 | parent_document hit@1 | similarity hit@10 | parent_document hit@10 |
|---|---|---|---|---|
| format_aware, 200 | 0.455 | 0.455 | 0.818 | 0.727 |
| format_aware, 500 | 0.545 | 0.545 | 0.727 | 0.818 |
| format_aware, 800 | 0.636 | 0.636 | 0.909 | 0.909 |
| uniform, 200 | 0.455 | 0.455 | 0.818 | 0.727 |
| uniform, 500 | 0.545 | 0.545 | 0.727 | 0.818 |
| uniform, 800 | 0.636 | 0.636 | 0.909 | 0.909 |

**Cannot be distinguished at this scale.** hit@1 is identical between
the two strategies in all 6 comparisons; hit@10 favors whichever
strategy by a single query (1 of 11) in either direction depending on
chunk_size, with no consistent winner. With only 11 policy_pdf queries,
a single query flipping is enough to flip which strategy "wins". This
dataset cannot support a claim that either retrieval strategy is
better than the other. A larger, more diverse policy corpus (this PDF is
77 pages; ParentDocumentRetriever's rationale, large-document parent
context around a small matched child chunk, likely needs a longer or
more heterogeneous document than this one to show a real effect) would
be needed before concluding anything stronger.

## Cleaned vs raw PDF: does stripping headers and front matter help retrieval?

Prediction and decision rule were fixed in writing (`eval/METHODOLOGY.md`
#8a) before these numbers were looked at: **cleaning counts as a real
effect on the policy_pdf bucket only if hit@5 changes by more than 0.182
at the representative cell** (`format_aware`, chunk_size=800,
`similarity`): 0.182 being the largest hit@5 swing already observed
between configurations that have nothing to do with cleaning (chunk_size,
retriever strategy). The stated prediction was that cleaning would make
**no detectable difference** at that bar, since front matter/headers are
a small fraction of the corpus unlikely to sit inside top-5 for real
policy queries.

**Primary test, at the representative cell:**

| | n | hit@1 | hit@5 | hit@10 | MRR | source-routing acc. |
|---|---|---|---|---|---|---|
| raw | 11 | 0.636 | 0.727 | 0.909 | 0.691 | 0.909 |
| cleaned | 11 | 0.455 | **0.727** | 0.909 | 0.577 | 0.909 |
| diff (cleaned − raw) | | −0.182 | **0.000** | 0.000 | −0.114 | 0.000 |

**hit@5 diff is 0.000: does not clear the 0.182 bar. Result: cannot be
distinguished at this scale. The prediction was correct.**

hit@1 dropped by exactly one threshold-unit (0.182) at this cell, which
is worth being transparent about rather than only reporting the metric
that came out flat. But it's a rank-1-vs-rank-2/3 reshuffle, not a
recall loss (hit@5 and hit@10 both stayed at 0.727/0.909 identically).
The plausible mechanism: cleaning shifts the exact character offsets
`RecursiveCharacterTextSplitter` splits on (removed header text changes
where each page's content starts), which can change which chunk lands
in rank 1 without changing whether the right chunk is in the top 5:
offered as a plausible explanation, not a proven one.

**Robustness check, all 6 chunk_size x retriever_strategy pairs:**

| chunk | retriever | raw hit@1 | cln hit@1 | raw hit@5 | cln hit@5 | raw hit@10 | cln hit@10 |
|---|---|---|---|---|---|---|---|
| 200 | similarity | 0.455 | 0.273 | 0.727 | 0.909 | 0.818 | 0.909 |
| 200 | parent_document | 0.455 | 0.182 | 0.727 | 0.727 | 0.727 | 0.909 |
| 500 | similarity | 0.545 | 0.545 | 0.545 | 0.636 | 0.727 | 0.727 |
| 500 | parent_document | 0.545 | 0.545 | 0.727 | 0.727 | 0.818 | 0.818 |
| 800 | similarity | 0.636 | 0.455 | 0.727 | 0.727 | 0.909 | 0.909 |
| 800 | parent_document | 0.636 | 0.455 | 0.727 | 0.727 | 0.909 | 0.909 |

hit@5 never differs by more than 0.182 anywhere in the grid (largest:
+0.182 at chunk 200/similarity, exactly at the bar, not past it, and in
the *opposite* direction from the representative cell's hit@1 drop).
chunk 500/parent_document shows **zero difference on any of hit@1/5/10**.
The single cleanest "no effect" data point in the grid. Per
METHODOLOGY.md #8a's rule: since the primary test didn't clear the bar,
this robustness check is read as confirming no effect at any
configuration tested, not searched for a config where cleaning "wins"
instead. Direction is inconsistent (hit@1 drops at 200/800, flat at 500;
hit@5 rises at 200/similarity, flat everywhere else): consistent with
noise around a null effect, not a masked trend.

**A methodological finding surfaced by this comparison, broader than
cleaning itself**: the FAQ bucket's *reported* hit-rate also shifted
slightly between raw and cleaned cells at the same representative
configuration (hit@3: 0.808 -> 0.769; hit@10: 0.846 -> 0.885), even
though FAQ documents and embeddings are byte-identical between the two:
cleaning never touches the FAQ side. Cause: `query_combined()` (`eval/
retrievers.py`) merges FAQ and policy hits into one global top-10 by
score before any bucket metric is computed, so a change in policy-side
scores can displace which FAQ hits survive into that shared top-10
window, at any k. The same displacement was independently confirmed
between `format_aware`/`uniform` ingestion-mode cells (which only change
FAQ-side scores) shifting the *policy* bucket's reported hit@5 the same
way. **Bucket metrics in this harness are not perfectly isolated to
changes on their own side of the corpus**. A real limitation of the
combined-top-k design, not a bug, and not specific to this comparison.
It doesn't change the primary-test conclusion above (policy-target
queries' ranking only moves when policy-side scores move, and FAQ-side
scores are provably unchanged here), but it means small (<0.05) cross-
bucket deltas elsewhere in this document should be read with the same
caveat.

Corpus size, for reference: cleaning drops 213 raw policy chunks to 196
(chunk_size=800): 4 fewer pages (title + 3 TOC pages) and stripped
header text on the remaining 73.

## Limitations

- **n=1 per cell**: embedding inference and Chroma similarity search are
  deterministic given fixed model weights (verified by
  `tests/test_eval_determinism.py`), so there is no run-to-run variance
  to report for this pipeline: not a gap in rigor, a property of the
  method (METHODOLOGY.md #7).
- **either (n=9) and neither (n=5) buckets are small.** Findings there
  (the confidence-separation overlap, the either-bucket hit@10 ceiling)
  are consistent across all 24 cells, which is reassuring, but neither
  bucket supports fine-grained claims the way the 26-query FAQ bucket does.
- **Bucket metrics are not perfectly isolated to changes on their own
  side of the corpus** (see "Cleaned vs raw PDF" above): `query_combined()`
  merges FAQ and policy hits into one global top-10 before any bucket
  metric is computed, so a change on one side can displace which hits
  from the *other* side survive into that shared window. Confirmed in
  both directions (PDF cleaning shifting reported FAQ-bucket numbers;
  FAQ ingestion mode shifting reported policy-bucket numbers). Cross-
  bucket deltas under ~0.05 anywhere in this document may reflect this
  rather than a real effect on the bucket being discussed.
- **No LLM-answer-quality arm was built or run**, per the phase 2 brief:
  documented as an optional, API-key-gated extension in the README,
  not attempted here. The cost-and-recall baseline arm above *was*
  built and run; don't conflate the two.
- **Token counts are approximate** (chars/4 heuristic), not a real
  tokenizer count. Good enough for the order-of-magnitude cost
  comparison above; not precise enough to bill against.
- **Index build cost (~17.2s) was measured once, on one machine, on one
  configuration**. Illustrative of the qualitative asymmetry (retrieval
  has a build step, stuffing doesn't), not a rigorous benchmark.
- **PDF text extraction quality (pypdf) is a fixed upstream factor**,
  not evaluated on its own. If page 31's population-norms text or
  similar were extracted more completely, `retrievable_but_incomplete`
  flags like pol-01 might not apply. This analysis treats the
  extracted text as given, not as something under test.
- **The confidence-separation finding (Finding #1) is about this
  embedding model and this corpus size specifically**. It should not
  be read as a general claim that similarity search can never support a
  confidence threshold, only that it doesn't reliably do so here, in
  every configuration this sweep tried.

## Headline (for the README)

The system cannot reliably tell "found it" from "didn't" from its
similarity score alone, in any of the 24 configurations tested: no
threshold on top-1 distance separates answerable queries from
unanswerable ones, because their score distributions overlap in every
cell. Format-aware ingestion beats uniform chunking on the FAQ bucket,
but conditionally: the effect only appears once chunk_size exceeds the
FAQ's ~178-char atomic-answer length, and the size of the win tracks a
directly-measured mechanism (rows merged per chunk), not just a
hit-rate gap that could have other explanations. A meaningful chunk of
that FAQ number is lexical rather than semantic match: the honest
semantic-only curve is 0.435/0.783/0.826/0.826 at k=1/3/5/10, plateauing
at k=5 rather than the 0.500/.../0.846 the whole-bucket numbers suggest;
that plateau (hit@10 identical to hit@5) is itself a finding: it means
17% of paraphrased queries never surface their target chunk at any
retrieval depth tested, a representation-quality ceiling that "retrieve
more chunks" cannot fix, compounding the confidence-separation problem
above. Weighed against whole-corpus context stuffing (recall=1.0 by
construction, ~40K tokens, no index build), retrieval at
k=5/chunk_size=800 is ~40x cheaper per query but caps at that same 0.826
ceiling on the hardest queries, a genuine trade-off at this corpus's
scale, not a case either arm wins outright. Retriever strategy
(similarity vs ParentDocumentRetriever) shows no distinguishable effect
on this policy corpus at this scale. Stripping the PDF's running
headers and front matter (title page, table of contents), visible as
page furniture in the demo's own "neither"-query output, was predicted
to make no detectable difference to retrieval before the numbers were
run, and didn't: hit@5 at the representative cell is identical (0.727,
both variants), and no configuration in the 6-way robustness grid clears
the pre-committed 0.182 bar. The ugliness was cosmetic, not costly.

## Phase 4: LLM-routed retrieval with an LLM abstention judge, built on LangGraph

Run clean (`run_invalid: false`, 397 real calls, zero fail-opens, per
`eval/METHODOLOGY.md` #15a). Four arms compared at the representative
cell (`cleaned_pdf__format_aware__cs800_ov100__similarity`, k=5) against
the frozen n=12/n=39 ground truth (#11), C1/C2 each run 3 times. Arm A
(never abstains, by construction) reproduces Finding #1 exactly: recall
0.0, precision undefined -- included as the reference point every other
arm is measured against, not as a competitor.

**Formal win condition: not met.** `#16` defines "better than the
threshold" as strictly higher recall AND not-lower precision than arm
B's optimized cutoff. The judge fails the recall clause outright: arm B
recalls 0.667 (8/12) against C1's flat 0.583 (7/12, identical in all 3
runs) and C2's 0.500-0.526 (10/19-10/20). Per #16's own framing -- "a
judge that trades recall for precision against the threshold... is a
tradeoff to report, not a win" -- that is exactly what happened, and it
is reported as a loss on the formal condition, not softened into one.

**The real result: a precision/recall tradeoff, not a wash.** Arm B buys
its recall lead by over-abstaining broadly: precision 0.421 (8 of 19
abstentions correct, 11 wrong) -- a single global cutoff can't
distinguish a genuine miss from an answerable query with a middling
score. Both LLM arms trade some of that recall for a large precision
gain: C1 0.538-0.583 (0.636-0.700 excluding the 3 retrievable_but_incomplete
hits); C2 0.667-0.714 on the full 39 (0.833-0.909 excluding those same
3). The smallest observed gap -- C1's worst run (0.538) against arm B
(0.421) -- is 0.117; C2's smallest gap (0.667 against 0.421) is 0.246.
Both comfortably clear each arm's own 3-run precision range (C1: 0.045;
C2: 0.048 -- `#15`'s committed bar), so this precision advantage survives
"cannot be distinguished" and is reported as real. The run-to-run
wobbles that don't survive that bar -- C2's recall (0.500-0.526, a
single query flipping) and C1's small precision dip in one run
(0.583->0.538, also one additional false positive) -- are exactly the
1-in-12/1-in-39 noise `#15`/`#15a` anticipated, and are reported as such,
not resolved into a story either arm is "really" better within that range.

**Mechanistic corroboration: does not confirm as predicted, and the
prediction itself was wrong about which clause would fail.** Before any
result existed, this section predicted the opposite of what happened:
that the judge's strictness would cost precision on the 39, not recall
on the 12. The actual result inverts that -- recall is where the judge
loses, precision is where it wins decisively -- and that reversal is
reported plainly rather than quietly reconciled. `#16` separately
predicted the precision advantage, once found, would come from the
judge catching the 7 "confidently-wrong-retrieval" misses (`pol-08`,
`pol-09`, `pol-10`, `faq-01`, `faq-05`, `faq-14`, `faq-16`) that a
distance-only threshold structurally can't see. Per-query inspection
(reconstructed from the run's cached LLM responses -- see the note
below -- and validated to reproduce the persisted aggregate confusion
matrices exactly before being used here) shows this is not what
happened either. Arm B actually catches 5 of those 7 (`pol-10`,
`faq-01`, `faq-05`, `faq-14`, `faq-16` all score at or above its 0.780
cutoff); C1 catches 4/7 (misses `faq-05`, `pol-08`, `pol-09`); C2
catches 3/7 (misses `faq-05`, `faq-14`, `pol-08`, `pol-09`). The
threshold does not fail on this set the way the prediction assumed --
most of the "confidently wrong" 7 don't actually carry a deceptively low
score. Only two, `pol-08` (top-1 distance 0.648) and `pol-09` (0.719),
sit meaningfully below the cutoff and are genuinely invisible to any
distance-only rule -- and neither the judge nor the threshold catches
either of them; both remain unsolved by every arm tested. The judge's
real precision advantage is not "catching hard misses arm B can't" --
it's fewer false abstentions on the 39: arm B wrongly abstains 11 times,
C1 5-6, C2 4-5. That is a different, still content-vs-score mechanism
(reading an excerpt lets the judge recognize a genuine, if
middling-scored, answer instead of abstaining on it), but it is not the
mechanism `#16` pre-registered, and that correction is stated here
rather than counting a different win as if it were the predicted one.

**The `#15`/`#16` variance-clause wording is inconsistent, noted for
integrity, and moot here.** `#16`'s win-condition sentence requires the
gap to exceed "the empirical run-to-run standard deviation from #15";
`#15` itself was later amended to commit the full 3-run range
(max-min), not a standard deviation, as the actual functional form. That
inconsistency exists in the committed text as written and is flagged
rather than silently reconciled. It does not change this run's
conclusion either way: the recall clause fails outright -- arm B's
recall lead is a full query or more above either LLM arm, with zero
variance in C1's case -- so the win condition is already unmet before
any range-vs-stdev question is reached.

**Routing accuracy, kept separate from abstention quality (`#17`).**
C2's routing accuracy is 0.703 (26/37 of `faq`/`policy_pdf` queries),
stable across all 3 runs. Of C2's 9-10 missed abstentions per run, 3
(occasionally 4) trace directly to a routing miss -- `faq-20`,
`faq-22`, `faq-23` (and, in one run, `faq-10`), all misrouted to
`policy_pdf` when `faq` was expected -- and are named here as routing
failures, not folded into "the judge failed." The remaining 6 missed
abstentions per run (`pol-09`, `either-05`, `either-09`, `neither-02`,
`neither-04`, `faq-05`) occur where routing was correct or not
applicable, and are judge failures proper.

**n=12, stated plainly.** `#16` pre-authorized "cannot be concluded at
n=12" as an acceptable outcome. This run does not report a declared
winner: it reports a mechanistically-grounded precision/recall
tradeoff, a formal loss on the pre-registered win condition, and an
honest correction to the pre-registered mechanism once per-query data
was available to check it against.

*Per-query note*: the harness (`eval/agentic_sweep.py`,
`eval/run_agentic_eval.py`) persists only aggregate confusion matrices
to `results/agentic_eval_results.json`; it does not log per-query
records. The per-query breakdown above was reconstructed by replaying
the same 51-query x 3-run x 2-arm loop against the run's own populated
LLM cache (`eval/.llm_cache.json`): every call was a cache hit (0 new
API calls), and the reconstruction was validated to reproduce the
persisted aggregate tp/fp/fn/tn exactly, for all 6 run/arm
combinations, before being used for the analysis above.

## Phase 5: corrective retrieval + grounded-generation loop, built on LangGraph

Run clean: **545 real (non-cached) OpenAI calls, `run_invalid: false`,
zero fail-opens** across all three runs (verified before any of the
analysis below was written). `src/adapters/agentic.py` and its linear
graph are unaffected -- confirmed by `git diff` (one docstring
clarification, no logic change) and by `tests/test_agentic.py` passing
unchanged (166 tests green overall, on both the original Windows
environment and the WSL environment the real run was executed from,
after a Norton TLS-interception issue on Windows made the direct run
impossible -- see below). Full methodology, ground-truth caveats, and
the faithfulness rubric: `eval/METHODOLOGY.md` #19-22.

Unlike phase 4, `results/corrective_eval_results.json` **does** persist
per-query records and node-path traces (`corrective_runs[].per_query`),
so every number below is checked against that file directly, not
reconstructed after the fact. One provenance note on that file itself:
its own `real_llm_calls_made` reads 0 and `run_invalid: true`, because
the committed JSON reflects a later, deliberate $0 cache-hit replay that
added the per-query/trace/per-run-cost fields to an already-validated
run -- the acceptance check correctly can't distinguish "legitimate full
replay" from "silent failure" and conservatively flags any zero-call run,
exactly as it's supposed to. The real, paid run that produced these
exact aggregate numbers made 545 calls and independently passed
`run_invalid: false` before that replay; the file's own
`_provenance_note` field records this.

### Predictions, checked against what was pre-registered in `#22`

**Prediction 1 (rescue rate vs. neither-recall) -- held, precisely.**
Of the 7 `RESCUE_TARGET_IDS` (genuinely answerable queries phase 2-4's
single-shot retrieval missed), the corrective loop does not abstain on
5: `pol-08`, `pol-09`, `faq-05`, `faq-14`, `faq-16` -- a 71.4% mechanical
rescue rate against arm A/B/C1/C2's structural 0% on this exact set (see
`#19a`). Meanwhile the 5 `neither` queries' recall -- 3/5 = 60% (`tp` on
`neither-01/03/05`, `fn` on `neither-02/04`) -- is **identical**, to the
percentage point, to phase 4's own C2 arm on the same 5 queries, checked
directly: replaying phase 4's unmodified `agentic_sweep.run_arm_c` against
its own already-populated cache (`0 new calls`) gives C2 neither-only
recall of 3/5 = 60% in all 3 of its runs too. The prediction's shape --
meaningful improvement on the rescue-target bucket, no comparable change
to `neither`-query behavior -- is exactly what happened.

Caveat that matters: the mechanical rescue rate overstates genuine
content recovery. Inspecting the actual generated answers (below) shows
one of the 5, `faq-16`, is not a real rescue -- see "the abstained flag
undercounts real non-answers." **Genuine content-rescue rate: 4/7 =
57.1%**, still clearly above 0%, still a real result, just smaller than
the mechanical number suggests if read alone.

**Prediction 2 (a fabricated claim slips past `grade_generation` as a
false "grounded") -- not observed; a different, more interesting gap was
found instead.** `faithfulness_rate` is 1.0 in all three runs. Every one
of the 12 `SHOULD_ABSTAIN_IDS` answers was inspected directly (not just
scored), plus a keyword scan across all 51 generated answers for
hedge/non-answer phrasing -- zero fabricated or unsupported claims were
found anywhere in this run. The prediction, as stated, is falsified here
and that is reported plainly rather than reframed as a near-miss.

What was found instead is arguably a more important gap: **the graph's
`abstained` flag undercounts real non-answers.** Three of 51 queries
(`neither-02`, `neither-04`, `faq-16`) produced a generated answer that
honestly states the excerpts don't contain the information asked for
(e.g. *"The excerpts do not provide specific information about the total
cash benefit given under PMMVY for a pregnant woman's first child"*), and
`grade_generation` correctly judges that claim faithful -- it is faithful,
the excerpts genuinely don't say it. But `respond_node` sets
`abstained: False` unconditionally whenever `grade_generation` says
"grounded," with no path that treats "the answer itself says it doesn't
know" as equivalent to abstention -- only budget exhaustion routes to
`abstain_node`. So these three are graded/faithful/non-fabricated and
still counted as false negatives in the confusion matrix. This is a real
architectural gap (a natural follow-on would be a check on the generated
answer's own content, or a "no answer" field in `GeneratedAnswer`, routed
to `abstain_node` directly) -- named here rather than silently absorbed
into the recall number, and it is why the genuine content-rescue rate
above (4/7) is lower than the mechanical one (5/7): `faq-16` is this
exact pattern, not a rescue.

**Prediction 3 (budget is not free, even on already-answerable queries)
-- held, and more materially than the prediction implied.**
`corrective_fire_rate` is 0.333 (17 of 51 queries triggered at least one
rewrite+re-retrieve). Of those 17: only **6 fired on their intended
target** (a `SHOULD_ABSTAIN_IDS` query); **11 (65%) fired on queries that
were already answerable by single-shot retrieval** and did not need
correction. Of those 11 unnecessary firings, 7 (64%) then **exhausted the
full budget and wrongly abstained** -- `pol-01`, `either-03`, `either-08`,
`faq-06`, `faq-07`, `faq-09`, `faq-12` -- converting a query arm A would
have answered correctly (arm A never abstains, by construction) into a
false refusal. Only 4 of the 11 (`pol-05`, `pol-06`, `either-02`,
`faq-22`) recovered and answered anyway, at the cost of extra calls for
no net benefit. **These 7 wrongful abstentions are the entire false-
positive count** (`fp: 7`) -- every single false positive in this run
traces to an unnecessary correction that didn't recover, not to some
other cause.

### Agentic (corrective) vs. arm-A baseline, all 51 queries, 3 runs

| | arm A (plain retrieval, never abstains) | Phase 5 corrective loop |
|---|---|---|
| Abstain recall (of 12) | 0.0 (0/12) | 0.417 (5/12) |
| Abstain precision | undefined (0 abstain predictions) | 0.417 (5/12 abstentions correct) |
| False positives (of 39 answerable) | 0 | 7 (18%) |
| Faithfulness rate (of non-abstained answers) | n/a (no generation step) | 1.0 (all 3 runs) |
| Rescue rate (of the 7 genuinely-answerable misses) | 0% (structural) | 71.4% mechanical / 57.1% genuine |
| `neither`-only recall (of 5) | 0.0 (structural) | 0.60 (identical to phase 4 C2) |
| Mean loops per query | 1 (no loop) | 1.569 |
| Corrective fire rate | n/a | 33.3% (17/51) |
| Run-to-run variance (3 runs) | none (deterministic, no LLM) | **none measured on the aggregate** -- see note below |
| Real LLM calls | 0 | 545 total |
| Estimated cost | $0 | ~$0.10 (rough, see below) |

**Zero measured variance across 3 runs, on the aggregate -- flagged, not
over-interpreted.** All three runs produced byte-identical `tp/fp/fn/tn`,
rescue rate, faithfulness rate, and mean-loops. This was checked, not
assumed: `run_index` was independently verified to reach all five
wrapped LLMs (a dedicated integration test,
`test_run_corrective_arm_advances_run_index_on_all_five_wrappers_not_just_some`,
confirms a fresh real call per run, not a cached replay), and the raw
per-query records confirm individual queries *do* differ between runs
(`run0 != run1` at the per-query level) -- the aggregate identity is
runs landing on the same totals through different per-query paths, not a
caching bug and not literal per-query determinism. This contrasts with
phase 4's own committed finding of small but real run-to-run wobbles
under the same temperature=0/seed=42 settings; noted as an observation
at n=3, not a claim that this graph is more deterministic than phase 4's.

**Cross-phase continuity, checked directly, not assumed:** of phase 5's
7 false negatives, 4 (`pol-09`, `neither-02`, `neither-04`, `faq-05`) are
**the same queries phase 4's own C2 arm already missed** (verified by
replaying phase 4's unmodified `run_arm_c` against its own cache, $0
cost) -- these are inherited weaknesses in the shared `JudgeDecision`/
`JUDGE_PROMPT` grading step (`grade_documents` reuses it verbatim), not
new regressions phase 5 introduced. The other 3 (`pol-08`, `faq-14`,
`faq-16`) are cases phase 4's C2 got *right* that phase 5 gets wrong. Two
of those three (`pol-08`, `faq-14`) never triggered a rewrite at all
(`loops: 1`) -- `grade_documents`, nominally the identical prompt/schema/
settings as phase 4's judge, returned a different verdict on what should
be the same call. `temperature=0`/`seed=42` is documented (here and by
OpenAI) as best-effort, not guaranteed, determinism; this looks like an
instance of that rather than something attributable to any change this
phase made, but it isn't fully explained beyond that, and is stated
plainly rather than papered over.

### Cost

545 real (non-cached) calls for the full 51-query x 3-run evaluation
(`route`'s calls were free -- reused verbatim from phase 4's own cache,
same node, same prompt, same deterministic settings, same 51 queries;
the other four call sites, `grade_documents`/`rewrite_query`/`generate`/
`grade_generation`, were all fresh). Rough, order-of-magnitude estimate
using gpt-4o-mini list pricing and an assumed blended ~800 input/~120
output tokens per call (`eval/run_corrective_eval.py::_estimate_cost_usd`,
which states its own caveats): **~$0.10** -- not reconciled against actual
OpenAI billing telemetry, since `CachedStructuredLLM` discards token-usage
metadata when caching a parsed result. For comparison, phase 4's own real
run made 459 calls for a measured ~$0.06; phase 5's higher per-call cost
(longer prompts carrying retrieved excerpts, plus a real generation step)
outweighs its lower-than-worst-case call count. The hard-stop budget
(1,377 calls, 1.5x the pre-registered ~918 estimate) was never
approached.

### Honest negatives, stated together

1. **18% of already-answerable queries (7/39) were wrongly refused**
   after the corrective loop exhausted its retry budget -- a real,
   measurable regression relative to arm A, which never refuses anything.
2. **65% of all corrective firings (11/17) were unnecessary** (fired on
   queries that didn't need correction), and **most of those (7/11)
   then made the outcome worse**, not merely costlier.
3. **The mechanical rescue rate (71.4%) overstates real capability**;
   one of the five "rescues" (`faq-16`) is the same honest-non-answer
   pattern as the `neither`-query failures, not genuine content recovery.
4. **The `abstained` flag misses real non-answers** on 3 of 51 queries,
   for architectural reasons named above, not a scoring choice made to
   flatter the numbers.
5. Of the 12 should-abstain misses, only 3 (`pol-08`, `faq-14`, `faq-16`)
   are novel to phase 5; the other 4 are inherited from phase 4's own
   judge, not new failures this phase introduced.

**Net read:** the corrective loop trades arm A's 0%/0-cost abstention
for a real, if partial, ability to catch genuinely unanswerable and
genuinely-missed queries (41.7% recall where arm A structurally scores
0%), at a real, measurable cost -- 18% of previously-fine queries wrongly
refused, and roughly two-thirds of all corrective firings spent on
queries that didn't need help. This is a tradeoff, consistent with this
repo's phase-2 and phase-4 findings, not a clean win in either direction,
and is reported as such.
