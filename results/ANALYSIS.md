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

Not run yet. This section holds only the pre-registered expected failure
mode, written before any LLM call is made, so it cannot be adjusted
after seeing results. Full findings replace this section once the
scored run (eval/METHODOLOGY.md #9-18) completes.

**Expected failure mode, named now, before results exist**: the judge
prompt (`src/adapters/agentic.py`) is deliberately strict: it explicitly
withholds credit from excerpts that are on-topic or merely reference
that something exists without stating the specifics asked for. That
strictness biases the judge toward abstaining whenever an excerpt is
even slightly indirect or incomplete, not only when it's genuinely
irrelevant. The likely consequence: recall on the 12 should-abstain
queries is probably fine, since strictness helps there, but precision on
the 39 should-not-abstain queries is the more likely place for the judge
to miss the win condition (METHODOLOGY.md #16: higher recall AND
precision not lower than arm B) -- a strict judge over-abstaining on
borderline-but-genuinely-answerable excerpts in the 39 would show up
exactly as a precision loss, not a recall loss. If the judge fails to
beat the threshold, this is the failure mode to check for first, before
concluding the content-based mechanism doesn't work at all.
