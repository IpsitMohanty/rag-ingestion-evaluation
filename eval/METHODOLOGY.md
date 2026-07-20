# Phase 2 evaluation methodology

Settled in writing before the harness was built or the sweep run, per
instruction: no rule below was chosen after seeing results.

## 1. Per-bucket reporting, never pooled

Every metric is reported separately for `faq`, `policy_pdf`, `either`, and
`neither` query buckets. There is no single pooled hit-rate/MRR number
anywhere in the harness output or `ANALYSIS.md` -- 26 of 51 queries are
FAQ-targeted, and format-aware ingestion's main effect is on that bucket
specifically, so a pooled number would be dominated by it and would
misrepresent the policy-side findings. `results/sweep_results.json`
stores metrics keyed by bucket; there is no top-level "overall" key.

## 2. Policy hit-rate with and without retrievable_but_incomplete

4 queries are flagged `retrievable_but_incomplete: true` in
`eval/queries.yaml` (pol-01, pol-05, pol-10, either-08) -- cases where the
correct chunk is genuinely retrievable but doesn't fully answer what was
asked. These score as hits under hit-rate/MRR as defined (the correct
chunk *was* retrieved), which would silently overstate how well the
system serves the user if left in one number.

For every bucket that contains at least one such query (`policy_pdf`:
pol-01/05/10; `either`: either-08), the harness reports hit-rate/MRR
**twice**: once over all queries in the bucket, once excluding the
retrievable_but_incomplete ones. Both numbers are reported side by side;
neither is treated as "the" headline number.

## 3. Scoring rules for "either" and "neither"

**either**: a hit counts if the top-k contains a match against *any* of
the listed `ground_truth` refs, regardless of which source it came from
(OR logic, not AND). Rationale: "either" means the system found *a*
correct answer, not that it found both sources' versions of it --
requiring both would be testing something the query was never meant to
demand.

**either queries are excluded from source-routing accuracy.**
Source-routing accuracy measures whether the top-k came from the source
type the query is supposed to route to. For an "either" query, both
source types are correct by definition, so there is nothing to
discriminate -- including them would make routing accuracy look
artificially good (almost any real hit "counts") without the metric
having tested anything. Source-routing accuracy is computed only over
`policy_pdf` and `faq` queries, where exactly one source type is correct.

**neither: hit-rate is undefined, not zero.** There is no correct
passage, so "did it hit" isn't a meaningful question, and treating a
"miss" as a success would just be measuring the inverse of a
still-undefined thing. `neither` queries are excluded from hit-rate/MRR
and from source-routing accuracy entirely.

Instead, `neither` queries are scored on **confidence separation**: the
harness records each query's top-1 similarity score (Chroma's raw
distance -- lower is more similar -- from whichever sub-index produced
the best match, comparable across sub-indexes because every sub-index in
a given sweep cell shares the same embedding model and the same default
distance metric) and compares the *distribution* of top-1 scores for
`neither` queries against the distribution for queries that do have a
correct answer (`faq` + `policy_pdf` + `either`, pooled for this
comparison *only* -- pooling is appropriate here because the question
being asked is "can the system tell confident from unconfident at all",
not "which bucket is confident"). If the two distributions overlap
substantially, the retriever cannot distinguish "I found the answer"
from "I found the least-bad thing available," which is a real,
reportable limitation of similarity search as a mechanism -- not a
footnote, its own section in `ANALYSIS.md` regardless of which way it
comes out.

## 4. Reworded-FAQ lexical overlap: measured, then stratified (not reworded further)

Jaccard similarity on normalized tokens (lowercased, punctuation
stripped, English stopwords removed) was computed between each of the 26
reworded queries and its stored `original_question`. Distribution:

- n=26, mean=0.134, median=0.101, max=0.50, min=0.00
- 16 of 26 share zero tokens with the original question
- 3 exceed 0.30: `faq-04` (0.50, "RCH Profile"), `faq-19` (0.50, "Home
  Visit"), `faq-15` (0.43, "Poshan Tracker Dashboard")

**Decision: keep as-is and stratify by overlap band, rather than reword
further.** All 3 high-overlap cases share sharing named UI
features/proper nouns (RCH Profile, Home Visit, Poshan Tracker
Dashboard) -- asking about a named feature without naming it produces
either an unrecognizably vague question or a different question
entirely, so further rewording would trade clarity for a cosmetically
lower score, not a more honest one. Given 23/26 already sit at or below
0.25 and the distribution has a long flat tail at zero, forcing the last
3 down further isn't worth the distortion.

Instead, `faq-04`, `faq-15`, and `faq-19` are tagged
`high_lexical_overlap: true` in the query set, and the harness reports
FAQ-bucket hit-rate/MRR for the high-overlap (n=3) and low-overlap
(n=23) subsets separately, alongside the whole-bucket number. This
directly answers the question the overlap check exists to ask: if the
high-overlap subset scores markedly better, that gap is a direct,
quantified estimate of how much of FAQ-bucket retrieval is riding on
lexical match rather than semantic match, instead of that effect hiding
inside a single aggregate.

## 5. Two additional decisions, load-bearing for the central hypothesis, settled now

Not asked for directly, but both determine whether the ingestion and
retriever sweeps test what they're supposed to -- settled here rather
than picked after seeing results.

### 5a. What "uniform chunking" means for the FAQ set

The FAQ loader already returns one `Document` per Q&A row (`tab`,
`subcategory`, `question` preserved as metadata) regardless of ingestion
mode -- rows are never merged into a single blob before that point.
Running `RecursiveCharacterTextSplitter` over those already-separate,
~178-char-average Documents individually would rarely change anything at
the chunk sizes in this sweep (200-800 chars): most rows would pass
through as one chunk each, and the "uniform ingestion" arm would be
almost indistinguishable from "format-aware" -- not a fair or meaningful
baseline, and not what the brief's own hypothesis text describes
("merges unrelated Q&As into one chunk and pollutes retrieval").

**Decision:** the `uniform` ingestion arm concatenates all 126 FAQ
Q&A texts into a single ordered blob first -- discarding row,
tab/subcategory, and question-boundary structure entirely -- then runs
the *same* `RecursiveCharacterTextSplitter` config used for the PDF over
that blob. This is what a generic, format-unaware pipeline actually
looks like when pointed at this corpus (no per-source-type branching,
no knowledge that the FAQ file is structured records rather than free
text), which is the scenario "uniform chunking" is supposed to stand in
for. The splitter runs with `add_start_index=True`; each resulting
chunk's character span is checked against every original row's known
span in the blob, and any row whose span overlaps the chunk is recorded
in that chunk's `faq_indices` list. A chunk overlapping more than one
row's span *is* the hypothesized "merges unrelated Q&As" failure mode,
made directly countable: the harness reports the mean number of
distinct FAQ rows per uniform-mode chunk as its own statistic, not just
inferred from a hit-rate gap.

### 5b. Retriever sweep (similarity vs ParentDocumentRetriever) applies to the policy corpus only

`ParentDocumentRetriever` splits documents into large "parent" chunks and
small "child" chunks, matching on the child and returning the parent.
Applied to the FAQ set -- already-atomic ~178-char answers -- there is no
meaningful parent/child split to make; forcing one would fragment
exactly the atomicity the format-aware ingestion arm is designed to
preserve, for a strategy whose whole rationale is long-document
handling.

**Decision:** the `retriever` sweep dimension (`similarity` vs
`parent_document`) changes only how the **policy PDF** sub-index is
retrieved. The FAQ sub-index always uses plain similarity search over
its (format-aware or uniform, per the ingestion arm) documents,
independent of the retriever arm. A lightweight combined-query step
merges scored hits from the FAQ sub-index and the policy sub-index (by
raw distance, both built with the same embedding model so scores are
comparable) and returns the overall top-k. For `parent_document`,
child-level distances from its internal vectorstore are used as the
comparable score, with the resolved parent `Document` returned as the
hit -- `ParentDocumentRetriever` has no native `_with_score` method, so
this is the harness's own scoring shim, documented in `eval/retrievers.py`.

## 6. Sweep grid

Deliberately small, per the brief ("a few sensible points, not an
exhaustive grid... runs in minutes"):

- ingestion mode: `format_aware`, `uniform` (2)
- chunk size/overlap: (200, 20), (500, 50), (800, 100) -- the last
  matches phase 1's `config.py` default for the policy PDF; 200 is
  smaller than the FAQ's ~178-char average answer, deliberately chosen
  to stress-test the uniform arm's merging behavior on the FAQ blob (3)
- retriever strategy (policy sub-index only): `similarity`,
  `parent_document` (2)

12 sweep cells total. `parent_document`'s parent splitter is fixed at
3x the cell's chunk_size/overlap (not swept independently -- one more
free dimension than the corpus and query-set size can support
meaningfully).

## 7. Run count

n=1 per cell, not repeated. Embedding inference
(`sentence-transformers/all-MiniLM-L6-v2`, no dropout at inference) and
Chroma similarity search are both deterministic given fixed model
weights -- repeating a cell produces bit-identical output, so there is
no run-to-run variance to report for *this* pipeline (unlike a
stochastic algorithm or an LLM-generation arm). Determinism itself is
asserted by a test (`tests/test_eval_determinism.py`) rather than
demonstrated by repeated sweep runs, which would just waste time
reproducing the same numbers.
