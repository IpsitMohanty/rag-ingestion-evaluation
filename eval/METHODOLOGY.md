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
- PDF variant (policy sub-index only, added per #8 below): `raw`,
  `cleaned` (2)

24 sweep cells total (was 12 before the PDF-variant axis). `parent_document`'s
parent splitter is fixed at 3x the cell's chunk_size/overlap (not swept
independently -- one more free dimension than the corpus and query-set
size can support meaningfully).

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

## 8. Cleaned-PDF ingestion variant: what "cleaned" means, and why not less

Added after inspecting an actual retrieval result for the "neither"
preset query (Poshan ke Paanch Sutra) at k=10: several of the returned
chunks were page furniture, not policy content -- the title page, the
table of contents, and a running header ("Saksham Anganwadi and Poshan
2.0" plus a printed page number) prefixed to every content page. The two-
source opposition (atomic FAQ vs. long PDF) is the experimental design
and stays; this adds a second, orthogonal axis -- how the PDF side is
*cleaned* before chunking -- alongside it, not instead of it. `raw` (the
existing, phase-1 behavior) and `cleaned` are both real, kept,
selectable via `config.SourceTypeSettings.clean` -- see
`adapters/ingestion.py::clean_policy_pdf_documents` for the
implementation this section explains.

**Running header/footer**: detected, not hardcoded to this document's
exact title text -- the most common non-blank first (or last) line
across all 77 raw pages, if it recurs on >=60% of them. Below that
threshold a line is treated as real, page-specific content (this
matters concretely: the document's own "***" section-end marker closes
5 of 77 pages -- 6%, nowhere near the threshold -- and is correctly left
alone rather than stripped as a false "footer"). The header line found
this way ("Saksham Anganwadi and Poshan 2.0") is stripped from each
page's first line when present, along with an immediately-following (or
blank-line-separated -- both forms occur in this PDF) page-number-only
line. No running footer was detected in this document (nothing recurs
above the threshold), so footer-stripping is a no-op here -- implemented
for completeness/reuse, not because this PDF needed it.

**Front matter**: two content-based signals, checked against every page,
not a hardcoded page range --
  - **Table of contents**: >=3 dot-leader patterns (`"…….12"`-style)
    on the page. This document's 3 actual TOC pages score 21-23 matches
    each; every other page scores 0-1 -- a wide margin, not a tuned
    threshold sitting close to real content.
  - **Title page**: page index 0 *and* under 300 characters once the
    header's stripped. Deliberately narrow (position AND content, not
    either alone) -- this document's real short content pages (closing
    paragraphs, "***" section-enders, e.g. printed pages 6 and 73) run
    150-250 characters but are never page index 0, so they're never at
    risk of being flagged.

Result on this corpus: pages 1-4 dropped (title page + 3 pages of table
of contents), 73 of 77 pages survive with their running header removed.
`metadata["page"]` is left unchanged on every surviving page -- this is
required, not incidental: `eval/queries.yaml`'s policy_pdf ground truths
are page-number references (e.g. `pol-07` -> page 18), and none of them
fall in the dropped 1-4 range (checked directly), so every existing
ground truth stays meaningful under both variants without any changes
to the query set.

**Explicitly not claimed**: this is a corpus-specific heuristic, checked
against this one document's actual structure (every threshold above is
justified with this document's real numbers, not chosen a priori). It is
not offered as a general-purpose PDF front-matter/header stripper --
`tests/test_ingestion_pdf.py::TestPdfCleaning` pins down the specific
pages/behavior this corpus produces, not a claim that the same
thresholds would work on a differently-structured document.

**Decision on which variant ships in the deployed app**: deferred until
the swept numbers (`results/ANALYSIS.md`) show whether cleaning
measurably changes retrieval quality -- reported, not assumed, per the
same standard as every other finding in this evaluation.

### 8a. Decision rule, written before the raw-vs-cleaned numbers were looked at

**Correction to the premise this was requested under**: this pipeline is
*not* a case of "rerunning gives different numbers, so there's a rough
noise floor to measure." Embedding inference and Chroma similarity
search are deterministic given fixed model weights -- `tests/
test_eval_determinism.py` asserts bit-identical output across repeated
runs of the same cell (METHODOLOGY.md #7). There is no run-to-run
variance here to build a threshold from; that description matches a
stochastic training pipeline (e.g. epoch-to-epoch variation in a model-
training repo), not this one. The honest equivalent available here is
**cross-configuration spread**: how much policy_pdf hit-rate already
moves between configurations that have nothing to do with cleaning
(chunk_size, retriever strategy), which puts a floor under how small a
raw-vs-cleaned difference can be before it's indistinguishable from
"numbers just move around."

Measured from the existing (pre-cleaning-axis) sweep, the 6
chunk_size x retriever_strategy combinations on the *raw* PDF:

| chunk_size | retriever | policy_pdf hit@1 | policy_pdf hit@5 | policy_pdf hit@10 | MRR |
|---|---|---|---|---|---|
| 200 | similarity | 0.455 | 0.727 | 0.818 | 0.579 |
| 200 | parent_document | 0.455 | 0.727 | 0.727 | 0.564 |
| 500 | similarity | 0.545 | 0.545 | 0.727 | 0.571 |
| 500 | parent_document | 0.545 | 0.727 | 0.818 | 0.620 |
| 800 | similarity | 0.636 | 0.727 | 0.909 | 0.691 |
| 800 | parent_document | 0.636 | 0.727 | 0.909 | 0.691 |

hit@5 spans 0.545-0.727 (a 0.182 spread) across configurations that
never touch cleaning at all. With n=11 policy_pdf queries, one query
flipping is worth exactly 1/11 = 0.0909 -- so this 0.182 spread is
already 2 queries' worth of movement from chunk_size/retriever choice
alone.

**Decision rule (fixed now, applied without adjustment once the numbers
land):**

> **Primary test**, at the same representative cell every other finding
> in this evaluation is reported against (`format_aware`, chunk_size=800,
> `similarity`): cleaning counts as a real effect on the policy_pdf
> bucket only if **hit@5 changes by more than 0.182** between the raw
> and cleaned cells there -- the largest swing already observed between
> configurations that aren't testing cleaning at all. Anything at or
> below 0.182 is reported as "cannot be distinguished at this scale," not
> rounded up to a winner, no matter which direction it points.
>
> **Robustness check**: the same raw-vs-cleaned comparison is also read
> across all 6 chunk_size x retriever_strategy pairs. This is reported
> descriptively, not gated on a second threshold -- if the primary test
> passes but the effect is conditional on chunk_size the way Finding #2
> (format-aware vs uniform chunking) was, that conditional shape gets
> written down as what it is, not forced into a single number. If the
> primary test fails (does not clear 0.182), the robustness check is
> reported as confirming "no effect at any configuration tested," not
> searched for a config where it might have cleared the bar instead.
>
> The same 0.182-equivalent bar (proportionally, since n differs)
> applies to hit@1/hit@10/MRR and to policy_pdf source-routing accuracy
> as secondary metrics, but hit@5 at the representative cell is the
> deciding number.

**Prediction, logged before the numbers are read:** cleaning will make
**no detectable difference** at this threshold. Front matter and running
headers amount to a handful of chunks (4 dropped pages, ~340 characters
of header text removed per surviving page) against 213 policy chunks at
chunk_size=800 (more at smaller sizes) -- unlikely to be within the
top-k for most of the 11 real policy queries, whose ground truth lives
in substantive content pages, not the pages/text being removed. If this
prediction holds, that is itself the reportable result: the front-matter
and header artifacts visible in the k=10 "neither" demo output were
cosmetic to retrieval quality, not costly to it -- a different claim
than "cleaning doesn't matter," and the writeup will say which one the
data supports, not whichever framing sounds more decisive after the
fact.
