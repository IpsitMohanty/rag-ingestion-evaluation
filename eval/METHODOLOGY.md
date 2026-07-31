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

# Phase 4: LLM-routed retrieval with an LLM abstention judge, evaluation methodology

Settled in writing before any LLM call is made, per the same discipline as
phase 2. Naming discipline, stated once here and binding for README,
ANALYSIS.md, and commit messages: this is **"LLM-routed retrieval with an
LLM abstention judge, built on LangGraph."** A linear four-node graph
with two LLM calls and no loops, no tool selection beyond retrieval, and
no self-correction is a routed pipeline, not an agent. Never call it an
"agentic AI system."

## 9. What this phase tests, and the four arms

Finding #1 (phase 2) is that similarity distance cannot separate
answerable from unanswerable queries: the `neither` score range sits
entirely inside the `should-hit` range, so no threshold gates an "I
don't know" response. This phase asks the direct follow-up: can an LLM
judge do what the similarity score cannot?

Four arms, compared on the same query set:

- **A, baseline**: similarity retrieval only, no abstention mechanism at
  all. Existing phase 2 numbers at the representative cell. Included so
  every other arm has a like-for-like reference point, not because A is
  expected to do anything: it always answers, never abstains, by
  construction.
- **B, threshold**: same retrieval as A, plus the best-possible distance
  cutoff fit directly on this labeled set (see #14). Given its best
  possible shot deliberately, so a loss here isn't a strawman.
- **C1, judge only**: the LLM judge sees the same both-index merged
  chunks A/B retrieve, with no routing gate. Isolates judge quality from
  routing quality.
- **C2, full path**: the LLM router gates which sub-index(es) are
  queried, then the judge sees only what that routed retrieval returned.
  The delta between C1 and C2 is the measured cost of routing.

## 10. Retrieval configuration, fixed across all four arms

Representative cell `cleaned_pdf__format_aware__cs800_ov100__similarity`,
k=5. This is the same cell phase 2 treats as representative everywhere
else in this document, and the same cleaned-PDF variant the deployed app
actually runs, so phase 4's findings describe the system as shipped, not
a different configuration invented for this evaluation. k=5 matches
phase 2's established "practical figure" convention (real systems
retrieve k=5, not k=1).

**Scoping decision, stated plainly**: this is one cell, not swept across
the 24-cell grid the way phase 2's core findings were. A full sweep at
LLM-call cost would multiply the estimated spend by up to 24x for a
targeted question that doesn't need it. Phase 4's findings are
conditioned on this one configuration, the one actually deployed, and
must be reported as such, never generalized to "this corpus" or "this
embedding model" the unqualified way phase 2's swept findings could be.
Carry this into every generalizability caveat in the writeup.

## 11. Ground-truth abstention labeling is computed per arm, not fixed from the baseline

A/B/C1 share one retrieval configuration (both-index merged, no routing
gate), so they share one ground-truth label set. C2 does not: routing
can change which chunks are retrieved for a given query, so a query
whose target chunk missed A/B/C1's retrieved set may be present in C2's,
or vice versa. Scoring C2 against an A/B/C1-derived label would judge it
against chunks it never saw.

**Rule, per query, per arm**: for `neither` queries, the label is always
"should abstain," arm-invariant, since no correct chunk exists anywhere
regardless of what any arm retrieves. For should-hit queries (`faq` /
`policy_pdf` / `either`), the label is "should abstain" if and only if
the target chunk is absent from *that arm's own retrieved set*, reusing
`eval/metrics.hit_at_k` unchanged (its existing OR-logic already handles
`either` queries' multiple valid ground-truth refs correctly, per
METHODOLOGY.md #3).

For A/B/C1 this is fixed and was computed directly from the
representative cell (deterministic, no LLM call, reproducing exactly
what phase 2's sweep already computed internally but never persisted):

**7 misses among the 46 should-hit queries at k=5**: `pol-08`, `pol-09`,
`pol-10`, `faq-01`, `faq-05`, `faq-14`, `faq-16`. Combined with the 5
`neither` queries, A/B/C1 share a ground truth of **n=12 "should
abstain," n=39 "should not abstain"** (51 total). This is a real
improvement over scoring against the 5 `neither` queries alone, though
n=12 is still small: single-query flips are worth 1/12 of the "should
abstain" class, and the writeup must say so plainly wherever it matters.

For C2, this label is **not** reused. It is recomputed independently,
per query, per run, from whatever chunks that run's routing decision
actually caused to be retrieved. This is deliberate, not a gap: it is
how the routing effect becomes directly visible in the data rather than
only a caveat. Mechanism worth stating in advance: `eval/retrievers.py`'s
`query_combined()` merges FAQ and policy hits into one ranked top-k
before either bucket is scored (already documented as a limitation in
the README and `ANALYSIS.md`, since a correct chunk from one source can
be displaced out of the shared top-k by a closer-scoring but irrelevant
hit from the other source). Correct single-source routing removes that
cross-source dilution entirely, so C2 could plausibly *recover* some of
A/B/C1's 7 baseline misses on correctly-routed queries, not only
introduce new ones on mis-routed queries. Both directions are real,
predicted mechanisms, not just a downside to caveat.

## 12. The four `retrievable_but_incomplete` queries under this labeling

Checked directly against the representative cell, not assumed:

| query | target chunk retrieved at k=5? | which class |
|---|---|---|
| pol-01 | yes | "should not abstain" (39) |
| pol-05 | yes | "should not abstain" (39) |
| either-08 | yes | "should not abstain" (39) |
| pol-10 | **no** | "should abstain" (12), already counted among the 7 misses |

Three of the four (`pol-01`, `pol-05`, `either-08`) sit in the "should
not abstain" class purely because their target chunk was retrieved,
even though that chunk doesn't fully answer the question, exactly the
tension METHODOLOGY.md #2 already established for hit-rate. `pol-10` is
different: it is already labeled "should abstain" for an unrelated
reason (its target chunk was missed entirely at k=5), so its
incompleteness flag doesn't create the same tension.

**Applying the METHODOLOGY.md #2 precedent**: C1 and C2's confusion
matrices (not A's or B's, which have no content judgment to be honest or
dishonest about) are each reported twice: once over all 51 queries, once
excluding `pol-01`, `pol-05`, and `either-08`. Neither number is "the"
headline. If the judge abstains on any of these three, the writeup
states plainly that this is arguably a more honest judgment than
hit-rate's credit, not simply a scored failure, per your explicit
instruction. This is a finding to state, not a footnote.

## 13. Primary metric: confusion matrix, never collapsed to one number

For each arm, against that arm's own ground truth (rule #11):

- **TP**: correctly abstained (should abstain, did abstain)
- **FP**: wrongly abstained (should answer, did abstain), over-caution
- **FN**: missed abstention (should abstain, did not abstain), the
  dangerous silent-failure case Finding #1 is actually about
- **TN**: correctly did not abstain

Report **precision = TP/(TP+FP)** and **recall = TP/(TP+FN)** on the
abstain decision, plus the full matrix. Never collapse to plain
accuracy: at roughly 12-should-abstain vs 39-should-not-abstain, a
mechanism that never abstains already scores 39/51 = 76.5% accuracy
while catching zero of the cases that matter, which is exactly arm A's
degenerate case below.

**Arm A's confusion matrix is Finding #1 restated, not a new result**:
A never abstains, by construction, so it has zero true positives and
zero false positives. Recall on the abstain decision is 0/12 = 0.0.
Precision is undefined (0 predicted positives): report it as "undefined,
no abstain predictions made," not as 0 or 1, either of which would
misleadingly imply a real (if extreme) trade-off was made.

## 14. Arm B: the threshold, given its best possible shot

Candidate cutoffs: every distinct top-1 distance value observed across
the 51 queries at the representative cell (arm A/B/C1's shared
retrieval), rule: "abstain if top1_distance >= cutoff."

**Optimization objective: maximize Youden's J statistic** (sensitivity +
specificity − 1) over the abstain decision, searched over every
candidate cutoff, not plain accuracy. Plain accuracy is the wrong
objective here for the same reason it's the wrong primary metric in
#13: never abstaining already clears 76.5% accuracy, so optimizing
accuracy would bias the "best" cutoff toward ignoring the minority
class entirely. Youden's J is a standard, defensible choice specifically
because it weighs both classes, chosen for that property alone, not to
flatter or handicap the threshold arm either way.

This is an **in-sample optimum**, fit on the same 51 queries being
evaluated: genuinely the best possible shot the brief asked for, not a
strawman, and explicitly not expected to generalize to a new, unseen
query. Report the winning cutoff itself and its full confusion matrix,
not only the optimized statistic.

## 15. Variance across runs, and when to say "cannot be distinguished"

C1 and C2 each run 3 times, temperature 0, seed fixed where the API
supports it.

Unlike phase 2 (METHODOLOGY.md #7), there is no pre-existing dataset of
LLM-judge run-to-run variance to derive a numeric bar from in advance.
Retrieval and Chroma similarity search are deterministic, which is
exactly why phase 2 could commit to a specific number (0.182) in writing
before running; an LLM API call at temperature 0 is explicitly not
guaranteed deterministic, per your own brief. Inventing a specific
numeric bar now, with no data to derive it from, would be exactly the
kind of ungrounded number this evaluation's discipline exists to avoid.

**Decision rule instead, functional form committed here, before any
result exists**: for each arm and each metric (precision, recall on the
abstain decision), compute the **full range across the 3 runs**
(max − min), not a standard deviation. n=3 is too small a sample to
estimate a standard deviation reliably, and a range is the more
conservative (wider) choice between the two, which is the right
direction to err in when the whole point of this rule is not to credit
an effect that might just be noise. Any comparison, C1 vs C2, either
C-arm vs B, whose observed difference is smaller than the larger of the
two arms' own 3-run ranges is reported as **"cannot be
distinguished,"** not rounded into a winner. This is the same
noise-floor-before-crediting-an-effect discipline as phase 2's 0.182
bar, adapted to the fact that the noise floor itself can only be known
after running, not before, and kept deliberately conservative given how
few runs it's estimated from.

**Stated outright, before any result exists, so it cannot be misread
after the fact**: at temperature 0 with a fixed seed, on a metric with
1/12 granularity (recall on the 12 should-abstain queries) or 1/39
granularity (precision-relevant count on the 39), the 3-run range will
almost certainly be **zero** unless the API actually flips a boundary
case, which is expected to be rare. This bar is therefore expected to be
**inert** in this run, not a meaningful filter, and "gap exceeds the
variance bar" will in practice collapse to "gap is greater than zero."
That collapse must not be read as "the gap cleared a real bar of
substance." When the range is zero (or near it), robustness must be
argued from **effect size against n=12** (how many of the 12 flipped,
stated as a count and as 8.3-point-per-query fractions, per the honest
expectation in #16) and corroborated qualitatively (does the mechanism
in #16 actually explain which queries flipped), not from "it beat the
variance bar" alone. The bar still does real work in the case that
matters most: if the API turns out to be less stable than expected and
produces a genuinely nonzero range, this rule is what keeps that
instability from being credited as a real effect.

## 15a. Run acceptance criteria: fail-opens on the 12 should-abstain queries

A fail-open (src/adapters/agentic.py's route/judge call failure,
defaulting to "both"/"answerable") anywhere is excluded from the scored
confusion matrix and reported explicitly, never folded into "the judge
chose to answer" (see the harness's `fail_open_ids` /
`_run_summary`). That handles fail-opens as a data-quality mechanism.

**This is a stricter, additional acceptance criterion for the run as a
whole, not just a per-query exclusion rule**: a fail-open landing on any
of the 12 should-abstain queries (the frozen `SHOULD_ABSTAIN_IDS`,
METHODOLOGY.md #11) is a **re-run trigger for that run, not a
footnote.** It drops the effective n below 12 on the exact hardest,
smallest, most load-bearing slice of this evaluation, and it signals the
judge pipeline is failing specifically on the cases that matter most,
which is itself worth knowing rather than averaging away. The clean-run
bar for accepting a run's numbers into the final comparison is **zero
fail-opens on the 12 should-abstain IDs**; a fail-open anywhere else in
the 39 is excluded and reported as usual (per the harness), but does not
by itself invalidate the run.

At n=12 "should abstain" cases, one query flipping its classification
moves recall by 1/12, about 8.3 percentage points. Stated plainly now:
small differences (1-2 queries) between arms may not be distinguishable
from run-to-run noise, and the writeup must say so rather than round it
into a narrative either way.

## 16. Stated before running: what would make the judge "better than the threshold," and by how much

**Prediction, logged before any LLM call is made**: arm B, even with its
best possible in-sample cutoff, will still perform poorly on precision
and/or recall of the abstain decision, plausibly worse relative to its
task than phase 2's original 5-query framing suggested. Reasoning: 7 of
the 12 true "should abstain" cases are queries where the retriever
returned *something* with a plausible-looking distance score, just not
the right chunk, a confidently-wrong retrieval. That is structurally the
hardest case for any threshold, which can only react to a score's
magnitude, never to whether the returned content is actually correct.

**Corresponding, mechanism-based hypothesis for the judge**: C1 and C2
see the retrieved chunks' text, never their distance score (by design,
see the graph proposal), so they have a plausible mechanism to catch a
confidently-wrong retrieval that a pure distance threshold structurally
cannot. If the judge substantively beats arm B, that would be a
mechanistically explained result, not a coincidence, precisely because
the two approaches have access to different information.

**"Better than the threshold," concretely defined now**: strictly higher
recall AND not-lower precision on the abstain decision than arm B's
optimized cutoff, with the gap exceeding the empirical run-to-run
standard deviation from #15. A judge that trades recall for precision
against the threshold (or vice versa) is a tradeoff to report, not a
win, per "do not collapse to one number."

**Honest expectation, set now, before any number exists**: at n=12, this
study may well not be powered to declare a winner in either direction.
"Cannot be concluded at n=12" is an acceptable, plausible outcome, to be
reported as such and not pushed toward a more decisive-sounding
narrative than the data supports.

## 17. Secondary metric: routing accuracy

Computed for C2 only (C1 has no router). Over `faq` and `policy_pdf`
queries only, excluding `either` (no single correct route exists) and
`neither` (no source is "correct" for an unanswerable query), per the
same exclusion rule as METHODOLOGY.md #3. Route's own explicit decision
is checked against `expected_source`.

**Precise distinction from phase 2's `source_routing_correct`,** stated
so the two numbers are never conflated in the writeup: phase 2 measured
an emergent property, whether the top-1 retrieved hit happened to come
from the right source once similarity ranking settled it, with no
explicit decision anywhere in the pipeline. Phase 4 measures an explicit
upfront decision, what the router said, before retrieval even runs.
Related, not identical.

**Reported separately from abstention quality, always**, per your
instruction: any C2 abstention shortfall traceable to a routing miss
(a query where C1's ground truth says "should not abstain" but C2's own
ground truth for that same query says "should abstain" because routing
changed what got retrieved) is named explicitly in the writeup as a
routing failure, never folded silently into "the judge failed."

## 18. Generalizability caveat, carried forward from #10

Every number in this phase, the 7-miss set, the n=12 ground truth, arm
B's optimized cutoff, is anchored to one sweep cell. A different
chunk_size, ingestion_mode, retriever_strategy, or PDF variant would
plausibly yield a different miss set, a different n, and a different
optimal cutoff for B. Phase 2's Finding #1 held in all 24 cells without
exception; phase 4's findings describe this one configuration, the one
actually deployed, and the writeup must say so every time a finding is
stated, not only here.

# Phase 5 methodology: corrective retrieval + grounded-generation loop

Settled in writing before eval/run_corrective_eval.py is run against the
real API, per the same instruction phase 2 and phase 4 followed: no rule
below is chosen after seeing results. Implemented in
src/adapters/corrective_rag.py, eval/corrective_sweep.py,
eval/run_corrective_eval.py.

## 19. What this phase adds, and what it deliberately reuses unchanged

Phase 4's graph (src/adapters/agentic.py) is linear -- route, retrieve,
judge, respond -- and its judge grades only whether the *retrieved
excerpts* answer the question. It has no generation step at all, so
nothing in this repo's eval harness has ever scored a *generated* answer's
faithfulness to its context before this phase. That judge and graph are
left completely unmodified here; phase 5 is an additional, separate graph,
not a replacement.

Reused unchanged: the representative cell and its build
(eval.agentic_sweep.build_representative_cell), the 51-query set
(eval.query_set.load_query_set), the frozen SHOULD_ABSTAIN_IDS ground
truth and RETRIEVABLE_BUT_INCOMPLETE_HITS (eval.agentic_sweep), the
confusion-matrix machinery (eval.agentic_sweep.confusion_matrix), arm A as
the non-agentic baseline (eval.agentic_sweep.arm_a_matrix), the
CachedStructuredLLM disk cache (eval.llm_cache), the deterministic
gpt-4o-mini/temperature=0.0/seed=42 setting (extended to
CorrectiveAgenticConfig, src/config.py), and the route/JudgeDecision
prompt and schema (adapters.agentic.ROUTE_PROMPT/JUDGE_PROMPT/
RouteDecision/JudgeDecision) -- grade_documents in the new graph IS that
same answerability judge, only reinterpreted as a retrieval-sufficiency
grade rather than a final abstention decision.

New in this phase, built from scratch, not an extension of an existing
check: the generate node (src/adapters/corrective_rag.py's GENERATE_PROMPT,
adapted from app/app_logic.py::generate_answer's grounded-prompt approach
but run through the src/adapters/llm.py seam rather than an inline
ChatOpenAI client), the grade_generation faithfulness judge and its rubric
(#20 below), the rewrite_query node, the two conditional edges
(decide_to_generate, decide_after_generation), and the budget guard
(loop_count vs max_iterations, default 3 total retrieve+generate passes).

## 19a. The ground-truth subtlety this phase introduces: "should abstain" vs "single-shot retrieval missed it"

SHOULD_ABSTAIN_IDS (12 ids) was frozen against arm A/B/C1's single-shot,
never-rewritten retrieval (METHODOLOGY.md #11). Of those 12:

- **5 are `neither` queries** (neither-01..05): the corpus has no answer no
  matter how the question is phrased. No rewrite can change this. Correctly
  abstaining on these is a clean, phase-invariant signal, directly
  comparable to phase 4's abstention quality on the same 5.
- **7 are queries the corpus CAN answer, but the original phrasing's
  single-shot retrieval missed** (pol-08, pol-09, pol-10, faq-01, faq-05,
  faq-14, faq-16 -- eval.corrective_sweep.RESCUE_TARGET_IDS). This is
  exactly the case the corrective rewrite loop is built to try to rescue.

Because this graph can rewrite the query and re-retrieve, a non-abstain on
one of the 7 RESCUE_TARGET_IDS is not automatically a false negative the
way it would be for arm A/B/C1/C2 (none of which ever change the query) --
it may be the correction loop doing exactly what it was built for.
Consequently this phase reports **two separate numbers, never merged into
one**:

1. The full 12-id confusion matrix (eval.corrective_sweep.matrices_for_run),
   for direct side-by-side comparability with phase 4's arms.
2. **Rescue rate** (eval.corrective_sweep.rescue_rate_for_run): of the 7
   RESCUE_TARGET_IDS only, the fraction NOT abstained on -- framed
   positively (a rescue), not as a miss.

The writeup must read both together and say plainly if the 12-id matrix's
precision/recall shifts are being driven mostly by rescues on the 7 (a
genuine capability gain) versus by something else (e.g. a change in
neither-query behavior, which would be a real regression, not a rescue).

## 20. Faithfulness rubric for grade_generation, stated before any answer is generated

grade_generation asks a narrower question than phase 4's judge: not "is
this content sufficient to answer the question" (already decided by
grade_documents) and not "is this answer correct" in any absolute sense,
but **"does every factual claim in the generated answer trace back to
something actually stated in the retrieved excerpts."** A fluent,
plausible-sounding answer that adds anything the excerpts don't say --
a number, a name, a specific claim -- is "not grounded" even if the added
claim happens to be true. Verbatim rubric:
src/adapters/corrective_rag.py's FAITHFULNESS_PROMPT.

This is an LLM-judged metric, not a programmatic one (e.g. n-gram overlap
or NLI entailment scoring), for the same reason phase 4's abstention judge
was LLM-judged rather than a distance threshold: whether a claim is
"supported" is a semantic question a fixed rule can't reliably answer.
The same limitation phase 4 flagged about its own judge applies here too --
this judge can itself be wrong, and its own failure mode (see #19's
grade_generation fail-open below) is treated the same way phase 4 treated
judge-call failures: excluded from the scored matrix, reported separately,
never silently folded into a content judgment.

**Fail-open asymmetry, deliberate:** grade_documents fails open to
"sufficient" (an infra failure there is not a content judgment; proceeding
to generate is the safe default, same reasoning as phase 4's judge failing
open to "answerable"). grade_generation fails open to "not_grounded" --
the opposite direction -- because an unverifiable answer must never be
presented as grounded just because the checker itself broke. This
asymmetry is intentional, not an inconsistency; see the code comment at
grade_generation_node for the same reasoning inline.

## 21. Cost estimate, stated before any real LLM call is made

Per query, one full pass (no correction needed) costs 4 calls: route,
grade_documents, generate, grade_generation. Each additional corrective
pass (insufficient documents, or an ungrounded answer) costs one
rewrite_query call plus another grade_documents+generate+grade_generation
triplet -- 4 more calls. Worst case at max_iterations=3: 1 (route) + 3x3
(grade_documents/generate/grade_generation across 3 passes) + 2 (two
rewrites between the 3 passes) = 12 calls for one query, one run.

Most of the 51 queries are expected to resolve in pass 1 (4 calls); only
the queries whose single-shot retrieval was known to be weak (the 7
RESCUE_TARGET_IDS, plus any query where grade_generation's stricter
faithfulness bar catches a plausible-but-unsupported answer arm
A/B/C1/C2's judge would have accepted) are expected to spend a second or
third pass. **Approved estimate: an average of 6 calls/query x 51 queries
x 3 runs (eval.corrective_sweep.N_RUNS) = 918 calls**
(eval.corrective_sweep.APPROVED_CALL_ESTIMATE), hard-stopped at 1.5x
(eval.corrective_sweep.HARD_STOP_CALLS = 1377) via the same
_check_call_budget pattern as eval/run_agentic_eval.py. At gpt-4o-mini
pricing, this is materially more than phase 4's ~$0.06/459-call run
(roughly double the call count) but still well under $1; the actual
run prints its real (non-cached) call count and this estimate is not
treated as the answer -- only the printed count is.

## 22. Predictions, logged before any real LLM call is made for this phase

**Prediction 1 (rescue rate vs neither-recall):** the corrective loop will
improve non-abstention on the 7 RESCUE_TARGET_IDS (rescue rate > 0%, which
arm A/B/C1/C2 all structurally score as 0% by definition of
SHOULD_ABSTAIN_IDS) more than it changes behavior on the 5 `neither`
queries -- rewriting a query that has no true answer in the corpus is not
expected to conjure one, so neither-recall should stay comparable to phase
4's C1/C2, not improve alongside the rescue rate.

**Prediction 2 (new failure surface, honestly expected):** the
generate+grade_generation step introduces failure modes phase 4 never had
to face: expect at least some queries where generate produces a fluent
but unsupported claim and grade_generation's own judge fails to catch it
(a false "grounded"). This is not evidence the approach is broken -- it is
the expected cost of adding a generation step at all, and must be reported
as such rather than treated as a surprising negative result.

**Prediction 3 (budget is not free, expected honestly):** expect a
non-trivial corrective_fire_rate even on queries that were already
answerable on pass 1 -- an overly strict grade_documents or
grade_generation call can trigger an unnecessary rewrite that costs budget
for no recall gain. If mean_loops is high relative to the eventual
recall/precision improvement over arm A, that is a real cost to report,
not to tune away.

**Honest-negative commitment, stated now:** if the corrective loop's
12-id confusion matrix does not beat arm A/phase 4's C2 after accounting
for the rescue-rate distinction in #19a, or if its call cost is not
justified by whatever gain it does show, results/ANALYSIS.md must say so
plainly, in the same register as phase 4's "formal win condition not met"
finding -- this phase does not get a different honesty bar because it
was built more recently.
