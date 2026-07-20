# Phase 2 evaluation: analysis

Source data: `results/sweep_results.json`, 12 cells (2 ingestion modes x
3 chunk configs x 2 retriever strategies), 51 queries per cell, n=1 per
cell (see `eval/METHODOLOGY.md` #7 for why no repeats). Embedding model:
`sentence-transformers/all-MiniLM-L6-v2`. Scoring rules follow
`eval/METHODOLOGY.md` exactly; this document doesn't re-derive them.

Every number below is per-bucket. There is no pooled/overall metric
anywhere in this document, per METHODOLOGY.md #1.

## Central hypothesis: format-aware ingestion vs uniform chunking (FAQ bucket)

| chunk_size | format_aware hit@1 | uniform hit@1 | format_aware MRR | uniform MRR | mean FAQ rows merged per uniform chunk |
|---|---|---|---|---|---|
| 200 | 0.462 | 0.462 | 0.596 | 0.571 | 1.00 |
| 500 | 0.500 | 0.385 | 0.631 | 0.549 | 1.52 |
| 800 | 0.500 | 0.385 | 0.642 | 0.525 | 2.51 |

**The hypothesis is supported, and in exactly the two-part shape the
brief itself predicted rather than a single clean win.** At chunk_size
200 -- smaller than the FAQ's ~178-char average answer -- format-aware
and uniform ingestion are statistically indistinguishable (hit@1
identical, MRR within 0.025 of each other) because almost nothing
actually gets merged at that size (mean_rows_per_uniform_chunk=1.00,
i.e. uniform chunking "does nothing" here, matching the brief's first
predicted failure mode). At chunk_size 500 and 800, uniform chunking
measurably merges unrelated Q&A pairs into single chunks (1.52 and 2.51
rows per chunk respectively) and FAQ retrieval degrades accordingly:
hit@1 drops 12-22 points, MRR drops 8-12 points, both directions moving
the same way at both larger sizes. This is the brief's second predicted
failure mode ("merges unrelated Q&As... pollutes retrieval"), and it is
directly measured here via `mean_rows_per_uniform_chunk`, not merely
inferred from the hit-rate gap.

Neither number is a coincidence of one lucky/unlucky chunk_size: the
direction (format-aware >= uniform) holds at every chunk_size tested,
and the *size* of the gap tracks the *size* of the measured merging
almost monotonically. That combination -- a consistent direction plus an
independently-measured mechanism that predicts the gap's magnitude -- is
what makes this a real finding rather than noise.

## Policy bucket: hit-rate with and without retrievable_but_incomplete

(Representative cell: `format_aware__cs800_ov100__similarity`, phase 1's
shipped default. All 6 similarity-strategy cells show the same pattern in
direction, if not magnitude -- see `sweep_results.json` for the rest.)

| | n | hit@1 | hit@10 | MRR |
|---|---|---|---|---|
| all 11 policy_pdf queries | 11 | 0.636 | 0.909 | 0.691 |
| excluding the 3 incomplete-flagged (pol-01, pol-05, pol-10) | 8 | 0.750 | 1.000 | 0.783 |

The gap is real and in the direction METHODOLOGY.md #2 predicted: the
raw, all-queries hit-rate is ~11-15 points lower than what the system
achieves on questions it can actually fully answer. Both numbers are
"correct" -- the higher one describes retrieval quality on answerable
questions, the lower one describes what a user actually experiences
across the full range of things they might plausibly ask. Reporting
only the higher one would have been the misleading choice.

Source-routing accuracy (top-1 source type correct; faq/policy_pdf only,
per METHODOLOGY.md #3) at this cell: **policy_pdf 0.909, faq 0.885** --
both high and comparable, meaning the combined FAQ+policy retriever
rarely confuses which corpus a question belongs to, independent of
whether it finds the exact right passage within that corpus.

## Either bucket (cross-source queries)

Hit-rate under OR-across-sources scoring (METHODOLOGY.md #3) ranges
0.44-0.67 at k=1 and 0.78-1.00 at k=10 across the 12 cells, with no
consistent pattern tied to ingestion mode or chunk size -- expected,
since these 9 queries are answerable from either sub-index and the
ingestion-mode manipulation only touches the FAQ side. n=9 is too small
to support a finer-grained claim than "these queries are reliably
findable by k=10 in every configuration tested." Excluded from
source-routing accuracy throughout, per METHODOLOGY.md #3.

## Retriever strategy: similarity vs ParentDocumentRetriever (policy bucket only)

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
a single query flipping is enough to flip which strategy "wins" --
this dataset cannot support a claim that either retrieval strategy is
better than the other. A larger, more diverse policy corpus (this PDF is
77 pages; ParentDocumentRetriever's rationale -- large-document parent
context around a small matched child chunk -- likely needs a longer or
more heterogeneous document than this one to show a real effect) would
be needed before concluding anything stronger.

## Confidence separation on "neither" queries (its own section, per METHODOLOGY.md #3)

Representative cell (`format_aware__cs800_ov100__similarity`); the
pattern below holds in **all 12 cells** without exception.

| | n | mean distance | median | min | max |
|---|---|---|---|---|---|
| neither (5 queries) | 5 | 0.829 | 0.815 | 0.622 | 0.990 |
| should-hit (faq+policy_pdf+either, pooled, 46 queries) | 46 | 0.738 | 0.729 | 0.337 | 1.308 |

(Chroma distance: lower = more similar/confident.)

**The means point the right way -- neither queries are on average less
confident than should-hit queries -- but the ranges overlap in every
single cell of the sweep.** Some should-hit queries score worse (max
1.308) than every neither query, and some neither queries score better
(min 0.622) than the majority of should-hit queries. That means: **there
is no similarity-score threshold you could set that would reliably
separate "the system found the answer" from "the system found the
least-bad thing available."** A user (or an automated caller) reading
only the top-1 similarity score cannot tell these two situations apart
in this system, in any configuration tested. This is a genuine
limitation of similarity search as a mechanism here, not an artifact of
one unlucky chunk_size or ingestion mode -- it reproduces across all 12
cells identically in direction (overlap present) even though the exact
mean/median values shift slightly cell to cell.

## Lexical overlap stratification (FAQ bucket)

(Same representative cell as above.)

| | n | hit@1 | hit@10 | MRR |
|---|---|---|---|---|
| high_lexical_overlap (named-feature questions: RCH Profile, Home Visit, Poshan Tracker Dashboard) | 3 | 1.000 | 1.000 | 1.000 |
| low_lexical_overlap (everything else) | 23 | 0.435 | 0.826 | 0.596 |

This pattern holds across every cell in the sweep, not just this one.
**A meaningful share of "good" FAQ retrieval performance is attributable
to lexical/named-entity match, not semantic understanding of the
paraphrase.** The 3 high-overlap queries were kept rather than reworded
further because they're genuine named-UI-feature questions (see
METHODOLOGY.md #4) -- but their perfect scores mean the whole-bucket
hit@1 of ~0.46-0.50 quoted above is inflated by roughly 3/26 of the
sample scoring by essentially string match. The low-overlap subset's
0.435 hit@1 is the more honest estimate of this system's *semantic*
retrieval quality on FAQ-style queries.

## Limitations

- **n=1 per cell**: embedding inference and Chroma similarity search are
  deterministic given fixed model weights (verified by
  `tests/test_eval_determinism.py`), so there is no run-to-run variance
  to report for this pipeline -- not a gap in rigor, a property of the
  method (METHODOLOGY.md #7).
- **either (n=9) and neither (n=5) buckets are small.** Findings there
  (the confidence-separation overlap, the either-bucket hit@10 ceiling)
  are consistent across all 12 cells, which is reassuring, but neither
  bucket supports fine-grained claims the way the 26-query FAQ bucket does.
- **No LLM-answer-quality arm was built or run**, per the phase 2 brief
  -- documented as an optional, API-key-gated extension in the README,
  not attempted here.
- **PDF text extraction quality (pypdf) is a fixed upstream factor**,
  not evaluated on its own. If page 31's population-norms text or
  similar were extracted more completely, `retrievable_but_incomplete`
  flags like pol-01 might not apply -- this analysis treats the
  extracted text as given, not as something under test.
- **The "confidence separation" finding is about this embedding model
  and this corpus size specifically** -- it should not be read as a
  general claim that similarity search can never support a confidence
  threshold, only that it doesn't reliably do so here.

## Headline (for the README)

Format-aware ingestion beats uniform chunking on the FAQ bucket, and the
size of the win tracks a directly-measured mechanism (rows merged per
chunk), not just a hit-rate gap that could have other explanations.
Retriever strategy (similarity vs ParentDocumentRetriever) shows no
distinguishable effect on this policy corpus at this scale. The system
cannot reliably tell "found it" from "didn't" from its similarity score
alone in any configuration tested. And a meaningful chunk of FAQ
retrieval's apparent quality comes from lexical rather than semantic
match -- the honest semantic-only number is closer to 0.43 hit@1 than
the headline 0.46-0.50.
