# Phase 6 results: training-data checkpoint and Stage 4 decomposition test

Full pre-registered design, both training-data amendments, and the
Stage 4 predictions are in `docs/phase6_preregistration.md` (`#0`-`#6a`).
This document reports what was actually run, checked against those
predictions -- it does not restate the design.

## Part 1: critic training -- two attempts, both below the workable floor

Summarized from `docs/phase6_preregistration.md` `#5a`/`#5b`, evidence in
`data/critic_training/`:

| Attempt | Query construction | Positive (`needs_correction=1`) | Gate |
|---|---|---|---|
| 1 | Verbatim FAQ question text | 0 / 92 | Failed |
| 2 | LLM-paraphrased (overlap-regime-validated: mean 0.153, in the hand-authored 26's own [0.05, 0.25] band) | 6 / 92 | Failed (floor: 10) |

**Root cause (structural, not a bug):** every FAQ row is indexed as one
whole document; a query built from any text stored in that row is
near-guaranteed self-retrieval in a 126-document corpus at k=5, regardless
of how the query is phrased -- verbatim text hits 92/92, and even
genuinely-paraphrased text (validated against the same lexical-overlap
regime the hand-authored eval queries occupy) only breaks that 6 times.

**What this finding is, and is not (`#5b`, restated because it is the
load-bearing caveat for everything below):** two independent, disciplined
attempts to build a critic-training pool from this specific 126-document
FAQ corpus both fell below a workable size. That is a finding about
**retrieval saturation on this corpus**, not a finding about whether a
trained critic (cross-encoder or LoRA) can beat the phase-5 prompted
grader -- **no critic was ever trained**, so that question is untested
here, not falsified. CE and LoRA are **deferred to a future,
separately-scoped corpus** with a non-saturated retrieval task, not
cancelled. No model weights exist in this repo as a result.

## Part 2: Stage 4 -- multi-agent decomposition vs. the phase-5 monolith, Baseline critic only

Per `#5b`'s scope change, Stage 4 tests only the *decomposition* question
(does splitting the loop into named agents change call-count/quality/
latency), not the trained-critic question. `src/adapters/corrective_rag.py`
is unmodified (`git diff main...phase6-multiagent-critic -- src/adapters/corrective_rag.py`
is empty) -- both graphs below are the real production code, run fresh,
back-to-back, uncached, over the same 51-query set and representative
cell, in the same session (`eval/run_multiagent_critic_eval.py`, real
run, `results/multiagent_critic_eval_results.json`).

**490 total real (non-cached) OpenAI calls**, under the 918 hard stop
(approved estimate 612, `#6a`). Zero fail-opens on either graph.

### Predictions checked, one at a time (`#6a`)

**Prediction 1 (quality: matches within phase 5's own already-documented
API-nondeterminism noise) -- held.** 49 of 51 per-query decisions
(`abstained`, `grounded`) agree exactly between the two graphs (96.1%).
The 2 disagreements (`faq-22`, `neither-03`) are consistent with the same
phenomenon `results/ANALYSIS.md` already documented for this exact
prompt/model (`grade_documents` returning a different verdict on a
nominally identical call across runs, e.g. `pol-08`/`faq-14` between
phases 4 and 5) -- not a new regression this decomposition introduced.
Rescue rate (5/7 mechanical), faithfulness rate (1.0), and corrective
fire rate (0.333) are **identical** between the two graphs; mean loops
are within 0.02 (1.588 old, 1.569 new). The scored confusion matrix
shifts by exactly the 2 mismatched queries (old: tp=4/fp=8/fn=8/tn=31;
new: tp=5/fp=7/fn=7/tn=32) -- both land in the same range phase 5's own
3-run spread already occupies, not outside it.

**Prediction 2 (call count: near-identical, same five call sites) --
held.** Old graph: 248 calls. New graph: 242 calls. Delta: **-6**, i.e.
the multi-agent graph made *fewer* calls, within the "a few calls"
band `#6a` predicted before either graph ran.

**Prediction 3 (latency: equal-or-worse, never meaningfully better) --
NOT confirmed as stated.** Measured: old graph 352.7s wall-clock
(1.406s mean per-call), new graph 298.8s (1.220s mean per-call) -- the
decomposed graph was **15.3% faster** (-54.0s), not equal-or-worse. This
is reported plainly, not reframed as a near-miss: the prediction was
wrong as measured.

**Read carefully, not spun either direction.** The 6-fewer-calls
difference explains only part of the gap proportionally (242/248 is 2.4%
fewer calls, not 15.3% less time) -- per-call latency itself was also
lower in the new graph's run (1.220s vs. 1.406s, 13.2% faster per call),
which call-count alone does not explain. Two explanations are both live
and this single comparison cannot distinguish them: (a) a genuine
structural effect (5 LangGraph node-transitions vs. 8, or some other
difference between the graphs), or (b) ordinary API/network timing
variance between two sequential ~5-6 minute runs in the same session --
the old graph ran *first*, so any warm-up, connection reuse, or
time-of-day server-load effect would bias against it regardless of graph
structure. **This was a single run per graph (`#6a`'s own design, a
structural/timing comparison, not a repeat of phase 5's 3-run
variance study) -- it is evidence, not proof, and should not be read as
"decomposition makes this faster" without a repeat run** (ideally with
run order swapped or randomized) to separate the two explanations. Filed
as an open question, not resolved here.

### Table: decomposition call-count and latency

| | Phase 5 monolith (8 nodes) | Phase 6 multi-agent (5 nodes, Baseline critic) | Delta |
|---|---|---|---|
| Real LLM calls (51 queries, 1 pass) | 248 | 242 | -6 |
| Wall-clock | 352.7s | 298.8s | -54.0s (-15.3%) |
| Mean per-call latency | 1.406s | 1.220s | -0.186s |
| Scored confusion matrix (of 51) | tp=4 fp=8 fn=8 tn=31 | tp=5 fp=7 fn=7 tn=32 | 2 queries |
| Rescue rate (of 7) | 5/7 (0.714) | 5/7 (0.714) | 0 |
| Faithfulness rate | 1.0 | 1.0 | 0 |
| Corrective fire rate | 0.333 | 0.333 | 0 |
| Mean loops | 1.588 | 1.569 | -0.02 |
| Fail-opens | 0 | 0 | 0 |

### Honest negatives, stated together

1. **The trained-critic question (this phase's original central
   question) is untested, not answered.** Two labeling attempts on this
   repo's FAQ corpus both fell below a workable training-pool size --
   `#5a`/`#5b` are a methodology-and-corpus finding, not a critic-quality
   finding, and must not be cited as either.
2. **The latency prediction was wrong as stated**, and the honest
   replacement claim is narrower than "decomposition is faster": the
   measured 15.3% latency improvement is real for this one comparison run
   but confounded with run order and cannot yet be attributed to the
   architecture change with confidence.
3. **Quality is not byte-identical** between the two graphs (49/51, not
   51/51) -- consistent with, not distinguishable from, this exact
   prompt/model's already-documented best-effort (not guaranteed)
   determinism, but stated as a measured fact, not assumed away.
4. **This is a single comparison run, not a 3-run variance study** --
   `#6a` pre-registered this scope deliberately (a structural/timing
   comparison, not a re-litigation of phase 5's own already-settled
   API-variance question), but it means neither the latency nor the
   2-query quality delta above has an error bar.

**Net read:** the multi-agent decomposition reproduces phase 5's
decision quality within the same noise band phase 5 itself already
documents, at a near-identical (very slightly lower) call count, with a
first-look latency advantage that is real in this run but not yet
separable from ordinary timing variance. The swappable critic seam this
architecture buys (`#6`) is real and unused this phase; the trained
critic it was built for remains a follow-on, pending a corpus where the
training pool measured in `#5a`/`#5b` would not saturate immediately.
