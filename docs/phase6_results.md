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
NOT confirmed as stated in the initial single run; the effect itself did
not replicate under paired testing.** Initial measurement: old graph
352.7s wall-clock (1.406s mean per-call), new graph 298.8s (1.220s mean
per-call) -- the decomposed graph was **15.3% faster** (-54.0s), not
equal-or-worse, reported plainly rather than reframed as a near-miss.
That single run had the old graph go first, an acknowledged confound
(`#6a`), so it did not settle whether "faster" tracked graph identity or
run order.

**Paired follow-up (4 runs, alternating which graph goes first --
`eval/run_paired_timing_runs.py`, `results/multiagent_critic_paired_timing_runs.json`,
1,936 additional real calls): the 15.3% gap dissolves, it does not
replicate.**

| Run | Order | Old wall-clock | New wall-clock | Delta (new - old) |
|---|---|---|---|---|
| 0 | old first | 296.0s | 299.9s | +3.8s |
| 1 | new first | 295.3s | 299.1s | +3.8s |
| 2 | old first | 314.0s | 295.9s | -18.1s |
| 3 | new first | 321.9s | 296.4s | -25.4s |

Identity effect (new graph's time minus old's, regardless of which ran
first): +3.8s, +3.8s, -18.1s, -25.4s -- **sign flips twice**, mean -9.0s
(~2.9%, roughly a third of the original 15.3%). Position effect
(whichever graph ran first minus whichever ran second, regardless of
identity): -3.8s, +3.8s, +18.1s, -25.4s -- also sign-flipping, mean
-1.8s. **Neither a consistent graph-identity pattern nor a consistent
run-order pattern emerged across 4 runs.** The original -54.0s/-15.3%
result was itself one draw from this same noisy distribution, not a
stable effect -- at n=4 per condition the honest conclusion is that
run-to-run timing variance (most plausibly ordinary API/network
variance, though this data cannot fully confirm even that) exceeds any
structural signal from the architecture change, not that the
architecture reliably speeds anything up or down. Cause of the
run-to-run variance itself is **unresolved**, filed as an open question,
not chased further here (`#6a`'s own scope: a structural/timing
comparison, not a full benchmark harness).

**Prediction accounting, one line:** 2 of 3 held (quality, call count);
1 missed favorably in its first measurement and did not hold up under
repeat testing (latency: initially 15.3% faster, dissolved to a
sign-flipping ~2.9% mean under paired runs) -- cause of the run-to-run
variance unresolved.

### Table: decomposition call-count and latency (single comparison run)

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
2. **The latency prediction was wrong in its first measurement, and the
   effect it found does not replicate.** The initial 15.3% latency
   improvement looked real for one comparison run; 4 paired runs with
   alternating order show a sign-flipping, much smaller mean delta
   (~2.9%) with no consistent identity or position pattern -- the honest
   claim is "no reliable latency effect detected, in either direction,"
   not "decomposition is faster."
3. **Quality is not byte-identical** between the two graphs (49/51, not
   51/51) -- consistent with, not distinguishable from, this exact
   prompt/model's already-documented best-effort (not guaranteed)
   determinism, but stated as a measured fact, not assumed away.
4. **The quality/call-count comparison is still a single run, not a
   3-run variance study** -- `#6a` pre-registered this scope
   deliberately, so the 2-query quality delta and the 6-call count delta
   still have no error bar, even though latency was checked further.

**Net read:** the multi-agent decomposition reproduces phase 5's
decision quality within the same noise band phase 5 itself already
documents, at a near-identical (very slightly lower) call count, and --
after the paired follow-up -- no reliably measurable latency difference
in either direction. The swappable critic seam this architecture buys
(`#6`) is real and unused this phase; the trained critic it was built for
remains a follow-on, pending a corpus where the training pool measured in
`#5a`/`#5b` would not saturate immediately.
