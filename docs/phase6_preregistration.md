# Phase 6 pre-registration: multi-agent corrective RAG with a fine-tuned critic

Written and committed **before any Phase 6 code that produces a number
exists** -- no training data has been extracted, no critic has been
trained, no graph refactor has been written. Same discipline as
`eval/METHODOLOGY.md` (settled in writing before any run): everything
below is a commitment, not a summary of results. Phase 5
(`src/adapters/corrective_rag.py`, `eval/METHODOLOGY.md` #19-22,
`results/ANALYSIS.md`'s Phase 5 section) is extended, not forked or
modified -- that module and its tests stay frozen throughout Phase 6.

## 0. Provenance: two corrections to the originating task brief

The task brief that scoped this phase assumed phase 5's "545" was a pool
of 545 labeled `(query, retrieved_context)` rows available for critic
training. Checked against the repo before writing anything else, that
assumption doesn't hold, for two independent reasons:

**0a. "545" is a call count across five node types, not a labeled-row
pool.** `results/corrective_eval_results.json`'s `real_llm_calls_made`
provenance note and `results/ANALYSIS.md` ("545 real (non-cached) OpenAI
calls... route/grade_documents/rewrite_query/generate/grade_generation")
confirm 545 is the sum of calls to all five LLM call sites across 3 runs
of the 51-query set. Counting only `grade_documents` -- the call site a
critic would actually replace -- gives 80 calls/run x 3 runs = **240**,
verified directly against `corrective_eval_results.json`'s
`corrective_runs[].per_query[].trace` (each run: 80 `grade_documents`
entries, 39 graded `insufficient`, identical across all 3 runs). The 3
runs are near-duplicates of the same 51 base queries (`results/ANALYSIS.md`
already documents this: "zero measured variance across 3 runs, on the
aggregate"), not 3x independent samples.

**0b. Holding out the eval set leaves zero training rows.** The split
rule this document requires (see #4) forbids any of the 51 eval queries,
or rows associated with them, from entering critic training. Every one of
phase 5's 545/240 calls is *on* those same 51 queries -- there is no
disjoint subset of phase 5's own run left to train on. Phase 5's calls are
therefore treated in this document as **analysis and context** (they are
what motivates the whole phase -- see the Context paragraph below), never
as a training-data source.

**Resolution (see #5):** a real, disjoint, mechanically-labeled training
pool exists elsewhere in the repo's own corpus -- 92 of the 126 FAQ rows
were never used as ground truth by any of the 51 eval queries
(`eval/queries.yaml`, checked directly: 34 distinct `faq_index` values are
referenced as `ground_truth` across all 51 entries; 126 - 34 = 92 unused).
This document builds the training pool from those 92 rows, not from phase
5's calls.

## Context: what motivates this phase

Phase 5's own findings (`results/ANALYSIS.md`) are the reason this phase
exists. Of 51 queries, `grade_documents` correctly triggers correction on
6 of the 12 queries that actually needed it, but also fires on 11 of the
39 queries that didn't -- and 7 of those 11 unnecessary firings then
exhaust the retry budget and wrongly abstain, converting an
already-answerable query into a false refusal. This phase asks: does
replacing the inline prompted `grade_documents` with a *trained* critic
(a cross-encoder or a LoRA-adapted small LLM) reduce that
over-firing, without giving up the genuine rescues the loop does
deliver? And separately: does splitting the monolithic loop into named
LangGraph agents change call-count or latency, for better or worse?

## 1. Three arms (a config-sweep axis, not three separate builds)

**Status (see #5b, dated 2026-08-09): CE and LoRA are deferred, not
built, this phase.** The design below is the original three-arm axis as
pre-registered; #5b records why Stage 4 executes Baseline-only and where
CE/LoRA move to. Left as originally written, not rewritten, so the
record shows both what was planned and what the training-data checkpoint
actually supported.

| Arm | Critic mechanism | Inference cost |
|---|---|---|
| **Baseline** | phase-5 prompted inline self-grade (`JUDGE_PROMPT`/`JudgeDecision`, frozen, unchanged) | LLM call |
| **CE** | fine-tuned cross-encoder relevance/sufficiency gate | keyless, CPU |
| **LoRA** | LoRA/PEFT small generative LLM as grader | GPU-trained, local-served |

All three run through one graph (#6), selected by a `critic:
"baseline" \| "ce" \| "lora"` config field -- consistent with how this
repo has always swept configuration (`eval/METHODOLOGY.md` #6's sweep
grid, `src/config.py`'s per-phase config dataclasses), not a reason to
fork the graph three times.

## 2. Win condition and primary metric (pin one number, not a loose "firing rate")

**Status (see #5b): not evaluated this phase.** The win condition and
curve below apply once a trained critic exists to compare against
Baseline; #5b defers that to a future corpus. Left as originally
pre-registered, not rewritten, as the standard this repo's eventual
trained-critic follow-on is still held to.

**Confusion table over the 51 queries, under phase 5, verified directly
against `results/corrective_eval_results.json` (run 0; identical on runs
1 and 2 -- checked, not assumed):**

| | needed correction (12, `SHOULD_ABSTAIN_IDS`) | didn't need it (39, arm-A-answerable) |
|---|---|---|
| **corrective loop fired** | 6 (`faq-01`, `faq-16`, `neither-01`, `neither-03`, `neither-05`, `pol-10`) | 11 (`either-02`, `either-03`, `either-08`, `faq-06`, `faq-07`, `faq-09`, `faq-12`, `faq-22`, `pol-01`, `pol-05`, `pol-06`) |
| **corrective loop did not fire** | 6 | 28 |

Of the 11 unnecessary firings, 7 (`either-03`, `either-08`, `faq-06`,
`faq-07`, `faq-09`, `faq-12`, `pol-01`) then exhausted the retry budget
and wrongly abstained; the other 4 recovered and answered anyway at the
cost of extra calls. These figures reproduce `results/ANALYSIS.md`'s
"11 (65%) unnecessary... 7 (64%) then exhausted the full budget"
independently from the raw per-query records, not copied from the prose.

**Primary metric: unnecessary firings** -- corrective-loop activations on
queries that didn't need correction (phase 5: 11 of 39, i.e. of the
already-answerable queries). A critic wins on this metric by firing less
on the 39-query "didn't need it" column specifically, not by firing less
overall -- a critic that fires less everywhere, including on the 12
queries that do need it, is not a win and must not be reported as one.

**Constraint (not optional, not a secondary tradeoff to trade away):**
genuine rescue rate must stay **>= 57.1%** (4 of the 7 `RESCUE_TARGET_IDS`
-- `results/ANALYSIS.md`'s *genuine content-rescue rate*, not the
mechanical 71.4% / 5-of-7 figure, which counts `faq-16`'s honest
non-answer as a rescue when it is not; see that document's "the abstained
flag undercounts real non-answers"). A critic that cuts unnecessary
firings by suppressing genuine corrections too fails the win condition
even if the primary-metric number looks better in isolation -- this floor
exists specifically to catch that failure mode.

**Secondary metric, reported alongside, never substituted for the
primary:** wrong-abstention count (phase 5: 7 of the 11 unnecessary
firings). A critic could plausibly cut unnecessary firings without
proportionally cutting wrong abstentions (e.g. if the firings it
eliminates are disproportionately ones that would have self-corrected
anyway) -- report this number so that gap is visible, not folded into the
primary metric.

**Tradeoff curve.** Reported as a curve with **three series**, not a
point: CE and LoRA each expose a genuine tunable decision threshold
(cross-encoder score cutoff; LoRA grader confidence, if the trained model
exposes one, else a discrete threshold over repeated-sample voting), so
each sweeps a real firing-rate/rescue-rate curve across threshold values.
**Baseline has no tunable threshold** (a single prompted classification
call, not a scored one) and is therefore plotted as a **single reference
point/line**, not a swept series -- stated here explicitly so its absence
of a curve is not later misread as a missing result.

## 3. Pre-registered null predictions (ship these even if they hold)

- **The training pool is small and imbalanced in absolute terms, not just
  by percentage.** Per #5's estimate, ~92 rows total, projected ~16
  positive (`needs_correction=1`) / ~76 negative. A stratified 80/20
  split (#5) puts on the order of **3 positive examples in validation**.
  Either trained critic -- CE or LoRA -- may fail to beat the prompt on
  this little signal, and LoRA (a generative model fine-tuned on ~13
  positive training rows) is the more likely of the two to fail. That is
  a publishable result, not a reason to abandon the arm or quietly drop
  its numbers.
- **Multi-agent decomposition may not beat the single monolithic loop on
  this corpus.** Splitting `corrective_rag.py`'s one graph into named
  retrieval/critic/generator agents changes code structure, not
  necessarily behavior -- report call-count and latency deltas either
  way, not just when they favor decomposition.
- **The training label and the eval target are related, not identical,
  constructs.** Training labels (#5) are retrieval-*sufficiency*
  (`hit@k`) on FAQ-verbatim, individually answerable queries. The eval
  target is *unnecessary-firing / rescue* on the heterogeneous 51-query
  set, which includes unanswerable (`neither`) queries and `policy_pdf`
  queries the critic never saw a training example of. A null result may
  reflect this construct gap rather than "fine-tuning doesn't help" --
  both explanations must be considered before concluding either.

## 4. Split rule (methodological integrity -- a hard rule, not a preference)

The 51-query eval set is held out from all critic training, **by
construction**: the training pool (#5) is drawn only from the 92 FAQ rows
that are never `ground_truth` for any of the 51 eval queries, so there is
no eval/train overlap to accidentally introduce. No eval query, and no
`(query, retrieved_context)` row associated with one, may enter CE or
LoRA training data at any point. Leakage here would inflate
firing-rate improvement across all three arms and make the win condition
measure memorization of the eval set, not a better gate.

**Retriever pinning.** The retriever configuration used to build the
92-row labels (embedding model, `k`, index build) must be the *literal
same* configuration the multi-agent graph's `retrieve` node uses when
evaluated: `eval.agentic_sweep.K = 5`, the same `CELL_ID` cell
(`build_representative_cell`, format-aware PDF variant, `chunk_size=800`
/ `chunk_overlap=100`, similarity retriever) phases 4 and 5 already use.
A mismatched retriever config at label-generation time would calibrate
the critic against a different retrieval distribution than the one it is
graded on at eval time -- this is stated as a requirement, and Stage 2's
extraction script must import `eval.agentic_sweep.build_representative_cell`
directly rather than reconstruct the cell, so this can't drift by
accident.

## 5. Training data: mechanically auto-labeled, no hand-labeling

**Rule.** For each of the 92 unused FAQ rows: use the FAQ's own question
text, verbatim (not reworded -- reworded pairing is what `eval/queries.yaml`
already calls `faq_reworded`, a distinct and deliberately-not-reused
origin), as the query. Retrieve at the pinned config (#4). Label via the
existing `hit_at_k` logic (`eval/metrics.py`, unmodified, the same
function phase 2's hit-rate curve was built on) against that row's own
`faq_index` as ground truth:

```
hit_at_k(hits, [{"source": "faq", "faq_index": N}])
    True  -> retrieval sufficient   -> needs_correction = 0
    False -> retrieval insufficient -> needs_correction = 1
```

Zero hand-labeling: the label is a deterministic function of the retriever
output and a fact already recorded in `data/faq/all_faq.json` (which row
this question came from), not a judgment call.

**Location and clearance.** `data/critic_training/` -- confirmed cleared
for public commit (this repo's docs and existing `eval/queries.yaml`
ground-truth are already public; this dataset is the same public FAQ
corpus, mechanically re-queried, no new sensitive content).

**Scope decision: FAQ-only, not hand-expanded to policy_pdf.**
Hand-authoring new `policy_pdf`-bucket queries to balance the pool across
source types was considered and rejected for this stage: hand-authoring
reintroduces exactly the labeling subjectivity the mechanical FAQ rule is
built to avoid, for a training-set-size problem that hand-authoring a
handful of policy queries wouldn't meaningfully fix anyway (61 unused PDF
pages have no existing verbatim-question source the way FAQ rows do, so
"mechanical" and "policy_pdf" don't compose the same way "mechanical" and
"FAQ" do). This is deferred as an explicit fallback, only to be reached
for if the 92-row FAQ pool proves unworkably thin once Stage 2 actually
measures it -- not a decision this document treats as closed forever.

**Consequence for the null predictions (#3):** the trained critics will
never see a `policy_pdf`-sourced or `neither`-type training example. Their
performance on the 51-query eval set's `policy_pdf` and `neither` buckets
is a generalization test, not an in-distribution one, and results must say
so explicitly rather than presenting one blended number across buckets
that were not equally represented in training.

**Expected class balance (an estimate, stated as one, not measured
yet).** Applying phase 2's own hit@5 rate (0.826 hit / 0.174 miss --
`results/ANALYSIS.md`'s honest semantic-only FAQ hit-rate curve) to 92
rows projects **~16 positive (`needs_correction=1`) / ~76 negative**.
This is a projection from a different query distribution (phase 2's
hit-rate curve was computed on some mix of original/reworded FAQ
questions, not necessarily this exact 92-row subset) and Stage 2 must
report the *actual* measured count from running retrieval on these
specific 92 rows, not this estimate, once it exists.

**Split.** Seeded, **stratified by label**, not a plain random split. At
an estimated ~16 positive rows total, a random split risks a validation
fold with zero or one positive example, which would make any val-set
metric (precision/recall on the positive class, the one that matters for
this critic) meaningless. The split must guarantee a minimum positive
count in validation -- `floor(0.2 x n_positive)` at a nominal 80/20 split
-- computed and logged exactly once Stage 2 runs, not assumed here.

## 5a. Amendment -- 2026-08-09: verbatim self-query labeling measured 0/92 positive, corrected to LLM-paraphrased queries

**What was pre-registered above (#5) did not work, measured directly, not
assumed.** Stage 2 ran the retriever spot-check first (passed cleanly:
`faq-05` reproduced its frozen phase-5 miss exactly; the full 51-query
drift check against `SHOULD_ABSTAIN_IDS` passed on all 51 --
`training/extract_critic_data.py::spot_check_retriever`), then ran the
mechanical `hit@k` labeling exactly as specified above. Result: **0 of 92
positive** (`needs_correction=1`), not the ~16 projected. hit@5 = 92/92
(100%), hit@1 (exact top-1 match) = 89/92 (96.7%), mean top-1 distance
0.359. A second mechanical variant was tried as a diagnostic (not
committed, not part of the pipeline): querying with each row's own
**answer** text instead of its question -- same result, 0/92 positive,
mean top-1 distance 0.280.

**Root cause, structural, not a bug or a bad draw.** FAQ rows are indexed
as one whole Document each (`split=False`, `src/config.py`'s
`IngestionConfig`) -- question and answer together, one chunk per row. A
query built from *any text stored in that same row* is therefore
near-guaranteed to retrieve that row at k=5 in a 126-document corpus: this
is self-retrieval, not a difficulty test, regardless of which field
supplies the query text. The ~16/92 estimate in #5 projected phase 2's
hit@5=0.826 onto this pool, but that figure was measured on the 51-query
eval set's hand-paraphrased (`faq_reworded`) and `policy_derived` queries
-- independently phrased, not verbatim excerpts of their own gold
document. Applying a hit rate measured on independently-phrased queries to
a pool of verbatim-own-text queries compared two structurally different
retrieval tasks; that comparison was wrong, and is corrected here rather
than carried forward.

A reusable mechanical paraphrase generator was checked for before writing
a new one: none exists. `eval/METHODOLOGY.md` #4's 26 `faq_reworded` eval
queries were **hand-authored**; that section's `jaccard_overlap` machinery
only measures and stratifies lexical overlap after the fact, it does not
generate rewordings.

**Amendment: one LLM-generated paraphrase per unused FAQ row**, replacing
verbatim question text as the query, with the disjointness guarantee
(#4) and the 92-row FAQ-only scope decision (above) both unchanged.
Fixed prompt (`training/generate_faq_paraphrases.py::PARAPHRASE_PROMPT`),
`gpt-4o-mini`, `temperature=0.0`, `seed=42` -- the same deterministic
convention as every other LLM call in this repo (`AgenticConfig`,
`CorrectiveAgenticConfig`), cached via `eval.llm_cache.CachedStructuredLLM`
the same way phase 4/5's real calls are, real call count and cost printed
before the run is treated as final, same discipline as
`eval/run_corrective_eval.py`. Approved estimate: 92 calls (one per
unused row), hard stop 138 (1.5x, matching `eval/agentic_sweep.py` and
`eval/corrective_sweep.py`'s own convention).

**Stated plainly: this is a partial retreat from #5's "zero hand-labeling,
zero LLM calls" framing**, not a silent one. Training-*label* generation
now touches an API key, once, offline, to generate 92 short paraphrases --
this is categorically different from hand-labeling `needs_correction`
values (still zero human judgment calls on the labels themselves: `hit@k`
against the row's own `faq_index` still produces the label mechanically,
only the *query text* is now LLM-paraphrased rather than verbatim). CE's
own inference at eval/serving time and the deployed Streamlit demo remain
fully keyless -- #9's deployment invariant is unaffected; this touches
offline training-data generation only.

**Acceptance gate, stated before any paraphrase is generated (not fit to
the results afterward):**

1. Mean Jaccard overlap (`eval.metrics.jaccard_overlap`, unchanged) across
   the 92 (paraphrase, original-question) pairs must fall in **[0.05,
   0.25]** -- the hand-authored 26's own regime (mean 0.134, median
   0.101), with headroom on both sides for a differently-sized, LLM- not
   human-authored sample.
2. **No more than 14 of the 92** (~15%) may exceed 0.5 overlap -- the hand
   set's high-overlap tail was 3/26 (11.5%), rounded up generously for a
   smaller/noisier automated sample.
3. The resulting `hit@k` relabeling, run immediately after the gate
   passes, must yield **at least 10 positive examples** -- passing the
   overlap-regime check alone is necessary but not sufficient; the pool
   must actually be usable, not just lexically dissimilar in aggregate.

**If any check fails: stop.** Do not re-run the paraphrase generator with
adjusted prompt wording to try to pass the gate -- tuning generation
parameters until a precommitted check passes is exactly the kind of
post-hoc parameter search `eval/METHODOLOGY.md` #8a's "decision rule
stated before the numbers were looked at" discipline exists to rule out.
A gate failure's next step is hand-authored paraphrasing (the "hand-author
paraphrases" alternative raised alongside this amendment), reported and
reviewed before proceeding, not a second automated attempt.

**Consequence for the null predictions (#3), sharpened:** the positive
class, if the gate passes, consists entirely of LLM-paraphrase-induced
misses on FAQ-sourced, individually-answerable questions -- an even
narrower slice of the eval set's actual query distribution than #5
originally described (which at least imagined naturally-occurring misses
on unmodified FAQ text). Any result must be read with this narrower
provenance in mind, not just the general train/eval construct gap #3
already names.

## 5b. Amendment -- 2026-08-09: paraphrase gate 1 passed, gate 2 failed (6/92) -- CE/LoRA training deferred, not attempted a third way

`training/generate_faq_paraphrases.py` ran as #5a specified: 92 real
`gpt-4o-mini` calls (well under the 138 hard stop), fixed prompt,
`temperature=0.0`, `seed=42`, cached in `eval/.llm_cache.json`.

**Gate 1 (Jaccard-overlap regime) passed.** Mean overlap 0.153 -- inside
the pre-registered [0.05, 0.25] band and close to the hand-authored 26's
own mean (0.134); 0 of 92 exceeded the 0.5 high-overlap threshold (limit
14). The paraphrases are genuinely comparable in character to the
hand-made eval queries, confirmed, not assumed.

**Gate 2 (>=10 positive examples after relabeling) failed.** Relabeling
with the paraphrased text (same `hit@k` logic, same pinned retriever)
gave **6 positive / 86 negative** -- a real improvement over #5a's 0/92,
but under the precommitted floor: a stratified split would leave ~1
positive example in validation, the exact meaninglessness problem the
floor exists to prevent. Per #5a's own text, the script halted on its
own recognition of the failed gate and did not write `faq_expansion.jsonl`
or proceed further.

**Decision, checked against #5a's stated anti-pattern before being made:**
do not lower `MIN_POSITIVE_EXAMPLES` from 10 to 6 to accept this pool --
revising a precommitted gate after seeing it fail is the exact pattern
#5a's text warns against ("do not re-run... tuning generation parameters
until a precommitted check passes"), and lowering the acceptance bar
after the fact is the same move by another name. Do not attempt a third
automated pass (e.g. a stronger paraphrase prompt) either -- two
independent attempts (verbatim text, LLM-paraphrased text) both landing
below the floor is itself the finding, not a reason to keep searching for
a prompt that clears it.

**What this finding is, and, precisely, what it is not.** Two
consecutive labeling attempts -- verbatim FAQ text (0/92) and
lexically-dissimilar LLM-paraphrased text validated against the
hand-authored eval queries' own overlap regime (6/92) -- both fall below
a workable training-pool size on **this specific 126-document corpus**.
That is a finding about retrieval saturation on a small, dense FAQ index,
not a finding about whether a trained critic (cross-encoder or LoRA)
can or cannot beat the phase-5 prompted grader: **no critic was ever
trained**, so that question is untested here, not falsified. Any later
write-up must keep this distinction explicit -- "the corpus didn't yield
enough training signal to attempt the comparison" is a materially
different, narrower claim than "trained critics don't help," and
conflating the two would overclaim beyond what this data supports.

**Consequence for Stage 4 (scope change, pre-registered here before Stage
4 runs, per this document's own discipline):** the CE and LoRA critic
arms are **deferred, not cancelled** -- both need a corpus where
independently-phrased queries genuinely miss at a rate a training split
can use, which this repo's current FAQ corpus does not provide at k=5 on
a 126-document index. Stage 4 proceeds **Baseline-critic-only**: the
four-node multi-agent decomposition (retrieval -> critic -> generator ->
thin router, #6) is tested against phase 5's monolithic loop using only
the frozen, prompted `grade_documents` critic, over the same 51-query
set, reporting call-count and latency deltas either way (#3's second
null prediction, unaffected by this amendment). This answers the
phase's *decomposition* question on schedule; it does not answer the
phase's *trained-critic* question, which moves to a **future,
separately-scoped corpus** with a non-saturated retrieval task (a larger
or more open-ended document set than this repo's 126-row FAQ index) --
named as follow-on work in the eventual results doc, not silently
dropped.

**Kept, not discarded, as evidence:** `data/critic_training/faq_expansion.jsonl`
(#5a's 92-row verbatim-text attempt, 0 positive),
`data/critic_training/faq_paraphrases.jsonl` (this amendment's 92
paraphrases + per-row Jaccard overlap), and
`data/critic_training/faq_paraphrases_summary.json` (both gates'
measured numbers) are committed alongside this section -- the 6 genuine
LLM-paraphrase-induced misses these files contain are real signal a
future corpus's training pool could be seeded with, not wasted work.

## 6. Architecture

```
retrieval agent -> critic agent -> generator agent -> (optional thin router)
```

LangGraph, in a **new** module (not a modification of
`src/adapters/corrective_rag.py`, which stays frozen -- verified by `git
diff` showing no changes to that file, same discipline phase 5 itself
used against phase 4's `adapters/agentic.py`). Node-by-node reuse, so no
retrieval or prompt logic is duplicated:

- **Retrieval agent**: `route_node` + `retrieve_node`, same shape as
  `corrective_rag.py`'s (`ROUTE_PROMPT`/`RouteDecision` from
  `adapters.agentic`, `eval.retrievers.query_routed`).
- **Critic agent**: the swappable component. `critic: "baseline" |
  "ce" | "lora"`, config-selected (`src/config.py`, a new
  `MultiAgentCriticConfig` dataclass following the existing
  `CorrectiveAgenticConfig` pattern). `baseline` invokes the identical
  `JUDGE_PROMPT`/`JudgeDecision` call `corrective_rag.py`'s
  `grade_documents_node` makes -- same prompt, same schema, same
  fail-open-to-`sufficient` behavior on a call failure, byte-for-byte,
  not "equivalent." `ce` scores `(query, concatenated excerpts)` with the
  trained cross-encoder against its chosen threshold, no LLM call, no
  API key. `lora` invokes the locally-served LoRA-adapted grader.
- **Generator agent**: `generate_node` + `grade_generation_node`, reusing
  `GENERATE_PROMPT`/`FAITHFULNESS_PROMPT`/`GeneratedAnswer`/
  `FaithfulnessGrade` unchanged from `corrective_rag.py`.
- **Thin router** (optional): the existing `decide_to_generate` /
  `decide_after_generation` conditional-edge shape, kept as conditional
  edges rather than promoted to a fifth named node unless Stage 4's
  implementation finds a concrete reason to split it out.

**Hard constraint, all three arms:** phase 5's **asymmetric fail-open
logic is unchanged in the Baseline arm** -- `grade_documents` fails open
to proceed (`sufficient`), `grade_generation` fails open to
`not_grounded`, for the reasons `corrective_rag.py`'s own inline comments
and `eval/METHODOLOGY.md` #20 already state (an infra failure is not a
content judgment; an unverifiable answer must never be presented as
grounded). CE and LoRA critic-call failures must fail open the same
direction the Baseline arm does (to "proceed", not to "insufficient") for
the comparison across arms to isolate critic *quality*, not differing
failure-handling policy.

**Dependency isolation.** CE and Baseline arms stay CPU-clean and keyless
-- `sentence-transformers` (CE's `CrossEncoder`) is already a base
dependency (`requirements.txt`), so the CE arm adds nothing to the base
install. `peft`/GPU-training dependencies (`accelerate`, `peft`) land in
a new `requirements-lora.txt`, imported only by the LoRA training script
and the LoRA critic-loading path -- never added to `requirements.txt` or
`requirements-dev.txt`, so the keyless arms' and the deployed Streamlit
app's dependency surface (`app/requirements.txt`) are unaffected.

## 6a. Stage 4 implementation note and pre-registered predictions -- dated 2026-08-09, before any decomposition code runs

**Implementation note, filling a gap #6 left open.** #6's diagram names
three agents plus a conditional-edge "thin router"; it does not say where
`rewrite_query` lives, since phase 5's own graph has no agent grouping to
place it in. Decision, made now, not discovered mid-implementation: the
**retrieval agent's node handles both entry modes** -- on first entry
(no `route` yet in state) it does `route` + `retrieve`, exactly as
`corrective_rag.py`'s `route_node`/`retrieve_node`; on a retry entry
(the thin router sent it back) it does `rewrite_query` + `retrieve`
instead, reusing `REWRITE_PROMPT`/`RewrittenQuery` unchanged, never
re-routing on a retry -- the same behavior phase 5's fixed edge shape
(`rewrite_query -> retrieve`, never `rewrite_query -> route`) already
has. This keeps the graph at exactly three named agent nodes plus the
existing `respond`/`abstain` terminals (not counted as agents, same as
phase 5), consistent with #6's diagram, while reproducing phase 5's loop
shape exactly rather than inventing a new one.

**Predictions, pre-registered before any run, per this document's own
standing discipline (#10) -- ship these even if they hold:**

1. **Quality: per-query decisions should match phase 5's, up to LLM
   API-level nondeterminism.** The Baseline critic agent, retrieval
   agent, and generator agent all invoke the identical prompts/schemas/
   fail-open logic `corrective_rag.py` already uses (reused, not
   retyped -- #6's hard constraint). There is no architectural reason a
   pure reorganization into agent-grouped nodes should change *what* gets
   decided, only how the code that decides it is arranged. Predict the
   confusion matrix, rescue rate, and faithfulness rate land within the
   same noise band phase 5 itself already documented (`temperature=0`/
   `seed=42` is OpenAI's own best-effort, not guaranteed, determinism --
   `eval/METHODOLOGY.md` #15, `results/ANALYSIS.md`'s own "zero measured
   variance... on the aggregate" note) -- not necessarily byte-identical,
   but not a systematic quality shift either.
2. **Call count: near-identical.** Same five LLM call sites (route,
   critic/`grade_documents`-equivalent, `rewrite_query`, `generate`,
   `grade_generation`) -- the thin router adds no LLM calls of its own.
   Predict the new graph's real call count over one pass of the 51-query
   set lands within a few calls of phase 5's own per-run figure (measured
   545 total / 3 runs, `results/ANALYSIS.md`).
3. **Latency: equal-or-worse for the decomposed graph, never meaningfully
   better.** LLM round-trip time dominates wall-clock cost; both graphs
   are already LangGraph state machines (phase 5 is not "no graph," it is
   an 8-node graph), so decomposition changes node *count* (8 down to 5:
   3 agents + respond + abstain) and *grouping*, not whether a graph
   framework is involved at all. Fewer, coarser node-transitions could
   shave negligible in-process overhead at best; there is no mechanism by
   which grouping existing LLM calls into fewer named nodes makes those
   calls faster. Predict the measured delta lands within timing noise, or
   the decomposed graph is measurably (if narrowly) slower.
4. **Explicit null framing:** if 1-3 hold as predicted -- equivalent
   quality, near-identical call count, no latency win -- that is the
   expected, reported result, not a failed search for something more
   interesting. This phase's multi-agent split's actual value, if any, is
   the swappable-critic seam and code organization (#6), not a runtime
   performance or quality improvement; the results doc must say this
   plainly rather than mining the numbers for a difference to report.

**Cost estimate, stated before any real call is made (mirrors
`eval/METHODOLOGY.md` #21's discipline).** Fair latency comparison
requires **fresh, uncached** calls for both graphs, run back-to-back in
the same session -- a cached call returns near-instantly and would not
reflect real latency, so this comparison cannot reuse phase 5's
populated `eval/.llm_cache.json` the way Stage 2's extraction did. Old
graph: `corrective_rag.build_corrective_graph`, imported unmodified, run
fresh over the 51-query set (one pass, not phase 5's original 3 -- this
is a structural/timing comparison, not a re-litigation of the
already-established API-variance question). New graph: this phase's
multi-agent graph, Baseline critic, same 51 queries, one fresh pass.
Same per-query worst-case-12-calls structure as phase 5 (`#19`'s
estimate: ~6 calls/query average). **Approved estimate: 51 queries x 2
graphs x ~6 calls/query = 612 calls, hard stop 918 (1.5x)**, same
hard-stop-not-just-warning discipline as every other real-call harness in
this repo.

## 7. Deliverables, mapped to stages

**Updated per #5b (2026-08-09):** items 2 and 4 change scope from what
was originally listed here. Original text kept below, struck through in
spirit (not literally deleted -- see the #5b status note under each
changed item), so the record shows the original commitment alongside
what the training-data checkpoint actually supported.

1. `docs/phase6_preregistration.md` -- **this document. Stage 1.**
2. ~~Critic training scripts (CE and LoRA), reproducible, seeded~~ --
   `training/extract_critic_data.py` + `training/generate_faq_paraphrases.py`
   were built and run for real (Stage 2); both attempts (#5a, #5b) fell
   below the training-pool floor. **`train_ce_critic.py` and
   `train_lora_critic.py` are deferred to the future non-saturated-corpus
   follow-on named in #5b, not built this phase** -- writing a training
   script against a known-unusable 6-example pool would produce a script
   that runs but whose output means nothing, which is not a real
   deliverable.
3. LangGraph multi-agent graph with config-selectable critic -- Stage 4,
   **built with the `critic` seam present (#6) but exercised
   Baseline-only** (#5b) -- the config field and node shape support
   `ce`/`lora` for when a critic exists to plug into it, they are just
   unused this phase.
4. ~~Eval harness extension: three arms over the existing 51-query set,
   tradeoff-curve output~~ -- **Stage 4 runs Baseline-only**: the
   four-node decomposition vs. phase 5's monolith, call-count and latency
   reported either way (#5b). No tradeoff curve this phase (a curve needs
   >=2 critics with a tunable threshold; Baseline is the single reference
   point #2 already described).
5. Results doc: honest negatives, **decomposition call-count/latency
   table** (not a firing-rate/rescue-rate curve, per #5b) -- Stage 4.
6. Tests green before any merge request; advertised count matches CI
   reality -- Stage 4, checked before any merge is proposed.

## 8. LoRA execution model

The development sandbox this phase is built in is Windows, CPU-only
torch (verified: `torch.cuda.is_available()` is `False`), no `peft`
installed, and has no access to the GCP account or Ollama/local-host
instance the originating brief names as the LoRA arm's intended
GPU-training and serving environment. Given that:

- `training/train_lora_critic.py` (Stage 3) is written to be **correct
  and reproducible** (seeded, documented inputs/outputs, run instructions
  in `training/README.md`) but is **not executed in this session**.
- Both this document and Stage 4's results doc must mark the LoRA arm
  explicitly as **"script complete, not run in session -- pending
  GCP/Ollama run."** Reported LoRA numbers, if any ever appear in this
  repo, come only from a real run on the user's own GCP/Ollama
  environment -- never estimated, interpolated, or simulated here.
- If a small CPU smoke-test of the LoRA script is run at any later stage
  purely to check the script doesn't crash, it is for script-correctness
  verification only: not committed as data, not reported as a result, and
  explicitly labeled a smoke test if it is mentioned at all.

## 9. Deployment invariant (do not violate)

The public Streamlit demo (`app/`) stays **phase-1/2 live-retrieval
only**. Phase 6 work lives in the README + repo, not the deployed app --
replaying LLM traces on a public no-key surface buys nothing over prose,
same reasoning already documented for why the app never grew phase 4/5's
LLM-judge features. The LoRA arm's GPU training (GCP Innovators credits)
and local serving (Ollama or a keyed local host) stay off any public
surface entirely. The LoRA critic is never wired into anything
deployable.

## 10. Standing scope guardrails

- Work happens on `phase6-multiagent-critic`. **No push to `origin`/master
  without explicit approval.** Diffs and evidence (test output, run
  output) are reported before any merge is proposed.
- No secret values touched or printed.
- No deletions, ever.
- `gh` CLI over raw REST API, wherever GitHub interaction is needed.
- No data committed without clearance (`data/critic_training/`: cleared,
  #5).
- Pre-register before results -- this document exists before any Phase 6
  code that produces a number. Failures are reported openly: a null
  result on the win condition, or on either trained critic beating the
  prompt, is the differentiator this phase is for, not something to
  smooth over after the fact.
