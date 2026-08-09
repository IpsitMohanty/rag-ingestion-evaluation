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

## 7. Deliverables, mapped to stages

1. `docs/phase6_preregistration.md` -- **this document. Stage 1.**
2. Critic training scripts (CE and LoRA), reproducible, seeded --
   `training/extract_critic_data.py` + `training/train_ce_critic.py`
   (Stage 2, executed for real); `training/train_lora_critic.py` (Stage
   3, script-only -- see #8).
3. LangGraph multi-agent graph with config-selectable critic -- Stage 4.
4. Eval harness extension: three arms over the existing 51-query set,
   tradeoff-curve output (Baseline + CE run for real; LoRA reported
   not-run per #8) -- Stage 4.
5. Results doc: honest negatives, firing-rate/rescue-rate curve, latency
   + call-count table -- Stage 4.
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
