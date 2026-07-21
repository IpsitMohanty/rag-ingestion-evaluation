# RAG Ingestion Evaluation

A LangChain RAG pipeline (`langchain-core`, `langchain-text-splitters`,
`langchain-huggingface`, `langchain-chroma` -- ingest -> split -> embed ->
index -> retrieve) built around two structurally opposite public corpora,
plus a retrieval evaluation testing the pipeline's central design bet:
format-aware, per-source-type ingestion beats forcing everything through
uniform chunking. See [Stack](#stack) for exact packages/versions, and
[Ingestion](#ingestion-per-source-type-not-global) for the one deliberate
place this repo does *not* use LangChain (document loading -- by design,
not by omission).

**Phase 1** built the pipeline (ingest through retrieve, end to end,
runnable via CLI). **Phase 2** built and ran the evaluation, including a
deterministic cost-and-recall baseline arm against naive whole-document
context stuffing. Two things worth knowing before anything else here,
because they qualify every other number in this README:

- **The system cannot reliably tell "found the answer" from "found
  nothing" using its similarity score alone**, in any of the 12
  configurations tested -- see [Phase 2](#phase-2-retrieval-evaluation).
- **The honest, semantic-only FAQ hit-rate curve is 0.435 (k=1) / 0.783
  (k=3) / 0.826 (k=5) / 0.826 (k=10)** -- not the 0.46-0.50 a whole-bucket
  number would suggest. hit@5=0.826 is the figure to use in practice
  (real systems retrieve k=5, not k=1); it's also where the curve stops
  improving -- see the [k=5-to-k=10 plateau](results/ANALYSIS.md#the-k5-to-k10-plateau-a-representation-ceiling-not-a-retrieval-depth-problem).

Full methodology (settled in writing *before* any run) is in
`eval/METHODOLOGY.md`; full findings are in `results/ANALYSIS.md`. Still
not built: an LLM generation step and a UI.

## Corpus

Two public sources, structurally opposite:

1. **Poshan Tracker app FAQs** -- 126 scraped Q&A pairs
   (`data/faq/all_faq.json`). Fields: `tab` (5 values), `subcategory`
   (13 named values + null for 31 rows), `question`, `answer`. Answers
   average ~178 characters -- already atomic retrieval units, so they are
   never split (see [Ingestion design](#ingestion-per-source-type-not-global)).

2. **Mission Saksham Anganwadi & Poshan 2.0 scheme guidelines** (PDF, ~1.5MB,
   `data/policy/`) -- the broad policy document covering ICDS and Poshan
   Abhiyaan. Deliberately excludes the ministry's circulars, device
   specifications, and inter-government communications: those are
   administrative correspondence, not policy content.

**Constraint: public documents only, no internal material, no
individual-level data.** Do not add other document types to this corpus
without explicit sign-off -- the per-source-type design makes it easy to
add a new source, which is exactly why this needs to stay a deliberate
decision rather than something that happens by accident.

### FAQ format: JSON, not CSV

Both formats are supplied (`data/faq/all_faq.json` and `all_faq.csv`); the
pipeline defaults to JSON. Several answers contain embedded newlines and
markdown-style bullet lists (e.g. the "kinds of beneficiaries" answer);
JSON represents these natively with zero ambiguity, while CSV requires
trusting that multi-line fields were quoted correctly by whatever scraped
the data. Both parse to the same 126 rows (see `tests/test_ingestion_faq.py`),
but JSON is the loader's default source; the CSV is kept for provenance,
not actively loaded by `pipeline/ingest.py`'s extension map.

## Architecture

```
src/
  config.py       chunk size/overlap, model names, k, per-source-type
                   settings -- all in one place
  adapters/        the ONLY layer that imports a LangChain integration
    ingestion.py     FAQ loader (no LangChain loader at all -- stdlib
                      csv/json) + PDF loader (pypdf directly, not a
                      LangChain PDF loader wrapper) + splitting
    embeddings.py    swappable backend: huggingface (default, local) /
                      watsonx / openai, same interface either way
    vectorstore.py   Chroma, persisted to disk
    llm.py           swappable backend seam -- not called by any phase 1
                      pipeline code
    retriever.py     strategy is a config value; only "similarity" is
                      implemented (see below)
  pipeline/        application code -- depends on adapters/, never
                   imports a LangChain integration directly
    ingest.py        discover corpus files, load into raw Documents
    split.py         apply per-source-type splitting
    index.py         embed + store
    retrieve.py      build a retriever, run a query
  cli.py           `python src/cli.py ingest <folder>...` /
                   `python src/cli.py query "<text>"`
```

The adapter boundary is the point: `langchain-community` was archived by
its maintainers on 19 June 2026 and receives no further releases (see
[Stack](#stack) below). LangChain's own post-sunset migration guidance is
to keep integrations behind your own adapters so a deprecated package
means swapping one adapter, not rewriting the application. `pipeline/` and
`cli.py` never import `langchain_huggingface`, `langchain_chroma`, etc.
directly -- only `adapters/` does.

### Ingestion: per-source-type, not global

`config.IngestionConfig.source_types` is a dict keyed by source type
(`"faq"`, `"policy_pdf"`), each with its own `split: bool` and, if
splitting, its own `chunk_size`/`chunk_overlap`:

- **FAQ**: `split=False`. Each Q&A pair loads as exactly one `Document`
  (`page_content` = "Question: ...\nAnswer: ..."), with `tab` and
  `subcategory` preserved as queryable metadata (subcategory is
  normalized from `null` to `""` for the 31 rows that have none -- Chroma's
  metadata store rejects `None`).
- **Policy PDF**: `split=True`, `RecursiveCharacterTextSplitter`
  (chunk_size=800, chunk_overlap=100). Loaded page-by-page first
  (`page` number kept in metadata), then split. `clean: bool = False` is
  a second, independent PDF-only option: strip running headers/footers
  and drop front matter (title page, table of contents) before
  splitting -- see [Phase 2](#phase-2-retrieval-evaluation) for whether
  the swept numbers show this actually improves retrieval, and
  `eval/METHODOLOGY.md` #8 for exactly what "cleaned" means and why.
  `False` (raw) stays the default.

Switching either source type's behavior -- turning splitting on for FAQs,
changing the PDF's chunk size, adding a third source type -- is a
`config.py` change, not a code change. This is deliberate: phase 2's
evaluation is exactly a comparison of this format-aware ingestion against
forcing both source types through identical, uniform chunking, and that
comparison only means something if neither path is hardcoded.

### Retriever strategies

`"similarity"` (a plain Chroma `as_retriever()`) and `"parent_document"`
are implemented. `config.RetrieverConfig.strategy` is real, selectable
config for the rest; each raises `NotImplementedError` naming the
specific reason it isn't built, instead of silently falling back to
similarity:

- **`parent_document`** -- needs no LLM at retrieval time (just a
  child/parent splitter pair and a docstore alongside the vectorstore),
  which is why it's the one other strategy phase 2 could actually
  evaluate in this repo's no-API-key path. Built on `langchain_classic`,
  not `langchain` -- see [Stack](#stack). Phase 2's finding: it can't be
  distinguished from plain similarity search at this corpus's scale (11
  policy queries) -- see `results/ANALYSIS.md`.
- **`multi_query`** -- needs an LLM at retrieval time to generate query
  variants. Seam only; out of scope while there's no LLM adapter wired in.
- **`self_query`** -- needs an LLM to build a structured metadata filter
  *and* the `lark` package. The reference lab (see below) flags it as
  flaky even with a working LLM ("you might encounter errors or blank
  content... re-run it several times"). Seam only.

## Stack

`langchain-community` is archived (19 June 2026, no further releases) --
not used anywhere in `src/`. Instead:

| Package | Pinned | Why |
|---|---|---|
| `langchain-core` | 1.4.9 | Document/Embeddings/VectorStore/Retriever base types |
| `langchain-text-splitters` | 1.1.2 | `RecursiveCharacterTextSplitter` |
| `langchain-huggingface` | 1.2.2 | Local embeddings backend |
| `langchain-chroma` | 1.1.0 | Chroma vector store integration |
| `langchain-classic` | 1.0.8 | `ParentDocumentRetriever` (see below) |
| `sentence-transformers` | 5.6.0 | Backend `HuggingFaceEmbeddings` needs |
| `pypdf` | 6.14.2 | Direct PDF text extraction (see below) |
| `pyyaml` | 6.0.3 | `eval/queries.yaml` |

All current-as-of-writing on PyPI; check before assuming they still are.

**Finding from building phase 2**: LangChain's 1.0 line slimmed the main
`langchain` package down to `agents`/`chat_models`/`embeddings`/`messages`/
`rate_limiters`/`tools` and moved legacy chain/retriever abstractions --
including `ParentDocumentRetriever`, which the phase 1 brief assumed would
just be `from langchain.retrievers import ParentDocumentRetriever` -- into
a separate, still-maintained `langchain-classic` compatibility package.
Confirmed by import error against `langchain==1.3.14`; not something
either brief anticipated.

PDF and FAQ loading do not use LangChain document loaders at all --
`adapters/ingestion.py` reads the PDF with `pypdf` directly (which is all
`langchain_community.PyPDFLoader` ever was, a thin wrapper around it) and
the FAQ files with the stdlib `csv`/`json` modules, building
`langchain_core.documents.Document` objects itself. This sidesteps the
community package for loading entirely, not just for the pieces this repo
happens to use.

**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (~90MB,
384-dim, CPU-friendly, no API key or account) is the default over the
course's `all-mpnet-base-v2` (~420MB, 768-dim) -- CPU/CI-friendliness
matters more here than the marginal quality difference. Swapping to
mpnet, or to watsonx/OpenAI embeddings, is a `config.EmbeddingConfig`
change (see `adapters/embeddings.py`); the watsonx/openai branches are
real code reading real environment variables, not stubs, though neither
package is installed by default so the repo runs with zero API keys out
of the box.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt

python src/cli.py ingest data/faq data/policy
python src/cli.py query "How many kinds of beneficiaries can be registered?"
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

85 tests, ~26s on this machine. All but one run fully offline: no network
call, no model download. The one exception
(`test_huggingface_backend_produces_expected_dimension`) actually
constructs the real `HuggingFaceEmbeddings` backend, which is *eager* --
it downloads/loads the sentence-transformers model at construction time,
not on first use -- so it's gated behind `RUN_NETWORK_TESTS=1` and not run
in CI. Everywhere else, adapter/pipeline contract tests -- including the
phase 2 harness tests (`test_eval_*.py`) -- use
`langchain_core.embeddings.DeterministicFakeEmbedding`, a hash-based
Embeddings implementation, to prove the vectorstore/retriever adapters
work with *any* Embeddings implementation -- which is the actual claim
"swappable backend" is making.

The phase 2 **sweep** (real embeddings, 12 config cells x 51 queries) is
not part of this suite and not run in CI -- it's a several-minute run, not
a test. Run it with:

```bash
python -m eval.run_sweep   # first run needs network to download+cache the
                            # embedding model; HF_HUB_OFFLINE=1 once cached
```

## Phase 2: retrieval evaluation

Built and run. Full methodology (settled in writing before any run --
scoring rules for "either"/"neither", how retrievable-but-incomplete and
lexical-overlap are handled, what "uniform chunking" means for a corpus
of already-atomic FAQ rows) is in `eval/METHODOLOGY.md`; the 51-query
labeled set is `eval/queries.yaml`; full results are
`results/sweep_results.json` and `results/ANALYSIS.md`. Findings, in the
order `results/ANALYSIS.md` presents them -- strongest/most surprising
first, with the caveat that headlines flatten nuance the linked analysis
doesn't:

1. **The system cannot reliably tell "found the answer" from "didn't"
   from its similarity score alone**, in any of the 24 configurations
   tested. Top-1 similarity-score distributions for unanswerable
   ("neither") queries and answerable queries overlap in every single
   cell of the sweep -- some answerable queries score worse than every
   unanswerable one, and vice versa. **Consequence: you cannot build a
   reliable "I don't know" gate on a similarity threshold with this
   embedding model on this corpus** -- and that's a common assumption in
   shipped RAG systems (retrieve, threshold, abstain below it). This is
   the strongest finding in the evaluation, not a footnote to the
   chunking result below.

2. **Format-aware ingestion beats uniform chunking on the FAQ bucket --
   conditionally.** The rule of thumb the data supports: format-aware
   ingestion matters once chunk_size exceeds the corpus's smallest atomic
   unit (~178 chars, the FAQ's average answer length here); below that,
   the two are statistically indistinguishable (identical hit@1 at
   chunk_size 200). Above it, the win's size tracks a directly-measured
   mechanism (average number of distinct FAQ rows merged into one chunk
   under uniform chunking -- 1.00 at chunk_size 200, rising to 2.51 at
   chunk_size 800), not just an unexplained hit-rate gap. "Format-aware
   ingestion helps once your chunks would otherwise span multiple atomic
   records" is the transferable takeaway -- not "always split FAQ-shaped
   content separately regardless of size."

3. **A real share of FAQ retrieval's apparent quality is lexical, not
   semantic.** 3 named-UI-feature queries ("RCH Profile", "Home Visit",
   "Poshan Tracker Dashboard" -- unavoidable proper nouns, not sloppy
   rewording) score a perfect hit@1. The other 23, paraphrased to avoid
   lexical overlap with the stored question text, score **0.435 (k=1) /
   0.783 (k=3) / 0.826 (k=5) / 0.826 (k=10)** -- this full curve is the
   honest, semantic-only number, not the 0.46-0.50 the whole FAQ bucket
   suggests. hit@5=0.826 is the practical figure; see next finding for
   why k=10 doesn't improve on it.

4. **The k=5-to-k=10 plateau is a representation ceiling, not a
   retrieval-depth problem.** hit@10 equals hit@5 *exactly* -- 0.826 at
   both -- meaning 4 of the 23 low-overlap queries (17%) never surface
   their target chunk at *any* k tested. "Just retrieve more chunks" is
   the obvious fix for a hit-rate gap; this data shows it doesn't work
   here. If it were a depth problem, hit@10 would keep climbing past
   hit@5 as more candidates get a chance to include the right one -- it
   doesn't, in any cell of the sweep. That points to a limit in how
   `all-MiniLM-L6-v2` represents those specific paraphrase/answer pairs
   in vector space, not a limit in how many candidates the retriever
   considers. This **compounds finding #1**, it doesn't just sit next to
   it: the system can't flag when it's found nothing good, and raising k
   -- the standard mitigation for "maybe it's just outside the top few"
   -- doesn't rescue these particular misses either.

5. **Weighed against whole-corpus context stuffing** (see [baseline
   arm](#baseline-arm-cost-and-recall-built-and-run) below), retrieval at
   k=5/chunk_size=800 is ~40x cheaper per query but caps at the 0.826
   ceiling above on the hardest (paraphrased, semantic-only) queries and
   never reaches 1.0 even at k=10. That's a genuine trade-off at this
   corpus's scale, not a case either arm wins outright -- and finding #1
   means a retrieval-only system has no cheap way to detect when it
   should have stuffed the whole document instead.

6. **Similarity vs `ParentDocumentRetriever` cannot be distinguished** on
   this policy corpus at this scale (11 labeled policy queries) -- hit@1
   is identical in every comparison; hit@10 favors whichever strategy by
   a single flipped query depending on chunk_size, with no consistent
   winner.

7. **Cleaning the PDF (stripping running headers, dropping the title
   page and table of contents) makes no detectable difference to
   retrieval.** Predicted and a decision threshold set in writing before
   the numbers were run (`eval/METHODOLOGY.md` #8a: real effect = hit@5
   changing by more than 0.182, the largest swing already seen between
   configurations that aren't testing cleaning). Result: hit@5 at the
   representative cell is identical (0.727, both variants), and no
   configuration in a 6-way robustness grid clears the bar either. The
   front-matter/header artifacts visible in the demo's own "neither"-
   query output are cosmetic, not a retrieval-quality cost -- see
   [Cleaned vs raw PDF](results/ANALYSIS.md#cleaned-vs-raw-pdf-does-stripping-headers-and-front-matter-help-retrieval)
   for the full numbers, including one methodological finding (bucket
   metrics aren't perfectly isolated to their own side of the corpus)
   that surfaced along the way.

### Limitations

**Per-bucket metrics measure that bucket's queries within combined
retrieval over the whole corpus, not that source in isolation -- so no
metric change can be attributed cleanly to one source's ingestion
config.** `query_combined()` (`eval/retrievers.py`) merges FAQ and policy
hits into a single ranked top-k by score before any bucket metric is
computed. That's the realistic operational setting -- a real system
returns one blended top-k across both source types, not a separate list
per source -- but it means a change on one side of the corpus can
displace which hits from the *other* side survive into that shared
window. This was caught directly: PDF cleaning (a policy-only change)
measurably shifted the *reported FAQ-bucket* hit-rate at the same
configuration, even though the FAQ documents and embeddings never
changed; the same displacement runs the other way too (FAQ ingestion
mode shifting the reported policy-bucket numbers). **This qualifies
every headline per-bucket number in this document, not only the
raw-vs-cleaned comparison** -- a cross-bucket delta under about 0.05
anywhere here may reflect this shared-ranking effect rather than a real
change in that bucket's own retrieval quality. Full detail:
[Cleaned vs raw PDF](results/ANALYSIS.md#cleaned-vs-raw-pdf-does-stripping-headers-and-front-matter-help-retrieval)
and `results/ANALYSIS.md`'s own Limitations section.

### Baseline arm: cost-and-recall, built and run

The reference course notebook that was expected to justify chunking
(`reference/course-notebooks/Full document retrieve limitation-v1.ipynb`)
doesn't actually measure a failure: it stuffs an ~8,235-token document into
a large-context-window model and the model answers correctly, because the
document fits. It never demonstrates truncation or a wrong answer -- the
"limitation" is asserted narratively, not measured. That's why this
corpus was designed to make whole-document stuffing a real competitor
(small enough to fit a modern context window) rather than a straw man --
and with the semantic-only FAQ curve topping out at hit@5=0.826 (finding
#3 above, plateauing per finding #4), it earns that status: **this
baseline arm was built and run, deterministically, no LLM**, not just
designed and left as a seam.

Whole corpus: 31,778 chars (FAQ) + 128,803 chars (policy PDF, 77 pages) =
160,581 chars, **~40,145 tokens** (approximate, ~4 chars/token -- not a
precise tokenizer count, but the right order of magnitude). That's
comfortably inside a modern 128K+-token context window.

| | tokens/query (approx) | vs. whole-corpus stuffing | recall (low-overlap FAQ, the honest curve) |
|---|---|---|---|
| whole-corpus stuffing | ~40,145 | 1x (baseline) | **1.0, by construction** -- everything's in the prompt |
| retrieval, k=1 | ~200 | 200x cheaper | 0.435 |
| retrieval, k=3 | ~600 | 67x cheaper | 0.783 |
| retrieval, k=5 | ~1,000 | 40x cheaper | 0.826 -- the practical figure |
| retrieval, k=10 | ~2,000 | 20x cheaper | 0.826 -- identical to k=5, see finding #4 |

Index build cost (retrieval only -- stuffing needs none): ~17.2s measured
on this machine to embed and index the representative cell's 339
documents/chunks, plus a ~1.1s one-time model load. Illustrative of the
qualitative asymmetry, not a rigorous benchmark.

**Recall = 1.0 by construction for stuffing** -- state that plainly
rather than letting retrieval look like it's competing on recall; it
isn't, by definition. **Lost-in-the-middle degradation** (LLMs attending
less reliably to content buried in the middle of a long context) is a
known, literature-supported limitation of context stuffing -- cited here,
**not** measured, since generating and grading answers is outside this
arm's scope (see below).

Whether the cost/recall trade-off favors retrieval or stuffing depends on
what a wrong or missing answer costs downstream -- outside this repo's
scope to decide. What this repo can say: at ~40K tokens, this corpus
isn't large enough to force the choice by scale alone; it would stop
being a genuine trade-off well before a much larger corpus, but that
boundary isn't measured here.

An optional LLM-answer-quality arm (actually generating answers from both
arms and judging them) remains a documented, **not built**, future
extension, gated behind an API key via the `llm` adapter and explicitly
excluded from CI -- don't confuse this with the cost-and-recall arm
above, which was built.

## Phase 3: Streamlit UI

A thin viewer over the phase 1 pipeline (`app/`), not a second
implementation of it. The *index* was built by the same
`adapters`/`pipeline` code `src/cli.py` uses (see
[ONNX: dropping torch from the deployed app](#onnx-dropping-torch-from-the-deployed-app)
below for why the running app loads that index rather than rebuilding
it). Retrieval-only by default, no API key or LLM call required.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

- **k is adjustable, 1-10, default 5** -- deliberately, not just for
  configurability. Raising k on the built-in "neither" preset (Poshan ke
  Paanch Sutra) is a two-click way to personally reproduce phase 2's
  k=5-to-k=10 plateau finding: results stop improving well before k=10,
  because the miss is a limit of the embedding's representation, not of
  how many chunks got considered.
- **Every result shows its source** (FAQ vs policy PDF), the relevant
  metadata (`tab`/`subcategory` for FAQ, `page` for the PDF), and the raw
  Chroma distance -- plus an in-UI note explaining why that score can't
  be read as a confidence gate (Finding #1), linking to
  `results/ANALYSIS.md`. Displaying the raw numbers is deliberate: the
  finding is that they can't separate answerable from unanswerable
  queries, and a visitor should be able to see that for themselves rather
  than take the README's word for it.
- **Optional LLM generation** is off by default; a sidebar field accepts
  a user-supplied OpenAI API key for one session's requests only. The key
  is never stored, logged, or written to disk anywhere in this app --
  passed directly to the client per call and out of scope the moment the
  function returns (`app/app_logic.py::generate_answer`). An absent or
  invalid key degrades cleanly to retrieval-only; it never crashes the app
  (verified in `tests/test_app_logic.py` with a monkeypatched failure,
  not a real API round-trip, since this suite stays network-free).
- **The policy PDF is indexed after cleaning** (running headers/title
  page/table of contents stripped) rather than raw -- a legibility
  choice for this demo, not a retrieval-quality one: cleaning scored
  0.182 lower on policy hit@1 at the representative sweep cell (a rank
  reshuffle, recall unchanged) with no consistent direction across the
  robustness grid, and didn't clear the pre-committed bar for a real
  effect either way. Full numbers and the tradeoff:
  [ONNX: dropping torch from the deployed app](#onnx-dropping-torch-from-the-deployed-app)
  below.

**Requirements files, deliberately separate**: `app/requirements.txt` is
the lean set Streamlit Community Cloud actually installs -- it does not
pull `pytest`, `nbformat`, `pyyaml`, or `langchain-classic` (none of which
the deployed app needs). `requirements.txt`/`requirements-dev.txt` cover
phases 1/2 and the full test suite; `langchain-openai` (the optional
generation arm) is intentionally in neither of those, only in
`app/requirements.txt`, since it's a feature of the *app* specifically.

### ONNX: dropping torch from the deployed app

**A first version of this app rebuilt the index from the raw corpus at
startup using the real HuggingFace/torch embeddings backend (phase 1's
default) -- measured at ~803MB RSS, only ~220MB under Streamlit Community
Cloud's 1GB ceiling with the platform's own overhead unmeasured. Too
tight to deploy.** Fixed the same way `ibm-ai-eng/cnn-vit-land-classification`
drops torch from its deployed demo: export the model to ONNX and run
inference with `onnxruntime` instead of torch/sentence-transformers, in
the app path only.

The architecture this enabled, not just a backend swap:

- **`app/build_index.py`** (dev-only, needs `requirements.txt`'s
  torch/sentence-transformers) builds the full Chroma index once,
  offline, and the result -- `app/prebuilt_index/`, ~3MB -- is committed
  to the repo. The deployed app **loads** this index; it never re-embeds
  the corpus at startup. Uses the **cleaned** PDF variant (stripped
  running headers, dropped title page/table of contents) -- a
  **presentation choice, not a retrieval-quality claim**: [Cleaned vs raw
  PDF](results/ANALYSIS.md#cleaned-vs-raw-pdf-does-stripping-headers-and-front-matter-help-retrieval)
  found no detectable retrieval difference between the two at the swept
  scale, so the deployed app uses whichever one demos more legibly (no
  table-of-contents dot-leaders or title-page fragments in view when a
  visitor inspects retrieved chunks). **This isn't free of tradeoffs even
  though it didn't clear the "real effect" bar**: at the representative
  cell, cleaned scored 0.182 lower than raw on policy hit@1 (0.455 vs
  0.636) -- a rank-1-vs-rank-2/3 reshuffle, not a recall loss (hit@5 and
  hit@10 both identical between variants), and with no consistent
  direction across the 6-configuration robustness grid. Reported here
  rather than left to sit only in `app/build_index.py`'s docstring.
  Phase 1's own default (`config.DEFAULT_CONFIG`, used by `python
  src/cli.py ingest`) stays raw, unchanged.
- **`app/export_onnx_model.py`** (also dev-only) exports
  `all-MiniLM-L6-v2` to ONNX (`app/onnx_model/`, ~87MB, also committed --
  same pattern as cnn-vit-land-classification's committed
  `cnn_model.onnx`).
- **`app/onnx_embeddings.py`** is the only thing the running app uses to
  embed anything: it embeds the user's *query* at request time via
  `onnxruntime` + the raw `tokenizers` library (not
  `transformers.AutoTokenizer` -- see below). Chroma only calls the
  embedding function to embed the query being searched for; it never
  re-embeds documents already stored in a loaded collection. That
  asymmetry is what makes dropping torch from the app safe: as long as
  ONNX query embeddings land in the same vector space as the
  torch-computed document embeddings already in `app/prebuilt_index/`,
  retrieval is unaffected.

**Parity was verified, not assumed** (`tests/test_onnx_parity.py`,
network-gated like the repo's other real-model tests since it needs the
torch model available/cached): cosine similarity between ONNX and torch
query embeddings on a stride-5 sample across all four query-set buckets
(11 of the 51 labeled queries) came back **≥0.999 for every query
tested** (max deviation ~1e-7, floating-point-level, not a real
divergence) -- confirmed before this path shipped, not after. If a
future model swap ever breaks that parity, this test fails loudly rather
than silently shipping a worse retriever.

**This parity test is skipped in CI -- a green CI run does not confirm
it.** `tests/test_onnx_parity.py` needs the torch model available (it's
gated behind `RUN_NETWORK_TESTS=1`, same as the repo's other real-model
tests, since a fresh CI checkout has no model cache). That means: after
*any* rebuild of `app/prebuilt_index/` (`app/build_index.py`) or
re-export of `app/onnx_model/` (`app/export_onnx_model.py`), you must run

```bash
RUN_NETWORK_TESTS=1 pytest tests/test_onnx_parity.py -v
```

**manually** before committing the result. CI staying green after either
of those regenerated artifacts change proves nothing about parity --
it only means the rest of the suite still passes. Both scripts print this
same instruction after they run, so it isn't only documented here.

### The `transformers`-import trap

**Non-obvious, and actionable for anyone shrinking a deployed embedding
model's footprint**: most ONNX-conversion writeups stop at "swap torch
for onnxruntime" and assume the job is done. It isn't, if tokenization
still goes through `transformers.AutoTokenizer`.

The first ONNX attempt here did exactly that -- reasonable, since
`AutoTokenizer` is the standard way to load a HuggingFace tokenizer --
and still measured ~803MB, barely better than the original torch path.
The cause: importing the `transformers` *package* costs ~340MB RSS on
its own, before any model loads and independent of torch entirely --
it's the cost of the package's model/config class registry, paid at
import time regardless of whether you ever touch a model class.

The fix: the raw `tokenizers` library (Hugging Face's Rust tokenizer
bindings -- what `transformers`' own "fast" tokenizer wraps internally).
Both read the exact same `tokenizer.json` and produce identical token
ids, so switching cost nothing in correctness (re-verified with the
parity test above after the switch) and recovered essentially all of
the ~340MB. Measured before/after:

| tokenizer choice | RSS after import | vs. torch/sentence-transformers baseline |
|---|---|---|
| `transformers.AutoTokenizer` | ~803MB total | barely better |
| `tokenizers.Tokenizer` (raw) | ~244MB total | ~560MB saved |

If you're exporting a small embedding model to ONNX for a memory-
constrained deployment, the tokenizer library you pick can matter as
much as dropping torch did -- check what your tokenizer import alone
costs before assuming the ONNX swap is the whole win.

**Requirements files, deliberately separate**: `app/requirements.txt` is
the lean set Streamlit Community Cloud actually installs -- no `torch`,
`sentence-transformers`, `pypdf`, `langchain-text-splitters`,
`transformers`, `pytest`, `nbformat`, `pyyaml`, or `langchain-classic`.
`requirements.txt`/`requirements-dev.txt` cover phases 1/2, the full test
suite, and the two dev-only scripts above (which need torch);
`langchain-openai` (the optional generation arm) is intentionally in
neither of those, only in `app/requirements.txt`, since it's a feature of
the *app* specifically.

**Measured memory footprint, ONNX path** (this machine, real corpus,
real prebuilt index):

| after | RSS |
|---|---|
| Python + Streamlit import | ~47MB |
| + `onnxruntime`/`tokenizers`/`chromadb` import (no torch, no transformers) | ~77MB |
| + loading the prebuilt index + ONNX model | ~228MB |
| + running a query | **~244MB** |

**~244MB against the 1GB ceiling -- roughly 780MB of headroom**, well
past the ~450-470MB estimate that motivated the switch (the
`transformers`-import finding above accounts for the difference: the
original estimate assumed `AutoTokenizer` was the lean option). Streamlit
Cloud's own overhead still isn't reflected in this local measurement, but
the margin is no longer the tight, single-point-of-failure number ~803MB
was.

## Reference material

`reference/course-notebooks/` -- the 6 (of 8) IBM "Generative AI
Applications with RAG and LangChain" course notebooks on hand, read for
what the labs covered but never executed and never ported: zero cells
have outputs or an execution count. Lab scaffolding and starter code are
IBM's; this repo's design was informed by them but doesn't reuse their
code. Licensed Apache 2.0 by IBM/Skills Network.

`reference/document-loaders-lab/` -- an earlier, unrelated, already
executed exploration of LangChain's document loaders from a prior
session, kept for reference and not part of this pipeline.
