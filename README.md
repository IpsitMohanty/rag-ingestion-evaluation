# Poshan RAG Evaluation

A RAG pipeline (ingest -> split -> embed -> index -> retrieve) built around
two structurally opposite public corpora, plus a retrieval evaluation
testing the pipeline's central design bet: format-aware, per-source-type
ingestion beats forcing everything through uniform chunking.

**Phase 1** built the pipeline (ingest through retrieve, end to end,
runnable via CLI). **Phase 2** built and ran the evaluation --
`eval/METHODOLOGY.md` settles the scoring rules in writing, and
`results/ANALYSIS.md` has the findings, including where the hypothesis
held and where the evidence doesn't support a strong claim. Still not
built: an LLM generation step, a UI, and the whole-document-into-prompt
baseline arm (see [Baseline arm: cost-and-recall, not
LLM-judged](#baseline-arm-cost-and-recall-not-llm-judged) below for why
that one's a documented seam, not an oversight).

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
  (`page` number kept in metadata), then split.

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
`results/sweep_results.json` and `results/ANALYSIS.md`. Headline, with
the caveat that headlines flatten nuance the linked analysis doesn't:

- **Format-aware ingestion beats uniform chunking on the FAQ bucket**,
  and the win's size tracks a directly-measured mechanism (average
  number of distinct FAQ rows merged into one chunk under uniform
  chunking -- 1.00 at chunk_size 200, rising to 2.51 at chunk_size 800),
  not just an unexplained hit-rate gap.
- **Similarity vs `ParentDocumentRetriever` cannot be distinguished** on
  this policy corpus at this scale (11 labeled policy queries) -- hit@1
  is identical in every comparison.
- **The system cannot reliably tell "found the answer" from "didn't"
  from its similarity score alone**, in any of the 12 configurations
  tested -- confirmed by comparing top-1 score distributions on
  unanswerable ("neither") queries against answerable ones. Means point
  the right way; ranges overlap every time.
- **A real share of FAQ retrieval's apparent quality is lexical, not
  semantic**: 3 named-UI-feature queries score a perfect hit@1;
  same-cell hit@1 on the other 23 is 0.435, not the ~0.46-0.50 the
  whole-bucket number suggests.

### Baseline arm: cost-and-recall, not LLM-judged

The reference course notebook that was expected to justify chunking
(`reference/course-notebooks/Full document retrieve limitation-v1.ipynb`)
doesn't actually measure a failure: it stuffs an ~8,235-token document into
a large-context-window model and the model answers correctly, because the
document fits. It never demonstrates truncation or a wrong answer -- the
"limitation" is asserted narratively, not measured.

Given that, and given this corpus is genuinely small enough that
whole-document context stuffing is a real competitor rather than a straw
man, phase 2's baseline arm is designed as **deterministic, no LLM**:

- Context stuffing has **recall = 1.0 by construction** -- the whole
  corpus is in the prompt, so nothing relevant can be missed. State this
  plainly rather than manufacturing a retrieval "win" on recall.
- The actual measured comparison is **tokens per query**: whole-corpus
  token count vs. `k * chunk_size` (plus one-time index build cost for
  the retrieval arm).
- **Lost-in-the-middle degradation** (LLMs attending less reliably to
  content buried in the middle of a long context) is cited as a known,
  literature-supported limitation of context stuffing -- **not** asserted
  as something this repo measured, since nothing here does.

An optional LLM-answer-quality arm (actually generating answers from both
arms and judging them) is documented as a future extension, gated behind
an API key via the `llm` adapter, and explicitly excluded from CI.

### Whole-document-into-prompt: a baseline, not a feature

`pipeline/` has no context-stuffing code path yet -- this is the seam
phase 2 fills in, not a phase 1 deliverable. It's a comparison baseline
against retrieval, not an alternative retrieval strategy.

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
