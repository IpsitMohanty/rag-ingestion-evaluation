"""The sweep driver: builds each grid cell (ingestion_mode x chunk config
x retriever strategy), runs all 51 queries against it, and returns a
JSON-serializable results dict. Grid and rationale: eval/METHODOLOGY.md #6.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import RetrieverConfig

from . import corpus, metrics, retrievers

SWEEP_CHUNK_CONFIGS = [(200, 20), (500, 50), (800, 100)]
SWEEP_INGESTION_MODES = ["format_aware", "uniform"]
SWEEP_RETRIEVER_STRATEGIES = ["similarity", "parent_document"]
SWEEP_PDF_VARIANTS = ["raw", "cleaned"]
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


def _simple_hit_mrr(pairs: list[tuple[list, list]]) -> dict:
    if not pairs:
        return {"n": 0}
    all_hits = [h for h, _ in pairs]
    all_gt = [gt for _, gt in pairs]
    return {
        "n": len(pairs),
        "hit_rate_at_k": {k: metrics.hit_rate([h[:k] for h in all_hits], all_gt) for k in K_VALUES},
        "mrr": metrics.mean_reciprocal_rank(all_hits, all_gt),
    }


def _bucket_metrics(bucket_name: str, per_query: list[dict]) -> dict:
    if bucket_name == "neither":
        return {
            "n": len(per_query),
            "hit_rate": None,
            "note": "undefined for neither -- see confidence_separation instead (METHODOLOGY.md #3)",
        }

    pairs = [(pq["hits"], pq["query"]["ground_truth"]) for pq in per_query]
    result = _simple_hit_mrr(pairs)

    if bucket_name in ("faq", "policy_pdf") and per_query:
        result["source_routing_accuracy"] = sum(
            metrics.source_routing_correct(pq["hits"], bucket_name) for pq in per_query
        ) / len(per_query)

    incomplete = [pq for pq in per_query if pq["query"].get("retrievable_but_incomplete")]
    if incomplete:
        complete_only = [pq for pq in per_query if not pq["query"].get("retrievable_but_incomplete")]
        result["excluding_retrievable_but_incomplete"] = _simple_hit_mrr(
            [(pq["hits"], pq["query"]["ground_truth"]) for pq in complete_only]
        )
        result["retrievable_but_incomplete_ids"] = [pq["query"]["id"] for pq in incomplete]

    if bucket_name == "faq":
        high = [pq for pq in per_query if pq["query"].get("high_lexical_overlap")]
        low = [pq for pq in per_query if not pq["query"].get("high_lexical_overlap")]
        result["by_lexical_overlap"] = {
            "high_overlap": _simple_hit_mrr([(pq["hits"], pq["query"]["ground_truth"]) for pq in high]),
            "low_overlap": _simple_hit_mrr([(pq["hits"], pq["query"]["ground_truth"]) for pq in low]),
        }

    return result


def evaluate_cell(queries: list[dict], faq_store, policy_index) -> dict:
    per_query = [
        {"query": q, "hits": retrievers.query_combined(q["text"], MAX_K, faq_store, policy_index)}
        for q in queries
    ]

    buckets = {
        bucket_name: _bucket_metrics(
            bucket_name, [pq for pq in per_query if pq["query"]["expected_source"] == bucket_name]
        )
        for bucket_name in ("faq", "policy_pdf", "either", "neither")
    }

    neither_scores = [
        s for pq in per_query if pq["query"]["expected_source"] == "neither"
        and (s := metrics.top1_score(pq["hits"])) is not None
    ]
    should_hit_scores = [
        s for pq in per_query if pq["query"]["expected_source"] in ("faq", "policy_pdf", "either")
        and (s := metrics.top1_score(pq["hits"])) is not None
    ]

    return {
        "buckets": buckets,
        "confidence_separation": metrics.confidence_separation(neither_scores, should_hit_scores),
    }


def run_sweep(embeddings, work_dir: Path, queries: list[dict]) -> dict:
    cells = []

    for pdf_variant in SWEEP_PDF_VARIANTS:
        # Built once per variant, not per cell -- raw/cleaned page text
        # doesn't depend on chunk_size, ingestion_mode, or retriever_strategy.
        policy_raw_pages = corpus.build_policy_raw_pages(pdf_variant)

        for ingestion_mode in SWEEP_INGESTION_MODES:
            for chunk_size, chunk_overlap in SWEEP_CHUNK_CONFIGS:
                faq_docs = corpus.build_faq_documents(ingestion_mode, chunk_size, chunk_overlap)
                policy_chunks = corpus.build_policy_chunks(chunk_size, chunk_overlap, pdf_variant)
                mean_rows = (
                    corpus.mean_rows_per_uniform_chunk(faq_docs) if ingestion_mode == "uniform" else 1.0
                )

                for retriever_strategy in SWEEP_RETRIEVER_STRATEGIES:
                    cell_id = (
                        f"{pdf_variant}_pdf__{ingestion_mode}__"
                        f"cs{chunk_size}_ov{chunk_overlap}__{retriever_strategy}"
                    )
                    persist_dir = work_dir / cell_id

                    retriever_config = RetrieverConfig(
                        strategy=retriever_strategy,
                        k=MAX_K,
                        child_chunk_size=chunk_size,
                        child_chunk_overlap=chunk_overlap,
                        parent_chunk_size=chunk_size * 3,
                        parent_chunk_overlap=chunk_overlap * 3,
                    )

                    faq_store = retrievers.build_faq_index(faq_docs, embeddings, persist_dir / "faq")
                    policy_index = retrievers.build_policy_index(
                        retriever_strategy, policy_chunks, policy_raw_pages, embeddings,
                        persist_dir / "policy", retriever_config,
                    )

                    cell_result = evaluate_cell(queries, faq_store, policy_index)
                    cell_result.update({
                        "cell_id": cell_id,
                        "pdf_variant": pdf_variant,
                        "ingestion_mode": ingestion_mode,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "retriever_strategy": retriever_strategy,
                        "n_faq_documents": len(faq_docs),
                        "n_policy_chunks": len(policy_chunks),
                        "n_policy_raw_pages": len(policy_raw_pages),
                        "mean_rows_per_uniform_chunk": mean_rows,
                    })
                    cells.append(cell_result)

    return {"cells": cells}
