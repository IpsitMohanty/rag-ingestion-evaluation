"""Run the answer-quality evaluation for baseline, corrective, and multi-agent RAG.

Usage:
    OPENAI_API_KEY=... python -m eval.run_answer_quality

This is intentionally a paid, local evaluation harness and is not run in CI.
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter  # noqa: E402
from adapters import llm as llm_adapter  # noqa: E402
from adapters.corrective_rag import (  # noqa: E402
    GENERATE_PROMPT,
    GeneratedAnswer,
    build_corrective_graph,
    get_corrective_structured_llms,
    run_corrective_query,
)
from adapters.multiagent_critic_rag import (  # noqa: E402
    build_multiagent_critic_graph,
    get_multiagent_structured_llms,
    run_multiagent_critic_query,
)
from config import DEFAULT_CONFIG  # noqa: E402
from eval.answer_quality import (  # noqa: E402
    CorrectnessGrade,
    FaithfulnessGrade,
    evaluate_variants,
)
from eval.query_set import add_reference_answers, load_query_set  # noqa: E402
from eval.retrievers import query_combined  # noqa: E402
from eval.retrievers import LexicalIndex, query_hybrid  # noqa: E402
from eval import corpus  # noqa: E402
from eval.corrective_sweep import K, build_representative_cell  # noqa: E402


def main() -> None:
    queries = add_reference_answers(load_query_set())
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)
    # Chroma can briefly retain an HNSW file handle on Windows after the
    # final query. Ignore only cleanup-time lock failures; evaluation errors
    # must still propagate and results are written after this block.
    with tempfile.TemporaryDirectory(
        prefix="answer-quality-", ignore_cleanup_errors=True
    ) as tmp:
        faq_store, policy_index = build_representative_cell(embeddings, Path(tmp))
        variants, correctness_llm, faithfulness_llm = build_variants(faq_store, policy_index)
        evaluated = evaluate_variants(
            queries, variants, correctness_llm, faithfulness_llm
        )

    serializable = {}
    for name, result in evaluated.items():
        serializable[name] = {
            "summary": result["summary"].__dict__,
            "results": [item.__dict__ for item in result["results"]],
        }
    output_path = REPO_ROOT / "results" / "answer_quality_results.json"
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


def build_variants(faq_store, policy_index):
    """Construct common LLM graders and runners over one retrieval cell."""
    config = DEFAULT_CONFIG.corrective_agentic
    base_llm = llm_adapter.get_llm(config.llm)
    correctness_llm = base_llm.with_structured_output(CorrectnessGrade)
    faithfulness_llm = base_llm.with_structured_output(FaithfulnessGrade)

    generate_llm = base_llm.with_structured_output(GeneratedAnswer)

    def baseline(query: str) -> dict:
        hits = query_combined(query, K, faq_store, policy_index)
        answer = generate_llm.invoke(
            GENERATE_PROMPT.format(
                query=query,
                excerpts="\n".join(hit.document.page_content for hit in hits),
            )
        ).answer
        return {"answer": answer, "abstained": False, "hits": hits}

    faq_lexical = LexicalIndex(
        corpus.build_faq_documents("format_aware", 500, 50), "faq"
    )
    policy_lexical = LexicalIndex(
        corpus.build_policy_chunks(500, 50, "cleaned"), "policy_pdf"
    )

    def hybrid(query: str) -> dict:
        hits = query_hybrid(
            query, K, faq_store, policy_index, faq_lexical, policy_lexical
        )
        answer = generate_llm.invoke(
            GENERATE_PROMPT.format(
                query=query,
                excerpts="\n".join(hit.document.page_content for hit in hits),
            )
        ).answer
        return {"answer": answer, "abstained": False, "hits": hits}

    route, judge, rewrite, generate, faithfulness = get_corrective_structured_llms(config)
    corrective_graph = build_corrective_graph(
        route, judge, rewrite, generate, faithfulness, faq_store, policy_index, k=K
    )

    multi_config = DEFAULT_CONFIG.multiagent_critic
    multi_graph = build_multiagent_critic_graph(
        *get_multiagent_structured_llms(multi_config),
        faq_store,
        policy_index,
        k=K,
        critic=multi_config.critic,
    )

    return (
        {
            "baseline": baseline,
            "hybrid": hybrid,
            "corrective": lambda query: run_corrective_query(
                corrective_graph, query, max_iterations=config.max_iterations
            ),
            "multiagent": lambda query: run_multiagent_critic_query(
                multi_graph, query, max_iterations=multi_config.max_iterations
            ),
        },
        correctness_llm,
        faithfulness_llm,
    )


if __name__ == "__main__":
    main()
