"""CLI: ingest a folder into the vector store, then query it.

    python src/cli.py ingest data/faq data/policy
    python src/cli.py query "How do I register a beneficiary?"

Run from the repo root; Python adds this script's own directory (src/) to
sys.path automatically, which is what makes the plain `from config import
...` / `from pipeline import ...` imports below resolve.
"""
import argparse
from dataclasses import replace
from pathlib import Path

from adapters import embeddings as embeddings_adapter
from adapters import vectorstore as vectorstore_adapter
from config import DEFAULT_CONFIG, Config
from pipeline import index as index_stage
from pipeline import ingest as ingest_stage
from pipeline import retrieve as retrieve_stage
from pipeline import split as split_stage


def run_ingest(folders: list[str], config: Config = DEFAULT_CONFIG) -> None:
    grouped_documents: dict = {}
    for folder in folders:
        for source_type, documents in ingest_stage.ingest_folder(Path(folder), config.ingestion).items():
            grouped_documents.setdefault(source_type, []).extend(documents)

    total_raw = sum(len(docs) for docs in grouped_documents.values())
    documents = split_stage.split_documents(grouped_documents, config.ingestion)
    index_stage.build_index(documents, config)

    print(f"Loaded {total_raw} raw document(s) across {len(grouped_documents)} source type(s):")
    for source_type, docs in grouped_documents.items():
        print(f"  {source_type}: {len(docs)} document(s)")
    print(
        f"Indexed {len(documents)} chunk(s) into collection "
        f"'{config.vectorstore.collection_name}' at {config.vectorstore.persist_directory}"
    )


def run_query(query: str, config: Config = DEFAULT_CONFIG) -> None:
    embeddings = embeddings_adapter.get_embeddings(config.embedding)
    store = vectorstore_adapter.get_vectorstore(config.vectorstore, embeddings)
    results = retrieve_stage.retrieve(query, store, config)

    print(f"Top {len(results)} result(s) for: {query!r}\n")
    for i, doc in enumerate(results, start=1):
        print(
            f"[{i}] source={doc.metadata.get('source')} "
            f"page={doc.metadata.get('page')} tab={doc.metadata.get('tab')}"
        )
        print(doc.page_content[:300])
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Ingestion Evaluation pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest one or more folders")
    ingest_parser.add_argument("folders", nargs="+")

    query_parser = subparsers.add_parser("query", help="Query the persisted vector store")
    query_parser.add_argument("query")
    query_parser.add_argument("--k", type=int, default=None)

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args.folders)
    elif args.command == "query":
        config = DEFAULT_CONFIG
        if args.k is not None:
            config = replace(config, retriever=replace(config.retriever, k=args.k))
        run_query(args.query, config)


if __name__ == "__main__":
    main()
