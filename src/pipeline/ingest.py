"""Pipeline stage 1: discover corpus files and load them into raw (unsplit)
Documents, grouped by source_type.
"""
from pathlib import Path

from langchain_core.documents import Document

from adapters import ingestion as ingestion_adapter

# Phase 1 covers exactly the two source types in the brief's corpus, mapped
# by extension. A new format sharing an existing shape (e.g. a markdown FAQ
# export) extends this map; a genuinely new shape needs its own loader in
# adapters/ingestion.py plus a config.IngestionConfig entry.
EXTENSION_TO_SOURCE_TYPE = {
    ".pdf": "policy_pdf",
    ".json": "faq",
}


def discover_files(folder: Path) -> dict[str, list[Path]]:
    """Group files under `folder` by source_type, based on extension.
    Unrecognized extensions (e.g. the .csv FAQ export kept alongside the
    .json one for provenance -- see README) are silently skipped.
    """
    grouped: dict[str, list[Path]] = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        source_type = EXTENSION_TO_SOURCE_TYPE.get(path.suffix.lower())
        if source_type is None:
            continue
        grouped.setdefault(source_type, []).append(path)
    return grouped


def ingest_folder(folder: Path) -> dict[str, list[Document]]:
    """Load every recognized file under `folder`, grouped by source_type.
    No splitting happens here -- that's pipeline/split.py's job, since
    whether/how to split is a per-source-type config decision.
    """
    grouped_documents: dict[str, list[Document]] = {}
    for source_type, paths in discover_files(folder).items():
        for path in paths:
            grouped_documents.setdefault(source_type, []).extend(
                ingestion_adapter.load_source(source_type, path)
            )
    return grouped_documents
