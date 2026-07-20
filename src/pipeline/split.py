"""Pipeline stage 2: apply each source type's configured splitting
behavior. This is the stage phase 2's evaluation varies -- format-aware
(this, driven by config.IngestionConfig.source_types) vs uniform chunking
(forcing every source type through identical splitter settings) -- so it
stays a separate, config-driven stage rather than being folded into ingest.
"""
from langchain_core.documents import Document

from adapters import ingestion as ingestion_adapter
from config import IngestionConfig


def split_documents(
    grouped_documents: dict[str, list[Document]], config: IngestionConfig
) -> list[Document]:
    all_documents: list[Document] = []
    for source_type, documents in grouped_documents.items():
        settings = config.source_types[source_type]
        all_documents.extend(ingestion_adapter.split_documents(documents, settings))
    return all_documents
