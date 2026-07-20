"""Coverage: the four pipeline stages (ingest -> split -> index -> retrieve)
wired together the way cli.py calls them, end to end, offline -- the real
embeddings backend is monkeypatched out for a deterministic fake so this
suite never touches the network or a downloaded model.
"""
import json
from dataclasses import replace

import pytest

from config import DEFAULT_CONFIG
from pipeline import index as index_stage
from pipeline import ingest as ingest_stage
from pipeline import retrieve as retrieve_stage
from pipeline import split as split_stage


@pytest.fixture
def sample_faq_folder(tmp_path):
    records = [
        {"tab": "Beneficiary", "subcategory": "General", "question": f"Question {i}?", "answer": f"Answer number {i}."}
        for i in range(5)
    ]
    faq_dir = tmp_path / "faq"
    faq_dir.mkdir()
    (faq_dir / "all_faq.json").write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


@pytest.fixture
def config(tmp_path, monkeypatch, fake_embeddings):
    monkeypatch.setattr(
        "adapters.embeddings.get_embeddings", lambda embedding_config: fake_embeddings
    )
    return replace(
        DEFAULT_CONFIG,
        vectorstore=replace(DEFAULT_CONFIG.vectorstore, persist_directory=tmp_path / "chroma"),
    )


def test_full_pipeline_ingest_to_retrieve(sample_faq_folder, config):
    grouped = ingest_stage.ingest_folder(sample_faq_folder)
    assert set(grouped) == {"faq"}
    assert len(grouped["faq"]) == 5

    documents = split_stage.split_documents(grouped, config.ingestion)
    assert len(documents) == 5  # FAQ never splits

    store = index_stage.build_index(documents, config)
    results = retrieve_stage.retrieve("Question 2", store, config)

    assert len(results) == config.retriever.k
    assert all("tab" in d.metadata for d in results)


def test_discover_files_skips_unrecognized_extensions(sample_faq_folder):
    (sample_faq_folder / "faq" / "all_faq.csv").write_text("tab,subcategory,question,answer\n", encoding="utf-8")
    grouped = ingest_stage.discover_files(sample_faq_folder)
    # .csv is intentionally not in EXTENSION_TO_SOURCE_TYPE (JSON is the
    # chosen default -- see README) so it must not show up here.
    assert grouped == {"faq": [sample_faq_folder / "faq" / "all_faq.json"]}
