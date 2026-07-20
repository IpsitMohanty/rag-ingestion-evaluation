import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DIR = REPO_ROOT / "reference"

# src/ modules use plain top-level imports (`from config import ...`,
# `from adapters import ingestion`) rather than a package-qualified
# `src.config` -- this is what makes `python src/cli.py` runnable directly
# (Python auto-adds a script's own directory to sys.path). Tests need the
# same thing done explicitly.
sys.path.insert(0, str(SRC_DIR))

# The eval/ package (phase 2 harness) is imported as `eval.metrics`,
# `eval.query_set`, etc. -- needs the repo root on sys.path regardless of
# pytest's own rootdir insertion.
sys.path.insert(0, str(REPO_ROOT))

RUN_NETWORK_TESTS = os.environ.get("RUN_NETWORK_TESTS") == "1"
requires_network = pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason=(
        "requires downloading a HuggingFace model over the network; "
        "set RUN_NETWORK_TESTS=1 to enable (not set in CI on purpose)"
    ),
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def faq_json_path(data_dir: Path) -> Path:
    return data_dir / "faq" / "all_faq.json"


@pytest.fixture(scope="session")
def faq_csv_path(data_dir: Path) -> Path:
    return data_dir / "faq" / "all_faq.csv"


@pytest.fixture(scope="session")
def policy_pdf_path(data_dir: Path) -> Path:
    return data_dir / "policy" / "Mission Saksham Anganwadi and Poshan 2.0 scheme guidelines.pdf"


@pytest.fixture(scope="session")
def default_config():
    from config import DEFAULT_CONFIG

    return DEFAULT_CONFIG


@pytest.fixture(scope="session")
def pdf_raw_docs(policy_pdf_path, default_config):
    """Session-scoped: PdfReader over a 1.5MB/77-page PDF is the suite's
    single slowest operation. Loading it once and sharing across tests
    keeps the whole suite comfortably under a minute."""
    from adapters import ingestion

    return ingestion.load_source("policy_pdf", policy_pdf_path)


@pytest.fixture(scope="session")
def pdf_chunks(pdf_raw_docs, default_config):
    from adapters import ingestion

    settings = default_config.ingestion.source_types["policy_pdf"]
    return ingestion.split_documents(pdf_raw_docs, settings)


@pytest.fixture
def fake_embeddings():
    """A deterministic, hash-based Embeddings implementation -- no model
    download, no network, no torch. Used to test the vectorstore/retriever
    adapters' *contract* (do they index and return k documents with
    metadata intact) independent of which real embeddings backend is
    plugged in.
    """
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=32)
