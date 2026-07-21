"""Coverage: ONNX query-embedding parity with the torch/sentence-
transformers pipeline the committed Chroma index (app/prebuilt_index/)
was actually built with (see app/build_index.py).

Why this matters: the deployed app drops torch and embeds queries via
ONNX (app/onnx_embeddings.py) at runtime, searching against document
vectors that were computed once, offline, with torch. If the two
pipelines ever diverged, queries would land in a subtly different
vector space than the stored documents -- a silent retrieval-quality
regression with no error, no crash, nothing to notice except worse
results. This test is the guard against shipping that by accident.

Mirrors ibm-ai-eng/cnn-vit-land-classification's TestParityWithTorchDemo
(onnx vs torch prediction parity) convention: same model, two runtimes,
assert they agree within tolerance.

The actual comparison is gated behind requires_network (conftest.py),
same reasoning as test_embeddings_adapter.py's real-HuggingFace-backend
test: it needs the torch model available, which means downloading it on
a fresh checkout with no cache. onnxruntime/transformers (the ONNX side)
are lightweight enough to always be installed (requirements-dev.txt), so
collection never fails; only the comparison itself is network-gated.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import requires_network

APP_DIR = Path(__file__).resolve().parent.parent / "app"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(APP_DIR))

from onnx_embeddings import OnnxEmbeddings  # noqa: E402

# Essentially identical is the bar, not "close enough" -- see the
# module docstring on why a subtle divergence here is worse than an
# obvious one.
COSINE_SIMILARITY_TOLERANCE = 0.999


def _cosine_similarity(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def onnx_embeddings():
    return OnnxEmbeddings()


@pytest.fixture(scope="module")
def sample_queries() -> list[str]:
    """Every 5th query from the real, labeled evaluation set -- a spread
    across the faq/policy_pdf/either/neither buckets, not a hand-picked
    set of easy cases."""
    data = yaml.safe_load((EVAL_DIR / "queries.yaml").read_text(encoding="utf-8"))
    all_queries = [q["text"] for q in data["queries"]]
    sample = all_queries[::5]
    assert len(sample) >= 8, "expected the stride to sample a reasonable spread"
    return sample


@requires_network
def test_onnx_query_embeddings_match_torch_within_tolerance(onnx_embeddings, sample_queries):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    torch_model = sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    torch_vectors = torch_model.encode(sample_queries, normalize_embeddings=True)

    failures = []
    for query, torch_vector in zip(sample_queries, torch_vectors):
        onnx_vector = onnx_embeddings.embed_query(query)
        similarity = _cosine_similarity(onnx_vector, torch_vector)
        if similarity < COSINE_SIMILARITY_TOLERANCE:
            failures.append((query, similarity))

    assert not failures, (
        f"{len(failures)}/{len(sample_queries)} queries fell below "
        f"{COSINE_SIMILARITY_TOLERANCE} cosine similarity between ONNX and "
        "torch query embeddings. Do not ship the ONNX path if this fails -- "
        f"rebuild the prebuilt index with ONNX embeddings instead: {failures}"
    )


@requires_network
def test_onnx_embedding_dimension_matches_torch(onnx_embeddings):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    torch_model = sentence_transformers.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    torch_vector = torch_model.encode(["a test sentence"], normalize_embeddings=True)[0]
    onnx_vector = onnx_embeddings.embed_query("a test sentence")
    assert len(onnx_vector) == len(torch_vector) == 384


def test_onnx_embeddings_are_unit_normalized(onnx_embeddings):
    """Offline, always-on sanity check (no torch needed): the ONNX path's
    own L2-normalization step actually produces unit vectors, independent
    of whether it agrees with torch."""
    vector = np.asarray(onnx_embeddings.embed_query("a sample query"))
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-4


def test_onnx_embeddings_are_deterministic(onnx_embeddings):
    a = onnx_embeddings.embed_query("How many kinds of beneficiaries can be registered?")
    b = onnx_embeddings.embed_query("How many kinds of beneficiaries can be registered?")
    assert a == b
