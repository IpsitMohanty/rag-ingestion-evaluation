"""Entry point for the phase 2 config sweep.

    python -m eval.run_sweep

Requires the sentence-transformers/all-MiniLM-L6-v2 model to be available
locally (network on first run to download+cache it, fully offline after --
see README). Not run in CI; this is a several-minute, real-embeddings run,
not a test.
"""
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adapters import embeddings as embeddings_adapter
from config import DEFAULT_CONFIG

from eval.query_set import by_bucket, load_query_set
from eval.sweep import run_sweep


def main() -> None:
    embeddings = embeddings_adapter.get_embeddings(DEFAULT_CONFIG.embedding)
    queries = load_query_set()

    # ignore_cleanup_errors: Chroma's sqlite/HNSW files can still be held
    # open by the process on Windows when the block exits, which turns an
    # otherwise-successful run into a crash during temp-dir teardown.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        results = run_sweep(embeddings, Path(tmp), queries)

    results["query_set_summary"] = {
        "total": len(queries),
        "by_bucket": {b: len(qs) for b, qs in by_bucket(queries).items()},
    }
    results["embedding_model"] = DEFAULT_CONFIG.embedding.model_name

    out_path = REPO_ROOT / "results" / "sweep_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(results['cells'])} cells, {len(queries)} queries each)")


if __name__ == "__main__":
    main()
