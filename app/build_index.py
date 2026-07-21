"""Dev-only: (re)builds the prebuilt, committed Chroma index at
app/prebuilt_index/ using the real torch/HuggingFace embeddings backend
(same as phases 1/2).

Run this offline whenever the corpus or embedding model changes, then
commit the regenerated app/prebuilt_index/ directory. This script never
runs as part of the deployed app -- it needs requirements.txt/
requirements-dev.txt (torch, sentence-transformers), which
app/requirements.txt deliberately excludes.

*** REQUIRED AFTER REBUILDING: run the ONNX parity test manually. ***
tests/test_onnx_parity.py is network-gated (RUN_NETWORK_TESTS=1) like
the repo's other real-model tests, which means it is SKIPPED in CI --
CI going green after an index rebuild does NOT mean the ONNX query
embeddings still match. If this index was rebuilt with a different
embedding model, or app/onnx_model/ wasn't re-exported to match (see
app/export_onnx_model.py), the deployed app would silently search a
mismatched vector space: no crash, no error, just quietly worse
retrieval. Run this before committing a rebuilt index:

    RUN_NETWORK_TESTS=1 pytest tests/test_onnx_parity.py -v

Usage: python app/build_index.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_logic import PREBUILT_INDEX_DIR, build_vectorstore  # noqa: E402


def main() -> None:
    if PREBUILT_INDEX_DIR.exists():
        shutil.rmtree(PREBUILT_INDEX_DIR)

    build_vectorstore(persist_directory=PREBUILT_INDEX_DIR)
    print(f"Prebuilt index written to {PREBUILT_INDEX_DIR}")
    print(
        "\nREQUIRED NEXT STEP: RUN_NETWORK_TESTS=1 pytest tests/test_onnx_parity.py -v\n"
        "CI will NOT catch a mismatch here -- that test is network-gated and skipped in CI."
    )


if __name__ == "__main__":
    main()
