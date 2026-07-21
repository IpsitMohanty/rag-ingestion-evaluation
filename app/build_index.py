"""Dev-only: (re)builds the prebuilt, committed Chroma index at
app/prebuilt_index/ using the real torch/HuggingFace embeddings backend
(same as phases 1/2).

Run this offline whenever the corpus or embedding model changes, then
commit the regenerated app/prebuilt_index/ directory. This script never
runs as part of the deployed app -- it needs requirements.txt/
requirements-dev.txt (torch, sentence-transformers), which
requirements-app.txt deliberately excludes.

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


if __name__ == "__main__":
    main()
