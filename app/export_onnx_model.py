"""Dev-only: exports sentence-transformers/all-MiniLM-L6-v2 to ONNX for
the deployed app's query-embedding path (see app/onnx_embeddings.py).

Mirrors the approach used in ibm-ai-eng/cnn-vit-land-classification
(torch checkpoint -> ONNX export -> onnxruntime in the deployed demo, to
drop torch from the app's memory footprint) -- same reasoning: torch
alone costs ~390MB RSS just to import, before any model weights load;
onnxruntime costs roughly a tenth of that.

This script needs torch + transformers (requirements-dev.txt /
requirements.txt) -- it is NOT part of the deployed app and never runs
there; requirements-app.txt deliberately excludes torch. Re-run this
only if the embedding model ever changes, then re-run
tests/test_onnx_parity.py before trusting the new export, then commit
the regenerated app/onnx_model/ directory.

Usage: python app/export_onnx_model.py
"""
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path(__file__).resolve().parent / "onnx_model"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Only input_ids/attention_mask are traced as real inputs -- token_type_ids
    # is fixed at all-zeros for our single-segment (no sentence-pair) use, which
    # matches how sentence-transformers itself calls this model.
    dummy = tokenizer(
        ["a sample sentence for tracing"], return_tensors="pt", padding=True, truncation=True, max_length=256
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(OUTPUT_DIR / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Exported {MODEL_NAME} to {OUTPUT_DIR}")
    print("Run tests/test_onnx_parity.py (RUN_NETWORK_TESTS=1 if the torch model isn't cached) before trusting this export.")


if __name__ == "__main__":
    main()
