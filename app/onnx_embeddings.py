"""ONNX-based Embeddings implementation for the deployed app's
query-time embedding path only -- no torch, no sentence-transformers.

The committed Chroma index (app/prebuilt_index/, see app/build_index.py)
was built once, offline, with the real torch/HuggingFace embeddings
backend -- this class is never used to build it. At query time, Chroma
only needs to embed the *query* string to search against those
already-stored vectors; it never re-embeds stored documents. That
asymmetry is what lets the deployed app drop torch entirely: as long as
query embeddings land in the same vector space as the torch-computed
document embeddings, retrieval is unaffected.

Uses the raw `tokenizers` library (Hugging Face's Rust tokenizer
bindings), not `transformers.AutoTokenizer` -- both read the same
tokenizer.json and produce identical token ids (transformers' "fast"
tokenizer IS this library under the hood), but importing the
`transformers` Python package alone costs ~340MB RSS just for its model/
config registry, before any model loads. `tokenizers` costs a few MB.
That gap is what actually determines whether this app fits Streamlit
Community Cloud's 1GB ceiling -- see the README's Phase 3 section.

The pipeline here (tokenize -> ONNX forward pass -> mean-pool over
tokens using the attention mask -> L2-normalize) reproduces exactly what
sentence-transformers/all-MiniLM-L6-v2 does internally, per its own
modules.json: Transformer -> Pooling(mode=mean) -> Normalize. Verified to
match within floating-point tolerance in tests/test_onnx_parity.py --
if you ever swap the embedding model, re-run app/export_onnx_model.py
and re-run that parity test before trusting this path again; a silent
mismatch would degrade retrieval without any visible error.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

ONNX_MODEL_DIR = Path(__file__).resolve().parent / "onnx_model"
MAX_SEQUENCE_LENGTH = 256  # matches all-MiniLM-L6-v2's sentence_bert_config.json


class OnnxEmbeddings(Embeddings):
    def __init__(self, model_dir: Path = ONNX_MODEL_DIR):
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        token_embeddings = self._session.run(
            ["last_hidden_state"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )[0]

        # Mean pooling over real (non-padding) tokens only.
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        mean_pooled = summed / counts

        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        return mean_pooled / np.clip(norms, 1e-9, None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0].tolist()
