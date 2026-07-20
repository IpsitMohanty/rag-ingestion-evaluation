"""Embeddings adapter: one function returns a LangChain-compatible
Embeddings object. Swapping the backend is a config change
(config.EmbeddingConfig.backend), not a code change -- pipeline/ and cli.py
only ever see the object this returns, never a specific integration class.

The watsonx and openai branches are real, not stubs: they build a working
client from environment variables. Neither package is a phase 1 install
requirement (see requirements.txt) since the default path must run with no
API key and no account -- installing `langchain-ibm` or `langchain-openai`
and setting the relevant env vars is what "switching backend" means for
those two.
"""
from langchain_core.embeddings import Embeddings

from config import EmbeddingConfig


def get_embeddings(config: EmbeddingConfig) -> Embeddings:
    if config.backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=config.model_name)

    if config.backend == "watsonx":
        import os

        from langchain_ibm import WatsonxEmbeddings

        return WatsonxEmbeddings(
            model_id=config.model_name,
            url=os.environ["WATSONX_URL"],
            project_id=os.environ["WATSONX_PROJECT_ID"],
            apikey=os.environ["WATSONX_APIKEY"],
        )

    if config.backend == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=config.model_name)

    raise ValueError(f"Unknown embedding backend {config.backend!r}")
