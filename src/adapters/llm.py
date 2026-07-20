"""LLM adapter: seam for phase 2's optional LLM-answer-quality arm and for
the multi_query/self_query retriever strategies -- none of which phase 1
implements or calls. Kept here, wired to real backends, so picking one up
later is a config change, not new adapter code.
"""
from langchain_core.language_models import BaseLanguageModel

from config import LLMConfig


def get_llm(config: LLMConfig) -> BaseLanguageModel:
    if config.backend is None:
        raise ValueError("LLMConfig.backend is not set -- no LLM is configured in phase 1")

    if config.backend == "huggingface":
        from langchain_huggingface import HuggingFacePipeline

        return HuggingFacePipeline.from_model_id(
            model_id=config.model_name or "gpt2",
            task="text-generation",
        )

    if config.backend == "watsonx":
        import os

        from langchain_ibm import WatsonxLLM

        return WatsonxLLM(
            model_id=config.model_name,
            url=os.environ["WATSONX_URL"],
            project_id=os.environ["WATSONX_PROJECT_ID"],
            apikey=os.environ["WATSONX_APIKEY"],
        )

    if config.backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model_name or "gpt-4o-mini")

    raise ValueError(f"Unknown LLM backend {config.backend!r}")
