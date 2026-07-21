"""Coverage: config.py -- the per-source-type ingestion contract the rest
of the brief depends on (FAQ never splits, PDF always does, both switchable
from one place).
"""
import dataclasses

import pytest

from config import Config, DEFAULT_CONFIG, IngestionConfig


def test_default_config_has_faq_and_policy_pdf_source_types():
    assert set(DEFAULT_CONFIG.ingestion.source_types) == {"faq", "policy_pdf"}


def test_faq_source_type_does_not_split():
    faq = DEFAULT_CONFIG.ingestion.source_types["faq"]
    assert faq.split is False
    assert faq.splitter is None


def test_policy_pdf_source_type_splits_with_configured_sizes():
    pdf = DEFAULT_CONFIG.ingestion.source_types["policy_pdf"]
    assert pdf.split is True
    assert pdf.splitter.chunk_size > 0
    assert 0 <= pdf.splitter.chunk_overlap < pdf.splitter.chunk_size


def test_default_config_uses_raw_pdf_not_cleaned():
    """Adding the cleaned-PDF variant must not silently change existing
    CLI/app behavior -- raw stays the default until the swept numbers
    (results/ANALYSIS.md) say otherwise."""
    assert DEFAULT_CONFIG.ingestion.source_types["policy_pdf"].clean is False


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.retriever.k = 99


def test_per_source_type_settings_are_independently_switchable():
    """The brief's core design constraint: switching one source type's
    splitting behavior must not require touching the other's config."""
    custom = dataclasses.replace(
        DEFAULT_CONFIG.ingestion,
        source_types={
            **DEFAULT_CONFIG.ingestion.source_types,
            "faq": dataclasses.replace(DEFAULT_CONFIG.ingestion.source_types["faq"], split=True),
        },
    )
    assert custom.source_types["faq"].split is True
    # policy_pdf's settings are untouched by changing faq's
    assert custom.source_types["policy_pdf"] == DEFAULT_CONFIG.ingestion.source_types["policy_pdf"]


def test_default_config_is_a_config_instance():
    assert isinstance(DEFAULT_CONFIG, Config)
    assert isinstance(DEFAULT_CONFIG.ingestion, IngestionConfig)
