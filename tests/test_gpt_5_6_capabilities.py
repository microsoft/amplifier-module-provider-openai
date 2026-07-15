"""Capability tests for gpt-5.6 (Sol / Terra / Luna).

Values verified against the live OpenAI API on 2026-07-14. All three tiers share
one capability descriptor (they differ in price/latency, handled in _cost.py).
"""

import pytest

from amplifier_module_provider_openai._capabilities import get_capabilities

_GPT_56_TIERS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"]


class TestGPT56Family:
    @pytest.mark.parametrize("model", _GPT_56_TIERS)
    def test_gpt_5_6_core_capabilities(self, model):
        caps = get_capabilities(model)
        assert caps.family == "gpt-5"
        assert caps.context_window == 900_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        # 272K standard-tier boundary (mirrors gpt-5.4). get_info() reports THIS as
        # the default context_window so the context manager compacts against the
        # safe window instead of the 1.05M max (prevents context_length_exceeded).
        assert caps.long_context_pricing_threshold == 272_000

    @pytest.mark.parametrize("model", _GPT_56_TIERS)
    def test_gpt_5_6_in_memory_retention_disabled(self, model):
        """gpt-5.6 rejects prompt_cache_retention='in_memory' ("compatible only with
        24h extended prompt caching" -- verified live 2026-07-14). This flag is the
        single source of truth for the drop in _drop_unsupported_in_memory_retention();
        pin False so a future edit can't silently revert the suppression."""
        caps = get_capabilities(model)
        assert caps.supports_in_memory_retention is False
        # 24h is the accepted default and must never be dropped for gpt-5.6.
        assert caps.supports_24h_retention is True

    def test_gpt_5_6_dated_snapshot_resolves(self):
        """A dated snapshot (what the API can echo in response.model) resolves the
        same via the version parser."""
        caps = get_capabilities("gpt-5.6-sol-2026-07-09")
        assert caps.family == "gpt-5"
        assert caps.context_window == 900_000
        assert caps.supports_in_memory_retention is False


class TestRegressionAfterGPT56:
    """Adding the minor==6 branch must not disturb neighbouring versions."""

    def test_gpt_5_5_unchanged(self):
        caps = get_capabilities("gpt-5.5")
        assert caps.context_window == 1_000_000
        assert caps.supports_in_memory_retention is False

    def test_gpt_5_4_unchanged(self):
        # gpt-5.4 keeps its own 1.05M window (the minor==6 900K ceiling is 5.6-only).
        caps = get_capabilities("gpt-5.4")
        assert caps.context_window == 1_050_000
        assert caps.long_context_pricing_threshold == 272_000
        assert caps.supports_in_memory_retention is True

    def test_gpt_5_7_still_inherits_latest(self):
        """The exact minor==6 branch must not capture 5.7+ (assumed-latest catch-all).
        5.7 falls through to the minor>=4 branch, so it keeps that branch's 1.05M
        window and permissive in_memory default -- NOT gpt-5.6's measured 900K."""
        caps = get_capabilities("gpt-5.7")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
        assert caps.supports_in_memory_retention is True
