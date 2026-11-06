"""Capability tests for gpt-5.6 (Sol / Terra / Luna).

Documented total context and independent input ceiling are separate constraints.
All three tiers share one descriptor (prices are handled in _cost.py).
"""

import pytest

from amplifier_module_provider_openai._capabilities import get_capabilities

_GPT_56_TIERS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6"]


class TestGPT56Family:
    @pytest.mark.parametrize("model", _GPT_56_TIERS)
    def test_gpt_5_6_core_capabilities(self, model):
        caps = get_capabilities(model)
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
        assert caps.max_input_tokens == 922_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        # 272K standard-tier boundary (mirrors gpt-5.4). get_info() reports THIS as
        # the default context_window so the context manager compacts against the
        # standard-priced input window instead of the full input ceiling.
        assert caps.long_context_pricing_threshold == 272_000

    @pytest.mark.parametrize("model", _GPT_56_TIERS)
    def test_gpt_5_6_in_memory_retention_disabled(self, model):
        """Legacy retention is not an eligibility signal for the new cache API."""
        caps = get_capabilities(model)
        assert caps.supports_in_memory_retention is False

    def test_gpt_5_6_dated_snapshot_resolves(self):
        """A dated snapshot (what the API can echo in response.model) resolves the
        same capability facts via the legacy snapshot reader, not request policy."""
        caps = get_capabilities("gpt-5.6-sol-2026-07-09")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
        assert caps.max_input_tokens == 922_000
        assert caps.supports_in_memory_retention is False


class TestRegressionAfterGPT56:
    """Exact model facts must not leak into neighbouring compatibility branches."""

    def test_gpt_5_5_uses_documented_total_and_standard_price_threshold(self):
        caps = get_capabilities("gpt-5.5")
        assert caps.context_window == 1_050_000
        assert caps.long_context_pricing_threshold == 272_000
        assert caps.supports_in_memory_retention is False

    def test_gpt_5_4_unchanged(self):
        # gpt-5.4 keeps its own documented 1.05M total window.
        caps = get_capabilities("gpt-5.4")
        assert caps.context_window == 1_050_000
        assert caps.long_context_pricing_threshold == 272_000
        assert caps.supports_in_memory_retention is True

    def test_gpt_5_7_still_inherits_latest(self):
        """The exact minor==6 branch must not capture 5.7+ (assumed-latest catch-all).
        5.7 falls through to the minor>=4 branch, so it keeps that branch's 1.05M
        window and permissive in_memory default, without a documented input ceiling."""
        caps = get_capabilities("gpt-5.7")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
        assert caps.max_input_tokens is None
        assert caps.supports_in_memory_retention is True
