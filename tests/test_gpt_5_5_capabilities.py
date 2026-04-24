"""Capability tests for gpt-5.5 / gpt-5.5-pro.

Values verified against the live OpenAI API on 2026-04-24.
"""

from amplifier_module_provider_openai._capabilities import get_capabilities


class TestGPT55Family:
    def test_gpt_5_5(self):
        caps = get_capabilities("gpt-5.5")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_000_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        # Pricing-derived; no public API source.
        assert caps.long_context_pricing_threshold is None

    def test_gpt_5_5_pro(self):
        caps = get_capabilities("gpt-5.5-pro")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_000_000
        assert caps.supports_reasoning is True

    def test_gpt_5_5_dated_snapshot(self):
        caps = get_capabilities("gpt-5.5-2026-04-23")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_000_000

    def test_gpt_5_5_pro_dated_snapshot(self):
        caps = get_capabilities("gpt-5.5-pro-2026-04-23")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_000_000


class TestRegressionGPT54:
    """Adding the gpt-5.5 branch must not affect prior versions."""

    def test_gpt_5_4_unchanged(self):
        caps = get_capabilities("gpt-5.4")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
        assert caps.long_context_pricing_threshold == 272_000

    def test_gpt_5_3_unchanged(self):
        caps = get_capabilities("gpt-5.3-codex")
        assert caps.context_window == 400_000

    def test_gpt_5_9_still_inherits_latest(self):
        # The narrow minor==5 branch must not preempt 5.6+ (assumed-latest).
        caps = get_capabilities("gpt-5.9-turbo")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
