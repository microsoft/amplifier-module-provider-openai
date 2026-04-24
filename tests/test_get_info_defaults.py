"""Tests for get_info() using ModelCapabilities for defaults.

Verifies that:
1. get_info() returns context_window and max_output_tokens from get_capabilities().
2. get_info() uses self.default_model instead of a hardcoded model string.
3. The long_context_pricing_threshold vs enable_long_context interaction still
   produces a cost-safe default when a model HAS a threshold (gpt-5.4).
   gpt-5.5 has no published threshold, so it reports its full 1M context
   regardless of the flag.
"""

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._constants import DEFAULT_MODEL


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class TestGetInfoUsesCapabilities:
    """get_info() must derive defaults from ModelCapabilities."""

    def test_default_model_reports_full_context(self):
        """With the default model (gpt-5.5), get_info() reports the full 1M
        context window because gpt-5.5 has no published pricing threshold.
        """
        provider = _make_provider()
        info = provider.get_info()
        caps = get_capabilities(DEFAULT_MODEL)
        assert caps.long_context_pricing_threshold is None
        assert info.defaults["context_window"] == caps.context_window
        assert info.defaults["context_window"] == 1_000_000

    def test_default_model_max_output_tokens_matches_capabilities(self):
        """max_output_tokens in get_info() must match the default model's
        capability value."""
        provider = _make_provider()
        info = provider.get_info()
        caps = get_capabilities(DEFAULT_MODEL)
        assert info.defaults["max_output_tokens"] == caps.max_output_tokens

    def test_default_model_id_is_gpt_5_5(self):
        provider = _make_provider()
        info = provider.get_info()
        assert info.defaults["model"] == "gpt-5.5"

    def test_enable_long_context_noop_for_gpt_5_5(self):
        """gpt-5.5 has no pricing threshold, so enable_long_context is a no-op
        for the context_window reported by get_info()."""
        provider = _make_provider(enable_long_context=True)
        info = provider.get_info()
        assert info.defaults["context_window"] == 1_000_000
        assert info.defaults["model"] == "gpt-5.5"

    def test_uses_self_default_model_not_hardcoded(self):
        """get_info() must use self.default_model, not a hardcoded string."""
        provider = _make_provider(default_model="gpt-5.3-codex")
        info = provider.get_info()
        # gpt-5.3 family has 400K context and no pricing threshold.
        caps = get_capabilities("gpt-5.3-codex")
        assert info.defaults["model"] == "gpt-5.3-codex"
        assert info.defaults["context_window"] == caps.context_window
        assert info.defaults["max_output_tokens"] == caps.max_output_tokens

    def test_static_defaults_unchanged(self):
        """Static defaults (max_tokens, temperature, timeout) remain unchanged."""
        provider = _make_provider()
        info = provider.get_info()
        assert info.defaults["max_tokens"] == 16384
        assert info.defaults["temperature"] is None
        assert info.defaults["timeout"] == 600.0


class TestGPT54CostSafeBehaviorRegression:
    """Regression: models with long_context_pricing_threshold (like gpt-5.4)
    still get the cost-safe default when they're selected explicitly.
    The behavior is threshold-based, not default-model-based."""

    def test_gpt_5_4_reports_272k_threshold_by_default(self):
        """gpt-5.4 (when selected explicitly) reports 272K cost-safe
        context, not the full 1,050K."""
        provider = _make_provider(default_model="gpt-5.4")
        info = provider.get_info()
        caps = get_capabilities("gpt-5.4")
        assert caps.long_context_pricing_threshold == 272_000
        assert info.defaults["context_window"] == 272_000
        assert info.defaults["model"] == "gpt-5.4"

    def test_gpt_5_4_reports_full_context_when_flag_set(self):
        """gpt-5.4 with enable_long_context=True reports the full 1,050K."""
        provider = _make_provider(default_model="gpt-5.4", enable_long_context=True)
        info = provider.get_info()
        assert info.defaults["context_window"] == 1_050_000
        assert info.defaults["model"] == "gpt-5.4"
