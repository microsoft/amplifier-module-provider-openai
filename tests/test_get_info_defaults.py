"""Tests for get_info() using ModelCapabilities for defaults (task-5).

Verifies that:
1. get_info() returns context_window and max_output_tokens from get_capabilities().
2. get_info() uses self.default_model instead of a hardcoded model string.
3. For the default model (gpt-5.4), context_window is 272K (not old 400K).
"""

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._constants import DEFAULT_MODEL


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


class TestGetInfoUsesCapabilities:
    """get_info() must derive defaults from ModelCapabilities."""

    def test_default_model_context_window_matches_capabilities(self):
        """context_window in get_info() defaults should match get_capabilities() for the default model."""
        provider = _make_provider()
        info = provider.get_info()
        caps = get_capabilities(DEFAULT_MODEL)
        assert info.defaults["context_window"] == caps.context_window

    def test_default_model_max_output_tokens_matches_capabilities(self):
        """max_output_tokens in get_info() defaults should match get_capabilities() for the default model."""
        provider = _make_provider()
        info = provider.get_info()
        caps = get_capabilities(DEFAULT_MODEL)
        assert info.defaults["max_output_tokens"] == caps.max_output_tokens

    def test_default_model_is_gpt54_with_272k_context(self):
        """Default model gpt-5.4 should report 272K context, not the old 400K."""
        provider = _make_provider()
        info = provider.get_info()
        assert info.defaults["context_window"] == 272_000
        assert info.defaults["model"] == "gpt-5.4"

    def test_uses_self_default_model_not_hardcoded(self):
        """get_info() must use self.default_model, not a hardcoded model string."""
        provider = _make_provider(default_model="gpt-5.3-codex")
        info = provider.get_info()
        # gpt-5.3 family has 400K context
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
