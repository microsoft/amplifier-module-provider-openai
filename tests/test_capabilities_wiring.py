"""Tests for wiring _capabilities.py into __init__.py (task-2).

Verifies that:
1. list_models() uses get_capabilities() for context windows, output limits, and tags.
2. _model_may_reason() delegates to caps.supports_reasoning instead of prefix matching.
3. No hardcoded capability values remain in list_models() or _model_may_reason().
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    return provider


def _fake_models_response(model_ids: list[str]):
    """Create a fake OpenAI models.list() response."""
    data = [SimpleNamespace(id=mid) for mid in model_ids]
    return SimpleNamespace(data=data)


def _make_list_models_provider(*model_ids: str) -> OpenAIProvider:
    """Create a provider wired to return specific model IDs from list_models()."""
    provider = _make_provider(filtered=False)
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(
        return_value=_fake_models_response(list(model_ids))
    )
    return provider


# ---------------------------------------------------------------------------
# list_models() wiring tests
# ---------------------------------------------------------------------------


class TestListModelsUsesGetCapabilities:
    """list_models() must delegate to get_capabilities() for per-model values."""

    def test_gpt5_model_uses_capabilities_context_window(self):
        """A gpt-5.1 model should get context_window from get_capabilities()."""
        provider = _make_list_models_provider("gpt-5.1")

        models = asyncio.run(provider.list_models())
        assert len(models) == 1

        caps = get_capabilities("gpt-5.1")
        assert models[0].context_window == caps.context_window
        assert models[0].max_output_tokens == caps.max_output_tokens

    def test_gpt5_model_uses_capabilities_tags(self):
        """A gpt-5.1 model should get capability tags from get_capabilities()."""
        provider = _make_list_models_provider("gpt-5.1")

        models = asyncio.run(provider.list_models())
        caps = get_capabilities("gpt-5.1")
        assert models[0].capabilities == list(caps.capability_tags)

    def test_deep_research_model_uses_capabilities(self):
        """Deep research models should get values from get_capabilities()."""
        provider = _make_list_models_provider("o3-deep-research")

        models = asyncio.run(provider.list_models())
        assert len(models) == 1

        caps = get_capabilities("o3-deep-research")
        assert models[0].context_window == caps.context_window
        assert models[0].max_output_tokens == caps.max_output_tokens
        assert models[0].capabilities == list(caps.capability_tags)

    def test_deep_research_defaults_include_background(self):
        """Deep research models should have background=True in defaults."""
        provider = _make_list_models_provider("o3-deep-research")

        models = asyncio.run(provider.list_models())
        assert models[0].defaults["background"] is True
        assert models[0].defaults["max_tokens"] == 32768

    def test_non_deep_research_defaults(self):
        """Non-deep-research models should have reasoning_effort defaults."""
        provider = _make_list_models_provider("gpt-5.1")

        models = asyncio.run(provider.list_models())
        assert models[0].defaults["reasoning_effort"] == "none"
        assert models[0].defaults["max_tokens"] == 16384

    def test_get_capabilities_is_called_for_each_model(self):
        """Verify get_capabilities() is actually called during list_models()."""
        provider = _make_list_models_provider("gpt-5.1")

        with patch(
            "amplifier_module_provider_openai.get_capabilities",
            wraps=get_capabilities,
        ) as mock_get_caps:
            asyncio.run(provider.list_models())
            mock_get_caps.assert_called_with("gpt-5.1")


# ---------------------------------------------------------------------------
# _model_may_reason() wiring tests
# ---------------------------------------------------------------------------


class TestModelMayReasonUsesCapabilities:
    """_model_may_reason() must delegate to caps.supports_reasoning."""

    def test_gpt5_model_may_reason(self):
        """gpt-5.1 supports reasoning via capabilities."""
        provider = _make_provider()
        assert provider._model_may_reason("gpt-5.1") is True

    def test_o_series_model_may_reason(self):
        """o4-mini supports reasoning via capabilities."""
        provider = _make_provider()
        assert provider._model_may_reason("o4-mini") is True

    def test_gpt5_mini_does_not_reason(self):
        """gpt-5-mini does NOT support reasoning per capabilities."""
        provider = _make_provider()
        caps = get_capabilities("gpt-5-mini")
        assert caps.supports_reasoning is False
        assert provider._model_may_reason("gpt-5-mini") is False

    def test_unknown_model_does_not_reason(self):
        """Unknown models (e.g., gpt-4.1-mini) don't support reasoning."""
        provider = _make_provider()
        assert provider._model_may_reason("gpt-4.1-mini") is False

    def test_empty_model_returns_false(self):
        """Empty model name returns False."""
        provider = _make_provider()
        assert provider._model_may_reason("") is False

    def test_delegates_to_capabilities(self):
        """Verify _model_may_reason() actually calls get_capabilities()."""
        provider = _make_provider()

        with patch(
            "amplifier_module_provider_openai.get_capabilities",
            wraps=get_capabilities,
        ) as mock_get_caps:
            provider._model_may_reason("gpt-5.4")
            mock_get_caps.assert_called_with("gpt-5.4")

    def test_deep_research_model_may_reason(self):
        """Deep research models support reasoning."""
        provider = _make_provider()
        assert provider._model_may_reason("o3-deep-research") is True
