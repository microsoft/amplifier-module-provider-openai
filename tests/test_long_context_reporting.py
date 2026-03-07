"""Tests for enable_long_context config flag and context_window reporting (task-2).

Verifies that:
1. Default (no flag) reports 272K context for GPT-5.4 (cost-safe).
2. enable_long_context=True reports 1,050K for GPT-5.4 (full capability).
3. Models without a pricing threshold are unaffected by the flag.
4. list_models() obeys the same reporting logic.
5. get_info() includes the enable_long_context ConfigField.
6. enable_long_context defaults to False when not in config.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _fake_models_response(model_ids: list[str]):
    """Create a fake OpenAI models.list() response."""
    data = [SimpleNamespace(id=mid) for mid in model_ids]
    return SimpleNamespace(data=data)


def _make_list_models_provider(**config_overrides) -> OpenAIProvider:
    """Create a provider wired to return specific model IDs from list_models()."""
    config = {"max_retries": 0, "filtered": False, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(
        return_value=_fake_models_response(["gpt-5.4"])
    )
    return provider


# ---------------------------------------------------------------------------
# enable_long_context attribute
# ---------------------------------------------------------------------------


class TestEnableLongContextDefault:
    """enable_long_context should default to False."""

    def test_enable_long_context_default_false(self):
        """Provider without config flag → enable_long_context is False."""
        provider = _make_provider()
        assert provider.enable_long_context is False

    def test_enable_long_context_true_when_set(self):
        """Provider with enable_long_context=True → attribute is True."""
        provider = _make_provider(enable_long_context=True)
        assert provider.enable_long_context is True


# ---------------------------------------------------------------------------
# get_info() context_window reporting
# ---------------------------------------------------------------------------


class TestGetInfoContextWindowReporting:
    """get_info() must report context_window based on the flag."""

    def test_default_reports_272k(self):
        """Provider with default config → get_info() shows context_window=272_000."""
        provider = _make_provider()
        info = provider.get_info()
        assert info.defaults["context_window"] == 272_000

    def test_long_context_enabled_reports_1050k(self):
        """Provider with enable_long_context=True → get_info() shows context_window=1_050_000."""
        provider = _make_provider(enable_long_context=True)
        info = provider.get_info()
        assert info.defaults["context_window"] == 1_050_000

    def test_no_threshold_model_unaffected_by_flag(self):
        """Model without pricing threshold → flag doesn't change reported context_window."""
        # gpt-5-mini has no long_context_pricing_threshold → always reports caps.context_window
        provider = _make_provider(enable_long_context=True, default_model="gpt-5-mini")
        info = provider.get_info()
        assert info.defaults["context_window"] == 128_000

    def test_no_threshold_model_default_flag(self):
        """Model without pricing threshold, default flag → reports caps.context_window."""
        provider = _make_provider(default_model="gpt-5-mini")
        info = provider.get_info()
        assert info.defaults["context_window"] == 128_000


# ---------------------------------------------------------------------------
# list_models() context_window reporting
# ---------------------------------------------------------------------------


class TestListModelsContextWindowReporting:
    """list_models() must report context_window based on the flag."""

    def test_list_models_default_272k(self):
        """list_models() without flag → GPT-5.4 model has context_window=272_000."""
        provider = _make_list_models_provider()
        models = asyncio.run(provider.list_models())
        gpt54_models = [m for m in models if m.id == "gpt-5.4"]
        assert len(gpt54_models) == 1
        assert gpt54_models[0].context_window == 272_000

    def test_list_models_long_context_1050k(self):
        """list_models() with enable_long_context=True → GPT-5.4 model has context_window=1_050_000."""
        provider = _make_list_models_provider(enable_long_context=True)
        models = asyncio.run(provider.list_models())
        gpt54_models = [m for m in models if m.id == "gpt-5.4"]
        assert len(gpt54_models) == 1
        assert gpt54_models[0].context_window == 1_050_000


# ---------------------------------------------------------------------------
# ConfigField presence
# ---------------------------------------------------------------------------


class TestConfigFieldPresent:
    """get_info() must include the enable_long_context ConfigField."""

    def test_config_field_present(self):
        """get_info() includes the enable_long_context ConfigField."""
        provider = _make_provider()
        info = provider.get_info()
        field_ids = [f.id for f in info.config_fields]
        assert "enable_long_context" in field_ids

    def test_config_field_is_boolean_type(self):
        """The enable_long_context ConfigField has field_type='boolean'."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "enable_long_context")
        assert field.field_type == "boolean"

    def test_config_field_not_required(self):
        """The enable_long_context ConfigField is not required."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "enable_long_context")
        assert field.required is False

    def test_config_field_default_false(self):
        """The enable_long_context ConfigField has default='false'."""
        provider = _make_provider()
        info = provider.get_info()
        field = next(f for f in info.config_fields if f.id == "enable_long_context")
        assert field.default == "false"
