"""Tests for DEFAULT_MODEL constant and display name mappings."""

from amplifier_module_provider_openai._constants import DEFAULT_MODEL
from amplifier_module_provider_openai import OpenAIProvider


class TestDefaultModel:
    """Verify DEFAULT_MODEL is set to gpt-5.4."""

    def test_default_model_is_gpt_5_4(self):
        assert DEFAULT_MODEL == "gpt-5.4"


class TestDisplayNames:
    """Verify display name mappings for GPT-5.4 family models."""

    def _get_display_name(self, model_id: str) -> str:
        provider = OpenAIProvider(api_key="test-key")
        return provider._model_id_to_display_name(model_id)

    def test_gpt_5_4_display_name(self):
        assert self._get_display_name("gpt-5.4") == "GPT 5.4"

    def test_gpt_5_4_pro_display_name(self):
        assert self._get_display_name("gpt-5.4-pro") == "GPT 5.4 Pro"

    def test_gpt_5_3_codex_display_name(self):
        assert self._get_display_name("gpt-5.3-codex") == "GPT-5.3 codex"

    def test_gpt_5_2_display_name(self):
        assert self._get_display_name("gpt-5.2") == "GPT 5.2"

    def test_gpt_5_2_pro_display_name(self):
        assert self._get_display_name("gpt-5.2-pro") == "GPT 5.2 Pro"

    def test_existing_gpt_5_1_display_name_preserved(self):
        """Existing entries should still work."""
        assert self._get_display_name("gpt-5.1") == "GPT 5.1"

    def test_existing_gpt_5_1_codex_display_name_preserved(self):
        assert self._get_display_name("gpt-5.1-codex") == "GPT-5.1 codex"

    def test_existing_gpt_5_mini_display_name_preserved(self):
        assert self._get_display_name("gpt-5-mini") == "GPT-5 mini"

    def test_existing_deep_research_display_name_preserved(self):
        assert self._get_display_name("o3-deep-research") == "o3 Deep Research"
