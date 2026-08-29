"""Tests for DEFAULT_MODEL constant and display name mappings."""

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._constants import (
    DEEP_RESEARCH_MODELS,
    DEFAULT_MODEL,
)


class TestDefaultModel:
    """Verify DEFAULT_MODEL is set to gpt-5.6-sol."""

    def test_default_model_is_gpt_5_6_sol(self):
        assert DEFAULT_MODEL == "gpt-5.6-sol"


class TestDisplayNames:
    """Verify display name mappings for GPT-5.4/5.5/5.6 family models."""

    def _get_display_name(self, model_id: str) -> str:
        provider = OpenAIProvider(api_key="test-key")
        return provider._model_id_to_display_name(model_id)

    def test_gpt_5_6_display_name(self):
        assert self._get_display_name("gpt-5.6") == "GPT 5.6"

    def test_gpt_5_6_sol_display_name(self):
        assert self._get_display_name("gpt-5.6-sol") == "GPT 5.6 Sol"

    def test_gpt_5_6_terra_display_name(self):
        assert self._get_display_name("gpt-5.6-terra") == "GPT 5.6 Terra"

    def test_gpt_5_6_luna_display_name(self):
        assert self._get_display_name("gpt-5.6-luna") == "GPT 5.6 Luna"

    def test_gpt_5_6_cyber_display_name(self):
        assert self._get_display_name("gpt-5.6-cyber") == "GPT 5.6 Cyber"

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


class TestDeepResearchModelsMembership:
    """X2: the stale 2025-dated snapshot ids were dropped; only the two
    un-dated aliases remain. Every consumer also does a
    startswith(("o3-deep-research", "o4-mini-deep-research")) check, so
    dated snapshots are still detected -- enumerating them was redundant."""

    def test_only_undated_aliases_remain(self):
        assert DEEP_RESEARCH_MODELS == frozenset(
            {"o3-deep-research", "o4-mini-deep-research"}
        )

    def test_stale_dated_entries_removed(self):
        assert "o3-deep-research-2025-06-26" not in DEEP_RESEARCH_MODELS
        assert "o4-mini-deep-research-2025-06-26" not in DEEP_RESEARCH_MODELS

    def test_dated_snapshot_still_detected_via_startswith(self):
        """Behavior-neutral: a dated snapshot not in the set is still a
        deep-research model via the startswith checks at every consumer."""
        snapshot = "o3-deep-research-2026-05-01"
        assert snapshot not in DEEP_RESEARCH_MODELS
        assert snapshot.startswith(("o3-deep-research", "o4-mini-deep-research"))
