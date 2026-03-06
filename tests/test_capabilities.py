"""Tests for ModelCapabilities dataclass and get_capabilities() lookup."""

import dataclasses

import pytest

from amplifier_module_provider_openai._capabilities import (
    ModelCapabilities,
    get_capabilities,
)


class TestModelCapabilitiesDataclass:
    """Test that ModelCapabilities is a proper frozen dataclass."""

    def test_frozen(self):
        caps = get_capabilities("gpt-5.4")
        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.family = "other"  # type: ignore[misc]

    def test_defaults(self):
        caps = ModelCapabilities(family="test")
        assert caps.family == "test"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is False
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        assert caps.capability_tags == ("tools", "streaming", "json_mode")


class TestGPT54Family:
    """Test GPT-5.4 family model capabilities."""

    def test_gpt_5_4(self):
        caps = get_capabilities("gpt-5.4")
        assert caps.family == "gpt-5"
        assert caps.context_window == 272_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        assert caps.capability_tags == (
            "tools",
            "reasoning",
            "streaming",
            "json_mode",
            "vision",
        )

    def test_gpt_5_4_pro(self):
        caps = get_capabilities("gpt-5.4-pro")
        assert caps.family == "gpt-5"
        assert caps.context_window == 272_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None

    def test_gpt_5_4_dated_snapshot(self):
        caps = get_capabilities("gpt-5.4-2026-03-05")
        assert caps.family == "gpt-5"
        assert caps.context_window == 272_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True


class TestGPT53CodexFamily:
    """Test GPT-5.3 codex model capabilities."""

    def test_gpt_5_3_codex(self):
        caps = get_capabilities("gpt-5.3-codex")
        assert caps.family == "gpt-5"
        assert caps.context_window == 400_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None


class TestGPT52Family:
    """Test GPT-5.2 family model capabilities."""

    def test_gpt_5_2(self):
        caps = get_capabilities("gpt-5.2")
        assert caps.family == "gpt-5"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort == "implicit"

    def test_gpt_5_2_pro(self):
        caps = get_capabilities("gpt-5.2-pro")
        assert caps.family == "gpt-5"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort == "implicit"


class TestGPT5MiniFamily:
    """Test GPT-5-mini model capabilities."""

    def test_gpt_5_mini(self):
        caps = get_capabilities("gpt-5-mini")
        assert caps.family == "gpt-5-mini"
        assert caps.context_window == 128_000
        assert caps.max_output_tokens == 64_000
        assert caps.supports_reasoning is False
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        assert caps.capability_tags == (
            "tools",
            "streaming",
            "json_mode",
            "vision",
            "fast",
        )

    def test_gpt_5_0_mini(self):
        """The gpt-5.0-mini variant routes to the gpt-5-mini family."""
        caps = get_capabilities("gpt-5.0-mini")
        assert caps.family == "gpt-5-mini"
        assert caps.context_window == 128_000
        assert caps.max_output_tokens == 64_000
        assert caps.supports_reasoning is False


class TestOSeriesFamily:
    """Test o-series model capabilities."""

    def test_o3(self):
        caps = get_capabilities("o3")
        assert caps.family == "o-series"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 100_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort == "implicit"
        assert caps.supports_vision is False
        assert caps.supports_streaming is True
        assert caps.capability_tags == ("tools", "reasoning", "streaming")

    def test_o4_mini(self):
        caps = get_capabilities("o4-mini")
        assert caps.family == "o-series"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 100_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort == "implicit"
        assert caps.supports_vision is False
        assert caps.supports_streaming is True


class TestDeepResearchFamily:
    """Test deep research model capabilities."""

    def test_o3_deep_research(self):
        caps = get_capabilities("o3-deep-research")
        assert caps.family == "deep-research"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is False
        assert caps.supports_streaming is False
        assert caps.capability_tags == ("deep_research", "web_search", "reasoning")

    def test_o4_mini_deep_research(self):
        caps = get_capabilities("o4-mini-deep-research")
        assert caps.family == "deep-research"
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is False
        assert caps.supports_streaming is False
        assert caps.capability_tags == ("deep_research", "web_search", "reasoning")


class TestForwardCompatibility:
    """Test forward compatibility with unknown model versions."""

    def test_unknown_gpt_5_version_assumes_latest(self):
        caps = get_capabilities("gpt-5.9-turbo")
        assert caps.family == "gpt-5"
        # Unknown gpt-5 version >= 5.4 assumes latest (5.4+) defaults
        assert caps.context_window == 272_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None

    def test_completely_unknown_model(self):
        caps = get_capabilities("claude-3-opus")
        assert caps.family == "unknown"
        # Conservative defaults
        assert caps.context_window == 200_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is False


class TestModelMayReason:
    """Test supports_reasoning as replacement for old prefix hack."""

    def test_gpt_5_4_may_reason(self):
        caps = get_capabilities("gpt-5.4")
        assert caps.supports_reasoning is True

    def test_o3_may_reason(self):
        caps = get_capabilities("o3")
        assert caps.supports_reasoning is True

    def test_gpt_5_mini_does_not_reason(self):
        caps = get_capabilities("gpt-5-mini")
        assert caps.supports_reasoning is False

    def test_gpt_4_1_mini_does_not_reason(self):
        caps = get_capabilities("gpt-4.1-mini")
        assert caps.family == "unknown"
        assert caps.supports_reasoning is False
