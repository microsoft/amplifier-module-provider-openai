"""Tests for ModelCapabilities dataclass and get_capabilities() lookup."""

import dataclasses

import pytest

from amplifier_module_provider_openai._capabilities import (
    ModelCapabilities,
    _detect_family,
    _detect_version,
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


class TestDetectFamily:
    """Unit tests for the _detect_family() helper."""

    def test_gpt_5_4(self):
        assert _detect_family("gpt-5.4") == "gpt-5"

    def test_gpt_5_4_pro(self):
        assert _detect_family("gpt-5.4-pro") == "gpt-5"

    def test_gpt_5_mini(self):
        assert _detect_family("gpt-5-mini") == "gpt-5-mini"

    def test_gpt_5_0_mini(self):
        assert _detect_family("gpt-5.0-mini") == "gpt-5-mini"

    def test_gpt_5_3_codex(self):
        assert _detect_family("gpt-5.3-codex") == "gpt-5"

    def test_o3_deep_research(self):
        assert _detect_family("o3-deep-research") == "deep-research"

    def test_o4_mini_deep_research(self):
        assert _detect_family("o4-mini-deep-research") == "deep-research"

    def test_o3(self):
        assert _detect_family("o3") == "o-series"

    def test_o4_mini(self):
        assert _detect_family("o4-mini") == "o-series"

    def test_unknown_model(self):
        assert _detect_family("claude-3-opus") == "unknown"


class TestDetectVersion:
    """Unit tests for the _detect_version() helper."""

    def test_gpt_5_4(self):
        assert _detect_version("gpt-5.4", "gpt-5") == (5, 4)

    def test_gpt_5_4_pro(self):
        assert _detect_version("gpt-5.4-pro", "gpt-5") == (5, 4)

    def test_gpt_5_3_codex(self):
        assert _detect_version("gpt-5.3-codex", "gpt-5") == (5, 3)

    def test_gpt_5_2(self):
        assert _detect_version("gpt-5.2", "gpt-5") == (5, 2)

    def test_gpt_5_mini_returns_5_0(self):
        # No .N in "gpt-5-mini" → minor defaults to 0
        assert _detect_version("gpt-5-mini", "gpt-5-mini") == (5, 0)

    def test_o3_returns_0_0(self):
        # Non-GPT family → always (0, 0)
        assert _detect_version("o3", "o-series") == (0, 0)

    def test_o4_mini_returns_0_0(self):
        assert _detect_version("o4-mini", "o-series") == (0, 0)

    def test_deep_research_returns_0_0(self):
        assert _detect_version("o3-deep-research", "deep-research") == (0, 0)

    def test_unparseable_gpt_returns_0_0(self):
        # Family starts with "gpt-" but regex won't match → (0, 0)
        assert _detect_version("gpt-unknown", "gpt-5") == (0, 0)
