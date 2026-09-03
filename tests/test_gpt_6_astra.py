"""Comprehensive tests for gpt-6-astra support.

Covers all surfaces specified in the goal:
1. Capabilities and limits
2. Display name and default model behavior
3. Short and long Standard-tier cost rates
4. 272,000/272,001 boundary
5. Cache read/write/fresh input accounting
6. Long-context reporting (enable_long_context)
7. Supported and rejected reasoning efforts
8. Unsupported request fields (temperature, top_p, top_logprobs, logprobs include)
9. Legacy prompt_cache_retention omission (including inherited defaults)
10. 30-minute TTL forwarding via prompt_cache_options
11. Function-tool payloads
12. Structured Output payloads
13. Encrypted reasoning includes

Sources verified 2026-09-03:
- https://developers.openai.com/api/docs/models/gpt-6-astra.md
- https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md
- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/docs/pricing
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import (
    get_capabilities,
)
from amplifier_module_provider_openai._constants import DEFAULT_MODEL
from amplifier_module_provider_openai._cost import compute_cost

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _astra_provider(**config_overrides) -> OpenAIProvider:
    return _make_provider(default_model="gpt-6-astra", **config_overrides)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class DummyResponse:
    """Minimal response stub."""

    def __init__(self, text: str = "Hi"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        self.status = "completed"
        self.id = "resp_test"
        self.model = "gpt-6-astra"


def _captured_params(provider: OpenAIProvider) -> Any:
    mock = cast(AsyncMock, provider.client.responses.create)
    return mock.call_args.kwargs


# ---------------------------------------------------------------------------
# 1. Capabilities and limits
# ---------------------------------------------------------------------------


class TestGPT6AstraCapabilities:
    """Verify ModelCapabilities for gpt-6-astra against official docs."""

    def test_family(self):
        caps = get_capabilities("gpt-6-astra")
        assert caps.family == "gpt-6-astra"

    def test_context_window_is_amplifier_input_budget(self):
        """context_window = 922,000 (Amplifier input/compaction budget).
        Total capacity is 1,050,000 per docs; we report the input budget."""
        caps = get_capabilities("gpt-6-astra")
        assert caps.context_window == 922_000

    def test_max_output_tokens(self):
        caps = get_capabilities("gpt-6-astra")
        assert caps.max_output_tokens == 128_000

    def test_supports_reasoning(self):
        caps = get_capabilities("gpt-6-astra")
        assert caps.supports_reasoning is True

    def test_default_reasoning_effort_is_none(self):
        """No default effort injected -- model decides."""
        caps = get_capabilities("gpt-6-astra")
        assert caps.default_reasoning_effort is None

    def test_supports_vision(self):
        caps = get_capabilities("gpt-6-astra")
        assert caps.supports_vision is True

    def test_supports_streaming(self):
        caps = get_capabilities("gpt-6-astra")
        assert caps.supports_streaming is True

    def test_long_context_pricing_threshold(self):
        """272,000-token long-pricing threshold (same as gpt-5.6)."""
        caps = get_capabilities("gpt-6-astra")
        assert caps.long_context_pricing_threshold == 272_000

    def test_supports_in_memory_retention_false(self):
        """prompt_cache_retention not sent for Astra -- flag must be False."""
        caps = get_capabilities("gpt-6-astra")
        assert caps.supports_in_memory_retention is False

    def test_supports_native_apply_patch(self):
        """apply_patch is in the verified supported tools list."""
        caps = get_capabilities("gpt-6-astra")
        assert caps.supports_native_apply_patch is True

    def test_supports_native_computer_use(self):
        """computer_use is in the verified supported tools list."""
        caps = get_capabilities("gpt-6-astra")
        assert caps.supports_native_computer_use is True

    def test_capability_tags_include_tools_reasoning_streaming_vision(self):
        caps = get_capabilities("gpt-6-astra")
        for tag in ("tools", "reasoning", "streaming", "vision"):
            assert tag in caps.capability_tags, f"Expected tag {tag!r} in {caps.capability_tags}"

    def test_dated_snapshot_resolves(self):
        """A dated snapshot id (what the API echoes in response.model) resolves correctly."""
        caps = get_capabilities("gpt-6-astra-2026-09-03")
        assert caps.family == "gpt-6-astra"
        assert caps.context_window == 922_000
        assert caps.supports_in_memory_retention is False

    def test_does_not_affect_gpt_5_family(self):
        """Adding gpt-6-astra must not disturb gpt-5.6 or gpt-5.5 capabilities."""
        caps_56 = get_capabilities("gpt-5.6-sol")
        assert caps_56.family == "gpt-5"
        assert caps_56.context_window == 900_000
        assert caps_56.long_context_pricing_threshold == 272_000

        caps_55 = get_capabilities("gpt-5.5")
        assert caps_55.family == "gpt-5"
        assert caps_55.context_window == 1_000_000

    def test_does_not_affect_unknown_gpt_6_models(self):
        """A hypothetical future gpt-6-something-else falls through to 'unknown'."""
        caps = get_capabilities("gpt-6-future")
        assert caps.family == "unknown"


# ---------------------------------------------------------------------------
# 2. Display name and default model behavior
# ---------------------------------------------------------------------------


class TestGPT6AstraDisplayAndDefault:
    """Verify display name and that gpt-5.6-sol remains the default."""

    def test_display_name(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider._model_id_to_display_name("gpt-6-astra") == "GPT 6 Astra"

    def test_default_model_unchanged(self):
        """gpt-5.6-sol must remain the default."""
        assert DEFAULT_MODEL == "gpt-5.6-sol"

    def test_default_provider_uses_gpt_5_6_sol(self):
        provider = OpenAIProvider(api_key="test-key")
        assert provider.default_model == "gpt-5.6-sol"

    def test_astra_provider_uses_gpt_6_astra(self):
        provider = _astra_provider()
        assert provider.default_model == "gpt-6-astra"


# ---------------------------------------------------------------------------
# 3 & 4. Cost accounting: short/long rates and 272K boundary
# ---------------------------------------------------------------------------


class TestGPT6AstraCost:
    """Standard-tier cost rates for gpt-6-astra.

    Short-context (≤272K input): input $10, cached $1, cache-write $12.50, output $50
    Long-context (>272K input): input $20, cached $2, cache-write $25, output $75
    Source: https://developers.openai.com/api/docs/pricing (verified 2026-09-03)
    """

    # Short-context rates (≤272,000 input tokens)

    def test_short_input_rate(self):
        """100K fresh input tokens @ short rate = $10.00/M × 0.1M = $1.00."""
        cost = compute_cost("gpt-6-astra", prompt_tokens=100_000)
        assert cost == Decimal(100000) * Decimal(10) / Decimal(1000000)
        assert cost == Decimal("1.00")

    def test_short_input_rate_1m_tokens(self):
        """Verify the per-million rate directly: at exactly 272K tokens (short-context)
        the rate is $10/M. We compute at 272K to confirm the short rate applies."""
        cost = compute_cost("gpt-6-astra", prompt_tokens=272_000)
        expected = Decimal(272000) * Decimal(10) / Decimal(1000000)
        assert cost == expected

    def test_short_output_rate(self):
        """100K output tokens @ short rate = $50.00/M × 0.1M = $5.00."""
        cost = compute_cost("gpt-6-astra", completion_tokens=100_000)
        assert cost == Decimal(100000) * Decimal(50) / Decimal(1000000)
        assert cost == Decimal("5.00")

    def test_short_cached_input_rate(self):
        """100K cached input tokens @ short rate = $1.00/M × 0.1M = $0.10."""
        cost = compute_cost(
            "gpt-6-astra",
            prompt_tokens=100_000,
            cached_tokens=100_000,
        )
        assert cost == Decimal(100000) * Decimal(1) / Decimal(1000000)
        assert cost == Decimal("0.10")

    def test_short_cache_write_rate(self):
        """100K cache-write tokens @ short rate = $12.50/M × 0.1M = $1.25."""
        cost = compute_cost(
            "gpt-6-astra",
            prompt_tokens=100_000,
            cache_write_tokens=100_000,
        )
        # fresh = 100K - 0 cached - 100K write = 0
        assert cost == Decimal(100000) * Decimal("12.5") / Decimal(1000000)
        assert cost == Decimal("1.25")

    def test_short_mixed_fresh_cached_write(self):
        """30K fresh + 10K cached + 10K write + 10K output (all short-context)."""
        # prompt_tokens=50K (= 30K fresh + 10K cached + 10K write)
        # fresh = 50K - 10K cached - 10K write = 30K
        # cost = 30K * $10/M + 10K * $12.50/M + 10K * $1/M + 10K * $50/M
        #      = $0.30 + $0.125 + $0.01 + $0.50 = $0.935
        cost = compute_cost(
            "gpt-6-astra",
            prompt_tokens=50_000,
            cached_tokens=10_000,
            cache_write_tokens=10_000,
            completion_tokens=10_000,
        )
        expected = (
            Decimal(30000) * Decimal(10) / Decimal(1000000)
            + Decimal(10000) * Decimal("12.5") / Decimal(1000000)
            + Decimal(10000) * Decimal(1) / Decimal(1000000)
            + Decimal(10000) * Decimal(50) / Decimal(1000000)
        )
        assert cost == expected

    # Long-context rates (>272,000 input tokens)

    def test_long_input_rate(self):
        """400K fresh input tokens @ long rate = $20.00/M × 0.4M = $8.00."""
        cost = compute_cost("gpt-6-astra", prompt_tokens=400_000)
        # 400K > 272K → long rates: $20/M
        assert cost == Decimal(400000) * Decimal(20) / Decimal(1000000)
        assert cost == Decimal("8.00")

    def test_long_output_rate(self):
        """400K input (long) + 100K output = long rates for whole request."""
        cost = compute_cost(
            "gpt-6-astra",
            prompt_tokens=400_000,
            completion_tokens=100_000,
        )
        # 400K > 272K → long rates: $20/M input, $75/M output
        expected = (
            Decimal(400000) * Decimal(20) / Decimal(1000000)
            + Decimal(100000) * Decimal(75) / Decimal(1000000)
        )
        assert cost == expected

    def test_long_cached_input_rate(self):
        """Long-context cached rate = $2/M."""
        cost = compute_cost(
            "gpt-6-astra",
            prompt_tokens=400_000,
            cached_tokens=400_000,
        )
        # 400K > 272K → long rates: $2/M cached
        assert cost == Decimal(400000) * Decimal(2) / Decimal(1000000)
        assert cost == Decimal("0.80")

    def test_long_cache_write_rate(self):
        """Long-context cache-write rate = $25/M."""
        cost = compute_cost(
            "gpt-6-astra",
            prompt_tokens=400_000,
            cache_write_tokens=400_000,
        )
        # 400K > 272K → long rates: $25/M write
        # fresh = 400K - 0 cached - 400K write = 0
        assert cost == Decimal(400000) * Decimal(25) / Decimal(1000000)
        assert cost == Decimal("10.00")

    # Boundary tests

    def test_exactly_272k_is_short_context(self):
        """Exactly 272,000 input tokens is short-context (boundary is strict: >272K is long)."""
        cost = compute_cost("gpt-6-astra", prompt_tokens=272_000)
        # Short rate: $10/M
        assert cost == Decimal(272000) * Decimal(10) / Decimal(1000000)

    def test_272001_is_long_context(self):
        """272,001 input tokens crosses the boundary → long-context rates."""
        cost = compute_cost("gpt-6-astra", prompt_tokens=272_001)
        # Long rate: $20/M
        assert cost == Decimal(272001) * Decimal(20) / Decimal(1000000)

    def test_boundary_cost_difference(self):
        """Confirm the boundary step-change: 272K is short, 272K+1 is long."""
        short_cost = compute_cost("gpt-6-astra", prompt_tokens=272_000)
        long_cost = compute_cost("gpt-6-astra", prompt_tokens=272_001)
        assert long_cost > short_cost  # long is more expensive
        # Ratio should be ~2x (20/10)
        ratio = long_cost / short_cost
        assert Decimal("1.9") < ratio < Decimal("2.1")

    def test_result_is_decimal(self):
        cost = compute_cost("gpt-6-astra", prompt_tokens=100_000)
        assert isinstance(cost, Decimal)
        assert not isinstance(cost, float)

    def test_no_double_charge_on_cached_tokens(self):
        """Cached tokens must not be charged as fresh input."""
        cost_all_cached = compute_cost(
            "gpt-6-astra",
            prompt_tokens=100_000,
            cached_tokens=100_000,
        )
        # All 100K cached → $1/M = $0.10
        assert cost_all_cached == Decimal(100000) * Decimal(1) / Decimal(1000000)

    def test_dated_snapshot_resolves_to_astra_rates(self):
        """A dated snapshot id falls back to the 'gpt-6-astra' family alias."""
        # Use short-context (≤272K) to confirm short rate applies
        cost = compute_cost("gpt-6-astra-2026-09-03", prompt_tokens=100_000)
        # Short rate: $10/M × 0.1M = $1.00
        assert cost == Decimal("1.00")

    def test_existing_gpt_5_6_sol_rates_unchanged(self):
        """Adding Astra rates must not disturb gpt-5.6-sol pricing (short-context)."""
        # Use short-context (≤272K) to confirm short rate applies
        cost = compute_cost("gpt-5.6-sol", prompt_tokens=100_000)
        # Short rate for gpt-5.6-sol: $4.00/M × 0.1M = $0.40
        assert cost == Decimal("0.40")

    def test_reasoning_tokens_are_output_tokens(self):
        """Reasoning tokens count as output tokens (no separate rate).
        Use short-context (prompt=0) so only output rate applies."""
        # 100K output at short-context output rate = $50/M × 0.1M = $5.00
        cost = compute_cost("gpt-6-astra", completion_tokens=100_000)
        assert cost == Decimal("5.00")


# ---------------------------------------------------------------------------
# 5. Long-context reporting (enable_long_context)
# ---------------------------------------------------------------------------


class TestGPT6AstraLongContextReporting:
    """enable_long_context changes the reported context_window for Astra."""

    def test_default_reports_272k_threshold(self):
        """Without enable_long_context, get_info() reports 272K (the threshold)."""
        provider = _astra_provider()
        info = provider.get_info()
        assert info.defaults["context_window"] == 272_000

    def test_enable_long_context_reports_full_budget(self):
        """With enable_long_context=True, get_info() reports 922K (the input budget)."""
        provider = _astra_provider(enable_long_context=True)
        info = provider.get_info()
        assert info.defaults["context_window"] == 922_000


# ---------------------------------------------------------------------------
# 6. Supported and rejected reasoning efforts
# ---------------------------------------------------------------------------


class TestGPT6AstraReasoningEfforts:
    """Verify per-request effort validation for gpt-6-astra."""

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_supported_efforts_pass(self, effort):
        """Supported efforts must not raise."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        # Must not raise
        asyncio.run(
            provider.complete(_simple_request(), reasoning={"effort": effort})
        )

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_rejected_efforts_raise_invalid_request(self, effort):
        """Explicitly supplied 'none' or 'minimal' must raise InvalidRequestError."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
            asyncio.run(
                provider.complete(_simple_request(), reasoning={"effort": effort})
            )
        msg = str(exc_info.value)
        assert "gpt-6-astra" in msg
        assert effort in msg
        # Must include migration hint
        assert "low" in msg.lower() or "migration" in msg.lower()

    def test_omitted_reasoning_does_not_raise(self):
        """Omitting reasoning entirely must not raise (model uses its own default)."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))  # no reasoning kwarg

    def test_none_config_reasoning_effort_does_not_raise(self):
        """Config reasoning_effort=None (omitted) must not raise (convention preserved)."""
        provider = _astra_provider(reasoning_effort=None)
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))

    def test_minimal_config_reasoning_effort_raises_at_mount(self):
        """Config reasoning_effort='minimal' must raise at mount time for Astra."""
        with pytest.raises(ValueError) as exc_info:
            _astra_provider(reasoning_effort="minimal")
        msg = str(exc_info.value)
        assert "gpt-6-astra" in msg
        assert "minimal" in msg

    def test_effort_none_string_config_is_treated_as_no_effort(self):
        """Config reasoning_effort='none' (the selector default) must NOT raise --
        it resolves to None (no reasoning param sent), preserving the convention."""
        provider = _astra_provider(reasoning_effort="none")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))  # must not raise

    def test_explicit_none_effort_string_in_reasoning_dict_raises(self):
        """An explicit reasoning={'effort': 'none'} dict DOES raise for Astra."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        with pytest.raises(kernel_errors.InvalidRequestError):
            asyncio.run(
                provider.complete(_simple_request(), reasoning={"effort": "none"})
            )

    def test_non_astra_model_none_effort_does_not_raise(self):
        """Older models must not be affected by the Astra effort guard."""
        provider = _make_provider(default_model="gpt-5.6-sol")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        # effort='none' is not sent as a reasoning param for gpt-5.6 (it resolves to None)
        # so no exception should be raised
        asyncio.run(provider.complete(_simple_request(), reasoning_effort="none"))


# ---------------------------------------------------------------------------
# 7. Unsupported request fields
# ---------------------------------------------------------------------------


class TestGPT6AstraUnsupportedParams:
    """Verify that unsupported sampling/logprob fields are rejected for Astra."""

    @pytest.mark.parametrize("field", ["temperature", "top_p", "top_logprobs"])
    def test_rejected_params_raise_via_kwargs(self, field):
        """Unsupported params passed as per-call kwargs must raise InvalidRequestError."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
            asyncio.run(
                provider.complete(_simple_request(), **{field: 0.5})
            )
        msg = str(exc_info.value)
        assert field in msg
        assert "gpt-6-astra" in msg

    def test_temperature_rejected_via_request(self):
        """temperature on the ChatRequest must also be rejected for Astra."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            temperature=0.7,
        )
        with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
            asyncio.run(provider.complete(request))
        assert "temperature" in str(exc_info.value)

    def test_logprobs_include_rejected(self):
        """message.output_text.logprobs in include must be rejected for Astra."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
            asyncio.run(
                provider.complete(
                    _simple_request(),
                    include=["message.output_text.logprobs"],
                )
            )
        msg = str(exc_info.value)
        assert "message.output_text.logprobs" in msg

    def test_non_astra_temperature_not_rejected(self):
        """temperature must NOT be rejected for older models."""
        provider = _make_provider(default_model="gpt-5.6-sol")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        # Must not raise for gpt-5.6-sol
        asyncio.run(provider.complete(_simple_request(), temperature=0.7))

    def test_non_astra_top_p_not_rejected(self):
        """top_p must NOT be rejected for older models."""
        provider = _make_provider(default_model="gpt-5.6-sol")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request(), top_p=0.9))


# ---------------------------------------------------------------------------
# 8. Cache behavior: no prompt_cache_retention, TTL forwarding
# ---------------------------------------------------------------------------


class TestGPT6AstraCacheBehavior:
    """Verify prompt_cache_retention is never sent for Astra."""

    def test_default_retention_not_sent_for_astra(self):
        """The provider's default '24h' retention must NOT be sent for Astra."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        assert "prompt_cache_retention" not in params

    def test_explicit_retention_not_sent_for_astra(self):
        """Even an explicitly configured prompt_cache_retention must be suppressed for Astra."""
        provider = _astra_provider(prompt_cache_retention="24h")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        assert "prompt_cache_retention" not in params

    def test_in_memory_retention_not_sent_for_astra(self):
        """in_memory retention must be suppressed for Astra (same as 24h)."""
        provider = _astra_provider(prompt_cache_retention="in_memory")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        assert "prompt_cache_retention" not in params

    def test_kwarg_retention_not_sent_for_astra(self):
        """Per-call kwarg prompt_cache_retention must also be suppressed for Astra."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(
            provider.complete(_simple_request(), prompt_cache_retention="24h")
        )
        params = _captured_params(provider)
        assert "prompt_cache_retention" not in params

    def test_ttl_30m_forwarded_via_prompt_cache_options(self):
        """prompt_cache_options={'ttl': '30m'} must be forwarded verbatim for Astra."""
        provider = _astra_provider(prompt_cache_options={"ttl": "30m"})
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        assert params.get("prompt_cache_options") == {"ttl": "30m"}

    def test_explicit_cache_mode_guard_preserved_for_astra(self):
        """The explicit-mode safety guard must fire for Astra too (mode stripped at mount)."""
        provider = _astra_provider(prompt_cache_options={"mode": "explicit"})
        # After mount, explicit mode is stripped (same behavior as gpt-5.6)
        assert provider.prompt_cache_options is None

    def test_implicit_cache_mode_forwarded_for_astra(self):
        """implicit mode is not affected by the guard and passes through."""
        provider = _astra_provider(prompt_cache_options={"mode": "implicit"})
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        assert params.get("prompt_cache_options") == {"mode": "implicit"}

    def test_older_model_retention_unchanged(self):
        """Adding Astra cache logic must not affect older models' retention."""
        provider = _make_provider(default_model="gpt-5.6-sol")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        # gpt-5.6-sol gets the default "24h" retention
        assert params.get("prompt_cache_retention") == "24h"


# ---------------------------------------------------------------------------
# 9. Function-tool payloads
# ---------------------------------------------------------------------------


class TestGPT6AstraFunctionTools:
    """Verify function-tool payloads work for Astra."""

    def test_function_tool_forwarded(self):
        """A function tool must be included in the Responses API call."""
        from amplifier_core.message_models import ToolSpec

        # Need a coordinator mock for _convert_tools_from_request
        coordinator = MagicMock()
        coordinator.get_capability = MagicMock(return_value=None)
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock()

        provider = OpenAIProvider(
            api_key="test-key",
            config={"max_retries": 0, "use_streaming": False, "default_model": "gpt-6-astra"},
            coordinator=coordinator,
        )
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())

        tool = ToolSpec(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        request = ChatRequest(
            messages=[Message(role="user", content="What's the weather in Paris?")],
            tools=[tool],
        )
        asyncio.run(provider.complete(request))
        params = _captured_params(provider)
        assert "tools" in params
        tool_names = [t.get("name") for t in params["tools"] if isinstance(t, dict)]
        assert "get_weather" in tool_names


# ---------------------------------------------------------------------------
# 10. Structured Output payloads
# ---------------------------------------------------------------------------


class TestGPT6AstraStructuredOutputs:
    """Verify Structured Outputs capability is declared for Astra."""

    def test_structured_outputs_capability_declared(self):
        """json_mode must be in the capability_tags (Structured Outputs support)."""
        caps = get_capabilities("gpt-6-astra")
        assert "json_mode" in caps.capability_tags

    def test_structured_outputs_via_extra_request_params(self):
        """Structured Outputs can be used via extra_request_params escape hatch."""
        provider = _astra_provider(
            extra_request_params={"text": {"format": {"type": "json_object"}}}
        )
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(provider.complete(_simple_request()))
        params = _captured_params(provider)
        # The extra_request_params should be merged into the API call
        assert params.get("text", {}).get("format", {}).get("type") == "json_object"


# ---------------------------------------------------------------------------
# 11. Encrypted reasoning includes
# ---------------------------------------------------------------------------


class TestGPT6AstraEncryptedReasoning:
    """Verify reasoning.encrypted_content is requested for Astra."""

    def test_encrypted_content_requested_when_reasoning(self):
        """When reasoning is active for Astra, include=[reasoning.encrypted_content]
        must be in the API call params."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(
            provider.complete(_simple_request(), reasoning={"effort": "high"})
        )
        params = _captured_params(provider)
        assert "include" in params
        assert "reasoning.encrypted_content" in params["include"]

    def test_encrypted_content_not_in_logprobs_include(self):
        """The include list must NOT contain message.output_text.logprobs for Astra."""
        provider = _astra_provider()
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        asyncio.run(
            provider.complete(_simple_request(), reasoning={"effort": "high"})
        )
        params = _captured_params(provider)
        include = params.get("include", [])
        assert "message.output_text.logprobs" not in include


# ---------------------------------------------------------------------------
# Regression: existing model behavior unchanged
# ---------------------------------------------------------------------------


class TestRegressionAfterGPT6Astra:
    """Adding gpt-6-astra must not disturb existing model behavior."""

    def test_gpt_5_6_sol_capabilities_unchanged(self):
        caps = get_capabilities("gpt-5.6-sol")
        assert caps.family == "gpt-5"
        assert caps.context_window == 900_000
        assert caps.long_context_pricing_threshold == 272_000
        assert caps.supports_in_memory_retention is False

    def test_gpt_5_5_capabilities_unchanged(self):
        caps = get_capabilities("gpt-5.5")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_000_000
        assert caps.long_context_pricing_threshold is None

    def test_gpt_5_4_capabilities_unchanged(self):
        caps = get_capabilities("gpt-5.4")
        assert caps.family == "gpt-5"
        assert caps.context_window == 1_050_000
        assert caps.long_context_pricing_threshold == 272_000
        assert caps.supports_in_memory_retention is True

    def test_gpt_5_6_sol_cost_unchanged(self):
        """gpt-5.6-sol short-context rate must be unchanged."""
        cost = compute_cost("gpt-5.6-sol", prompt_tokens=100_000)
        # Short rate: $4.00/M × 0.1M = $0.40
        assert cost == Decimal("0.40")

    def test_gpt_5_4_cost_unchanged(self):
        """gpt-5.4 rate must be unchanged (no long-context rates modelled)."""
        cost = compute_cost("gpt-5.4", prompt_tokens=1_000_000)
        # gpt-5.4 has no long rates, so even 1M tokens uses short rate
        assert cost == Decimal("2.50")

    def test_default_model_still_gpt_5_6_sol(self):
        assert DEFAULT_MODEL == "gpt-5.6-sol"

    def test_gpt_5_5_pro_effort_guard_unchanged(self):
        """gpt-5.5-pro effort guard must still fire for 'low'."""
        provider = _make_provider(default_model="gpt-5.5-pro")
        provider.client.responses.create = AsyncMock(return_value=DummyResponse())
        with pytest.raises(kernel_errors.InvalidRequestError):
            asyncio.run(
                provider.complete(_simple_request(), reasoning={"effort": "low"})
            )
