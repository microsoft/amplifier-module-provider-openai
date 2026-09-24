"""Offline regression tests for the exact ``gpt-6-sol`` / ``gpt-6-luna`` integration.

Mirrors ``test_gpt_6_astra.py`` for the shared GPT-6 machinery, and adds
coverage for the two documented deltas from Astra:

- Sol/Luna accept ``reasoning.effort="none"`` (Astra does not).
- Sol/Luna only reject sampling fields (temperature/top_p/logprobs/
  top_logprobs) while reasoning is ACTIVE -- i.e. when effort is anything
  other than "none". Astra rejects them unconditionally.

Every OTHER GPT-6 wire-compatibility rule is shared across all three exact
model IDs (Astra, Sol, Luna): the legacy `prompt_cache_retention` field is
always dropped (never sent, one-time warning), `prompt_cache_options.ttl`
accepts only `None`/`"30m"`, and `include: ["message.output_text.logprobs"]`
is rejected outgoing. Those are asserted here too, not just for Astra.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message, ToolSpec
from pydantic import ValidationError

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._cost import compute_cost


def _provider(model: str, **config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": model,
            "max_retries": 0,
            "use_streaming": False,
            **config,
        },
    )


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _response(
    *,
    model: str,
    status: str = "completed",
    service_tier: str | None = "default",
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_gpt6",
        model=model,
        status=status,
        service_tier=service_tier,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        ),
    )


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
class TestCapabilities:
    def test_capabilities_and_safe_input_budget(self, model: str) -> None:
        caps = get_capabilities(model)

        assert caps.family == model
        assert caps.context_window == 922_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        assert caps.supports_native_apply_patch is True
        assert caps.supports_native_computer_use is True
        assert caps.long_context_pricing_threshold == 272_000
        assert caps.supports_in_memory_retention is False

    def test_only_exact_id_receives_family_capabilities(self, model: str) -> None:
        assert get_capabilities(f"{model}-2099-01-01").family != model

    def test_long_context_reporting_defaults_to_standard_pricing_budget(
        self, model: str
    ) -> None:
        assert _provider(model).get_info().defaults["context_window"] == 272_000
        assert (
            _provider(model, enable_long_context=True).get_info().defaults[
                "context_window"
            ]
            == 922_000
        )

    def test_list_models_uses_friendly_name_and_reported_limits(self, model: str) -> None:
        provider = _provider(model, hide_dated_models=False)
        provider._client = AsyncMock()
        provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(id=model)])
        )

        models = asyncio.run(provider.list_models())

        assert len(models) == 1
        expected_name = "GPT 6 Sol" if model == "gpt-6-sol" else "GPT 6 Luna"
        assert models[0].display_name == expected_name
        assert models[0].context_window == 272_000
        assert models[0].max_output_tokens == 128_000


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize(
    "effort", ["none", "low", "medium", "high", "xhigh", "max"]
)
def test_sol_luna_allow_documented_reasoning_efforts(model: str, effort: str) -> None:
    provider = _provider(model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request(), reasoning={"effort": effort}))

    assert provider.client.responses.create.call_args.kwargs["reasoning"]["effort"] == effort


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"])
def test_gpt6_family_sends_no_default_effort_when_omitted(model: str) -> None:
    """No exact GPT-6 model ID gets an explicit 'medium' stamped by Amplifier
    when the caller supplies a reasoning dict with no 'effort' key -- each
    model's own documented server-side default applies untouched (Astra:
    no default documented; Sol/Luna: "medium"). Regression for a
    pre-GPT-6-family `!= "gpt-6-astra"` check that would otherwise sweep
    Sol/Luna into the legacy "stamp medium" bucket meant for older models.
    """
    provider = _provider(model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request(), reasoning={"summary": "detailed"}))

    call = provider.client.responses.create.call_args.kwargs
    assert "effort" not in call["reasoning"]


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_reject_undocumented_reasoning_effort(model: str) -> None:
    provider = _provider(model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request(), reasoning={"effort": "minimal"}))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.2},
        {"extra_request_params": {"top_p": 0.5}},
        {"extra_request_params": {"top_logprobs": 1}},
        {"extra_request_params": {"logprobs": True}},
    ],
)
def test_sol_luna_reject_sampling_fields_while_reasoning_is_active(
    model: str, kwargs: dict
) -> None:
    """Default (omitted) effort is 'reasoning active' -- sampling fields still
    reject just like Astra when reasoning.effort is not explicitly 'none'."""
    provider = _provider(model, **kwargs)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request()))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_allow_sampling_fields_when_effort_is_none(model: str) -> None:
    provider = _provider(model, temperature=0.3)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request(), reasoning={"effort": "none"}))

    call = provider.client.responses.create.call_args.kwargs
    assert call["reasoning"]["effort"] == "none"
    assert call["temperature"] == 0.3


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_null_sampling_param_is_always_omitted(model: str) -> None:
    """A None value is dropped regardless of effort -- it means 'no opinion',
    not 'send null', matching Astra's existing behavior."""
    provider = _provider(model, extra_request_params={"temperature": None})
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request()))

    assert "temperature" not in provider.client.responses.create.call_args.kwargs


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
@pytest.mark.parametrize("configured_retention", ["in_memory", "24h"])
def test_sol_luna_legacy_retention_is_always_dropped_like_astra(
    model: str, configured_retention: str, caplog
) -> None:
    """Sol/Luna reject `prompt_cache_retention` outright, exactly like Astra --
    NOT the generic `_drop_unsupported_in_memory_retention` path (which only
    drops the specific "in_memory" value and would let "24h" through)."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(model, prompt_cache_retention=configured_retention)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request()))

    assert "prompt_cache_retention" not in provider.client.responses.create.call_args.kwargs
    warnings = [
        record for record in caplog.records if "Dropping prompt_cache_retention" in record.message
    ]
    assert len(warnings) == 1
    assert model in warnings[0].getMessage()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_implicit_default_retention_is_neither_sent_nor_warned(
    model: str, caplog
) -> None:
    """The module's own "24h" default (never explicitly configured/per-call)
    must not trip the drop warning -- mirrors Astra's equivalent test."""
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(model)
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    asyncio.run(provider.complete(_request()))

    assert "prompt_cache_retention" not in provider.client.responses.create.call_args.kwargs
    assert not [
        record for record in caplog.records if "prompt_cache_retention" in record.message
    ]


def test_legacy_retention_warning_is_tracked_per_model_not_globally(caplog) -> None:
    """One provider instance can serve different GPT-6 models per call (the
    model is resolved per-request, not fixed at construction). The warn-once
    guard must be keyed per model -- if it were a single shared flag, tripping
    it for the first model would silently suppress the warning a second,
    different model deserves on its own first offending call.
    """
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider("gpt-6-astra", prompt_cache_retention="24h")
    provider.client.responses.create = AsyncMock(return_value=_response(model="gpt-6-astra"))

    asyncio.run(provider.complete(_request()))
    asyncio.run(provider.complete(_request(), model="gpt-6-sol"))
    asyncio.run(provider.complete(_request(), model="gpt-6-luna"))

    warnings = [
        record for record in caplog.records if "Dropping prompt_cache_retention" in record.message
    ]
    warned_models = {record.getMessage().split("for ")[-1].split(";")[0] for record in warnings}
    assert warned_models == {"gpt-6-astra", "gpt-6-sol", "gpt-6-luna"}


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_accept_only_documented_cache_ttl(model: str) -> None:
    provider = _provider(model, prompt_cache_options={"ttl": "30m"})
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))
    asyncio.run(provider.complete(_request()))
    assert provider.client.responses.create.call_args.kwargs["prompt_cache_options"] == {
        "ttl": "30m"
    }

    invalid = _provider(model, prompt_cache_options={"ttl": "24h"})
    invalid.client.responses.create = AsyncMock(return_value=_response(model=model))
    with pytest.raises(kernel_errors.InvalidRequestError, match="30m"):
        asyncio.run(invalid.complete(_request()))
    invalid.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_reject_output_text_logprobs_include(model: str) -> None:
    provider = _provider(model, extra_request_params={"include": ["message.output_text.logprobs"]})
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request()))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_native_apply_patch_tool_serializes_bare_wire_shape(model: str) -> None:
    """Native apply_patch is a passthrough tool type -- declared with no extra
    fields, same as every other model that supports it (see `_constants.py`
    NATIVE_TOOL_TYPES and `_convert_tools_from_request`)."""
    provider = _provider(model)
    provider._apply_patch_native = True
    provider.client.responses.create = AsyncMock(return_value=_response(model=model))
    apply_patch = ToolSpec(
        name="apply_patch",
        description="Apply a patch",
        parameters={"type": "object", "properties": {}},
    )

    asyncio.run(
        provider.complete(
            ChatRequest(
                model=model,
                messages=[Message(role="user", content="edit a file")],
                tools=[apply_patch],
            )
        )
    )

    sent_tools = provider.client.responses.create.call_args.kwargs["tools"]
    assert sent_tools == [{"type": "apply_patch"}]


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_native_computer_tool_serializes_bare_wire_shape(model: str) -> None:
    """Native computer-use tool accepts ZERO declaration fields on the wire
    (see `_constants.py` NATIVE_TOOL_TYPES docstring) -- confirm Sol/Luna
    follow the same bare-dict serialization as Astra, with no WebSocket
    transport or extra config fields introduced. A `computer` tool routes
    through the raw-JSON fallback path (`with_raw_response.create`), same as
    every other model that declares it -- see `_params_declare_computer_tool`."""
    provider = _provider(model)
    coordinator = SimpleNamespace(
        hooks=SimpleNamespace(emit=AsyncMock()),
        get_capability=lambda _: None,
    )
    provider.coordinator = coordinator
    raw_body = {
        "id": "resp_gpt6_computer",
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "computer_call",
                "call_id": "call_screenshot",
                "actions": [{"type": "screenshot"}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        },
    }

    class _RawSDKResponse:
        async def parse(self) -> None:
            raise ValidationError.from_exception_data(
                "Response",
                [
                    {
                        "type": "missing",
                        "loc": ("output", 0, "pending_safety_checks"),
                        "input": {},
                    }
                ],
            )

        async def json(self) -> dict:
            return raw_body

    client = MagicMock()
    with_raw_create = AsyncMock(return_value=_RawSDKResponse())
    client.responses.with_raw_response.create = with_raw_create
    provider._client = client
    computer = ToolSpec(
        name="computer",
        description="Take a screenshot",
        parameters={"type": "object", "properties": {}},
    )
    computer.type = "computer"

    result = asyncio.run(
        provider.complete(
            ChatRequest(
                model=model,
                messages=[Message(role="user", content="take a screenshot")],
                tools=[computer],
            )
        )
    )

    assert result.tool_calls[0].arguments == {"actions": [{"type": "screenshot"}]}
    sent_tools = with_raw_create.call_args.kwargs["tools"]
    assert sent_tools == [{"type": "computer"}]
    # Confirm no native WebSocket transport was engaged for this request --
    # only the raw-response HTTP fallback was used, never a socket/stream API.
    assert with_raw_create.await_count == 1


@pytest.mark.parametrize(
    ("model", "tier", "expected"),
    [
        ("gpt-6-sol", "default", Decimal("0.0002")),
        ("gpt-6-sol", "flex", Decimal("0.0001")),
        ("gpt-6-sol", "priority", Decimal("0.0004")),
        ("gpt-6-luna", "default", Decimal("0.00001")),
        ("gpt-6-luna", "fast", Decimal("0.00002")),
    ],
)
def test_sol_luna_pricing_uses_actual_service_tier(
    model: str, tier: str, expected: Decimal
) -> None:
    assert compute_cost(model, prompt_tokens=100, service_tier=tier) == expected


def test_sol_pricing_long_boundary_and_unknown_tier() -> None:
    assert compute_cost("gpt-6-sol", prompt_tokens=272_000) == Decimal("0.544")
    assert compute_cost("gpt-6-sol", prompt_tokens=272_001) == Decimal("1.088004")
    assert compute_cost("gpt-6-sol", prompt_tokens=100, service_tier=None) is None
    assert compute_cost("gpt-6-sol", prompt_tokens=100, service_tier="scale") is None
    assert compute_cost("gpt-6-sol-2099-01-01", prompt_tokens=100) is None


def test_luna_pricing_long_boundary_and_unknown_tier() -> None:
    assert compute_cost("gpt-6-luna", prompt_tokens=272_000) == Decimal("0.0272")
    assert compute_cost("gpt-6-luna", prompt_tokens=272_001) == Decimal("0.0544002")
    assert compute_cost("gpt-6-luna", prompt_tokens=100, service_tier=None) is None
    assert compute_cost("gpt-6-luna-2099-01-01", prompt_tokens=100) is None


# ---------------------------------------------------------------------------
# Near-miss exact-ID: an almost-but-not-quite model id must NOT receive any
# GPT-6-specific validation (effort set, sampling-field gate) -- only the
# curated exact IDs get it. Mirrors test_astra's equivalent capability check,
# but exercises the validation function directly rather than just capabilities.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "near_miss_model", ["gpt-6-sol-2099-01-01", "gpt-6-sol-preview", "gpt-6-lunar"]
)
def test_near_miss_model_id_skips_gpt_6_specific_validation(near_miss_model: str) -> None:
    """An undocumented/near-miss id is NOT one of the exact GPT-6 IDs, so it
    must fall through to generic behavior -- no effort-set restriction, no
    sampling-field gate, no forced-drop of prompt_cache_retention."""
    provider = _provider(near_miss_model, temperature=0.5)
    provider.client.responses.create = AsyncMock(
        return_value=_response(model=near_miss_model)
    )

    # "minimal" is rejected for every real GPT-6 model but must be ACCEPTED
    # (passed straight through) for a near-miss id -- it is not in
    # _GPT_6_ALLOWED_EFFORTS, so _validate_gpt_6_params() no-ops.
    asyncio.run(provider.complete(_request(), reasoning={"effort": "minimal"}))

    call = provider.client.responses.create.call_args.kwargs
    assert call["reasoning"]["effort"] == "minimal"
    assert call["temperature"] == 0.5


# ---------------------------------------------------------------------------
# Structured outputs (text.format json_schema) -- passthrough via
# extra_request_params, same escape hatch every other model uses. There is no
# GPT-6-specific structured-output code path to test beyond confirming the
# generic passthrough reaches the wire unchanged for these exact model IDs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_structured_output_json_schema_passes_through_to_wire(model: str) -> None:
    json_schema_format = {
        "type": "json_schema",
        "name": "answer",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
        "strict": True,
    }
    provider = _provider(model, extra_request_params={"text": {"format": json_schema_format}})
    provider.client.responses.create = AsyncMock(
        return_value=_response(model=model, status="completed")
    )

    asyncio.run(provider.complete(_request()))

    call = provider.client.responses.create.call_args.kwargs
    assert call["text"] == {"format": json_schema_format}


# ---------------------------------------------------------------------------
# Full usage accounting -- cache read, cache write, and reasoning tokens
# together, exercised through the real complete() path (not just compute_cost
# in isolation), confirming usage AND cost are both correct end to end.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_full_usage_accounting_cache_and_reasoning_tokens(model: str) -> None:
    provider = _provider(model)
    usage = SimpleNamespace(
        input_tokens=100_000,
        output_tokens=20_000,
        output_tokens_details=SimpleNamespace(reasoning_tokens=15_000),
        input_tokens_details=SimpleNamespace(cached_tokens=10_000, cache_write_tokens=20_000),
    )
    response = SimpleNamespace(
        id="resp_usage",
        model=model,
        status="completed",
        service_tier="default",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        incomplete_details=None,
        usage=usage,
    )
    provider.client.responses.create = AsyncMock(return_value=response)

    result = asyncio.run(provider.complete(_request()))

    # `usage.input_tokens` already contains cached-read tokens (per OpenAI's
    # raw usage semantics) but has cache_write_tokens subtracted out, since
    # cache-write is billed at its own rate rather than the ordinary input
    # rate (100_000 - 20_000); the gross total is separately available via
    # `total_tokens`.
    assert result.usage.input_tokens == 80_000
    assert result.usage.total_tokens == 100_000
    assert result.usage.cache_read_tokens == 10_000
    assert result.usage.cache_write_tokens == 20_000
    assert result.usage.output_tokens == 20_000
    assert result.usage.reasoning_tokens == 15_000
    expected_cost = compute_cost(
        model,
        prompt_tokens=100_000,
        completion_tokens=20_000,
        cached_tokens=10_000,
        cache_write_tokens=20_000,
        service_tier="default",
    )
    assert result.usage.cost_usd == expected_cost
    assert expected_cost is not None


# ---------------------------------------------------------------------------
# Streaming text / reasoning / tool-call parsing -- exercise the real stream
# event loop (not just the terminal-response recovery path already covered
# by test_misalignment_policy_stream.py) against an exact GPT-6 model.
# ---------------------------------------------------------------------------


class _FakeHooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class _FakeCoordinator:
    def __init__(self) -> None:
        self.hooks = _FakeHooks()


class _FakeEventStream:
    def __init__(self, events: list, response: object) -> None:
        self._events = list(events)
        self._response = SimpleNamespace(headers={})
        self._final_response = response
        self._pos = 0

    def __aiter__(self) -> _FakeEventStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._pos >= len(self._events):
            raise StopAsyncIteration
        ev = self._events[self._pos]
        self._pos += 1
        return ev

    async def get_final_response(self) -> object:
        return self._final_response


class _MockStreamContext:
    def __init__(self, stream: object) -> None:
        self._stream = stream

    async def __aenter__(self) -> object:
        return self._stream

    async def __aexit__(self, *args: object) -> None:
        return None


def _event(type_str: str, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(type=type_str, **kwargs)


def _item(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_streaming_parses_text_reasoning_and_tool_call_blocks(model: str) -> None:
    """A single stream carrying a reasoning block, a text block, and a
    function-call block must emit block_start/delta/end for each, in order,
    with the right block_type -- exercised against an exact GPT-6 model id."""
    provider = _provider(model, use_streaming=True)
    fake_coordinator = _FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    final_response = _response(model=model)
    events = [
        _event(
            "response.output_item.added",
            output_index=0,
            item=_item(type="reasoning"),
        ),
        _event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            delta="thinking...",
        ),
        _event(
            "response.output_item.done",
            output_index=0,
            item=_item(type="reasoning"),
        ),
        _event(
            "response.output_item.added",
            output_index=1,
            item=_item(type="message"),
        ),
        _event(
            "response.output_text.delta",
            output_index=1,
            delta="Hello!",
        ),
        _event(
            "response.output_item.done",
            output_index=1,
            item=_item(type="message"),
        ),
        _event(
            "response.output_item.added",
            output_index=2,
            item=_item(type="function_call", name="search"),
        ),
        _event(
            "response.output_item.done",
            output_index=2,
            item=_item(type="function_call", name="search"),
        ),
    ]
    stream = _FakeEventStream(events, final_response)
    provider.client.responses.stream = MagicMock(
        return_value=_MockStreamContext(stream)
    )

    asyncio.run(provider.complete(_request()))

    stream_events = [
        (name, payload)
        for name, payload in fake_coordinator.hooks.events
        if name.startswith("llm:stream_")
    ]
    block_types_by_index = {
        payload["block_index"]: payload["block_type"]
        for name, payload in stream_events
        if name == "llm:stream_block_start"
    }
    assert block_types_by_index == {0: "thinking", 1: "text", 2: "tool_use"}

    deltas = [
        payload for name, payload in stream_events if name == "llm:stream_block_delta"
    ]
    assert any(d["block_type"] == "thinking" and d["text"] == "thinking..." for d in deltas)
    assert any(d["block_type"] == "text" and d["text"] == "Hello!" for d in deltas)

    tool_start = next(
        payload
        for name, payload in stream_events
        if name == "llm:stream_block_start" and payload["block_index"] == 2
    )
    assert tool_start["name"] == "search"


# ---------------------------------------------------------------------------
# Unauthorized / unavailable model errors -- exercised specifically against
# an exact GPT-6 model id, using the same generic error-translation code
# every model shares.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_unauthorized_raises_authentication_error(model: str) -> None:
    provider = _provider(model)
    response = httpx.Response(
        401, request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    native = openai.AuthenticationError(
        "Incorrect API key provided", response=response, body=None
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
        asyncio.run(provider.complete(_request()))

    assert exc_info.value.status_code == 401
    assert exc_info.value.__cause__ is native


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_sol_luna_unavailable_model_raises_not_found_error(model: str) -> None:
    """An unprovisioned/unavailable exact GPT-6 model id (e.g. no access on
    this account yet) surfaces as HTTP 404 -> kernel NotFoundError, same
    generic translation every model gets -- this provider is stateless-only,
    so there is no `previous_response_id` chain to invalidate and retry."""
    provider = _provider(model)
    response = httpx.Response(
        404, request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    native = openai.NotFoundError(
        f"The model `{model}` does not exist or you do not have access to it.",
        response=response,
        body={
            "error": {
                "type": "invalid_request_error",
                "code": "model_not_found",
                "message": f"The model `{model}` does not exist.",
            }
        },
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.NotFoundError) as exc_info:
        asyncio.run(provider.complete(_request()))

    assert exc_info.value.status_code == 404
    assert exc_info.value.__cause__ is native
