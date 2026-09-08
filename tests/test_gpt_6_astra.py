"""Offline regression tests for the exact ``gpt-6-astra`` integration."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message, ToolSpec
from pydantic import ValidationError

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._cost import compute_cost


def _provider(**config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": "gpt-6-astra",
            "max_retries": 0,
            "use_streaming": False,
            **config,
        },
    )


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _response(
    *,
    status: str = "completed",
    service_tier: str | None = "default",
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_astra",
        model="gpt-6-astra",
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


class _StreamContext:
    def __init__(self, stream: object) -> None:
        self._stream = stream

    async def __aenter__(self) -> object:
        return self._stream

    async def __aexit__(self, *args: object) -> None:
        return None


class _TerminalFailedStream:
    def __init__(self, failed_response: SimpleNamespace) -> None:
        self._failed_response = failed_response
        self._sent = False
        self._response = SimpleNamespace(headers={})

    def __aiter__(self) -> _TerminalFailedStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return SimpleNamespace(type="response.failed", response=self._failed_response)

    async def get_final_response(self) -> None:
        raise RuntimeError("Didn't receive a `response.completed` event.")


class _CompletedStream:
    def __init__(self, response: SimpleNamespace) -> None:
        self._response = SimpleNamespace(headers={})
        self._final_response = response
        self._sent = False

    def __aiter__(self) -> _CompletedStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return SimpleNamespace(type="response.completed", response=self._final_response)

    async def get_final_response(self) -> SimpleNamespace:
        return self._final_response


class TestCapabilities:
    def test_exact_astra_capabilities_and_safe_input_budget(self) -> None:
        caps = get_capabilities("gpt-6-astra")

        assert caps.family == "gpt-6-astra"
        assert caps.context_window == 922_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_reasoning is True
        assert caps.default_reasoning_effort is None
        assert caps.supports_vision is True
        assert caps.supports_streaming is True
        assert caps.supports_native_apply_patch is True
        assert caps.supports_native_computer_use is True
        assert caps.long_context_pricing_threshold == 272_000

    def test_only_exact_astra_id_receives_astra_capabilities(self) -> None:
        assert get_capabilities("gpt-6-astra-2099-01-01").family != "gpt-6-astra"

    def test_long_context_reporting_defaults_to_standard_pricing_budget(self) -> None:
        assert _provider().get_info().defaults["context_window"] == 272_000
        assert _provider(enable_long_context=True).get_info().defaults["context_window"] == 922_000

    def test_list_models_uses_friendly_name_and_reported_limits(self) -> None:
        provider = _provider(hide_dated_models=False)
        provider._client = AsyncMock()
        provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(id="gpt-6-astra")])
        )

        models = asyncio.run(provider.list_models())

        assert len(models) == 1
        assert models[0].display_name == "GPT 6 Astra"
        assert models[0].context_window == 272_000
        assert models[0].max_output_tokens == 128_000


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_astra_allows_documented_reasoning_efforts(effort: str) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request(), reasoning={"effort": effort}))

    assert provider.client.responses.create.call_args.kwargs["reasoning"]["effort"] == effort


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning": {"effort": "none"}},
        {"reasoning": {"effort": "minimal"}},
        {"reasoning": {"effort": "unknown"}},
        {"extra_request_params": {"reasoning": {"effort": "none"}}},
        {"extra_request_params": {"temperature": 0}},
        {"extra_request_params": {"top_p": 0}},
        {"extra_request_params": {"top_logprobs": 0}},
        {"extra_request_params": {"logprobs": True}},
        {"extra_request_params": {"include": ["message.output_text.logprobs"]}},
    ],
)
def test_astra_rejects_unsupported_outgoing_fields_before_sdk(kwargs: dict) -> None:
    provider = _provider(**kwargs)
    provider.client.responses.create = AsyncMock(return_value=_response())

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request()))

    provider.client.responses.create.assert_not_awaited()


def test_astra_none_selector_omits_reasoning() -> None:
    provider = _provider(reasoning_effort="none")
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    assert "reasoning" not in provider.client.responses.create.call_args.kwargs


@pytest.mark.parametrize("use_streaming", [False, True])
@pytest.mark.parametrize(
    ("config", "expected_options"),
    [
        ({}, None),
        ({"prompt_cache_options": {"ttl": "30m"}}, {"ttl": "30m"}),
    ],
    ids=["default_only", "ttl_only"],
)
def test_astra_implicit_default_retention_is_neither_sent_nor_warned(
    config: dict[str, object],
    expected_options: dict[str, str] | None,
    use_streaming: bool,
    caplog,
) -> None:
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(use_streaming=use_streaming, **config)
    if use_streaming:
        provider.client.responses.stream = MagicMock(
            return_value=_StreamContext(_CompletedStream(_response()))
        )
    else:
        provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    call = (
        provider.client.responses.stream.call_args
        if use_streaming
        else provider.client.responses.create.call_args
    )
    assert "prompt_cache_retention" not in call.kwargs
    assert call.kwargs.get("prompt_cache_options") == expected_options
    assert not [
        record for record in caplog.records if "prompt_cache_retention" in record.message
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reasoning": {"effort": "none"}},
        {"reasoning": {"effort": "minimal"}},
        {"temperature": 0.1},
        {"include": ["message.output_text.logprobs"]},
    ],
)
def test_astra_rejects_invalid_per_call_fields_before_sdk(kwargs: dict) -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(return_value=_response())

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request(), **kwargs))

    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        ({"prompt_cache_retention": "24h"}, {}),
        ({}, {"prompt_cache_retention": "24h"}),
        ({"extra_request_params": {"prompt_cache_retention": "24h"}}, {}),
    ],
    ids=["config", "per_call", "final_extra"],
)
def test_astra_explicit_legacy_retention_is_omitted_and_warned_once_across_requests(
    config: dict[str, object], kwargs: dict[str, object], caplog
) -> None:
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(
        side_effect=[_response(status="incomplete"), _response(), _response()]
    )

    asyncio.run(provider.complete(_request(), **kwargs))
    asyncio.run(provider.complete(_request(), **kwargs))

    calls = provider.client.responses.create.call_args_list
    assert len(calls) == 3
    assert all("prompt_cache_retention" not in call.kwargs for call in calls)
    warnings = [
        record
        for record in caplog.records
        if "Dropping prompt_cache_retention" in record.message
    ]
    assert len(warnings) == 1
    assert "prompt_cache_options.ttl" in warnings[0].message


@pytest.mark.parametrize(
    ("config", "kwargs"),
    [
        ({"prompt_cache_retention": None}, {}),
        ({"prompt_cache_retention": ""}, {}),
        ({"prompt_cache_retention": "24h"}, {"prompt_cache_retention": None}),
        ({"prompt_cache_retention": "24h"}, {"prompt_cache_retention": ""}),
    ],
    ids=["config_null", "config_empty", "per_call_null", "per_call_empty"],
)
def test_astra_null_or_empty_retention_opts_out(
    config: dict[str, object], kwargs: dict[str, object], caplog
) -> None:
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request(), **kwargs))

    assert "prompt_cache_retention" not in provider.client.responses.create.call_args.kwargs
    assert not [
        record for record in caplog.records if "prompt_cache_retention" in record.message
    ]


@pytest.mark.parametrize(
    ("default_model", "request_model", "expected_retention"),
    [
        ("gpt-5.4", "gpt-6-astra", None),
        ("gpt-6-astra", "gpt-5.4", "24h"),
        ("gpt-6-astra-2099-01-01", None, "24h"),
    ],
    ids=["override_to_astra", "override_from_astra", "nearby_model"],
)
def test_astra_implicit_retention_gating_uses_effective_exact_model(
    default_model: str,
    request_model: str | None,
    expected_retention: str | None,
    caplog,
) -> None:
    caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")
    provider = _provider(default_model=default_model)
    provider.client.responses.create = AsyncMock(return_value=_response())
    kwargs = {"model": request_model} if request_model is not None else {}

    asyncio.run(provider.complete(_request(), **kwargs))

    assert (
        provider.client.responses.create.call_args.kwargs.get("prompt_cache_retention")
        == expected_retention
    )
    assert not [
        record for record in caplog.records if "prompt_cache_retention" in record.message
    ]


def test_astra_drops_legacy_retention_reintroduced_by_final_extra_merge() -> None:
    provider = _provider(extra_request_params={"prompt_cache_retention": "24h"})
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    assert "prompt_cache_retention" not in provider.client.responses.create.call_args.kwargs


def test_astra_accepts_only_documented_cache_ttl() -> None:
    provider = _provider(prompt_cache_options={"ttl": "30m"})
    provider.client.responses.create = AsyncMock(return_value=_response())
    asyncio.run(provider.complete(_request()))
    assert provider.client.responses.create.call_args.kwargs["prompt_cache_options"] == {
        "ttl": "30m"
    }

    invalid = _provider(prompt_cache_options={"ttl": "24h"})
    invalid.client.responses.create = AsyncMock(return_value=_response())
    with pytest.raises(kernel_errors.InvalidRequestError, match="30m"):
        asyncio.run(invalid.complete(_request()))
    invalid.client.responses.create.assert_not_awaited()


def test_astra_null_unsupported_sampling_param_is_omitted() -> None:
    provider = _provider(extra_request_params={"temperature": None})
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    assert "temperature" not in provider.client.responses.create.call_args.kwargs


def test_astra_null_logprobs_is_omitted() -> None:
    provider = _provider(extra_request_params={"logprobs": None})
    provider.client.responses.create = AsyncMock(return_value=_response())

    asyncio.run(provider.complete(_request()))

    assert "logprobs" not in provider.client.responses.create.call_args.kwargs


def test_astra_continuation_validates_final_extra_params_before_second_sdk_call() -> None:
    provider = _provider(extra_request_params={"seed": 1})
    call_count = 0

    async def create(**_: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            provider.extra_request_params = {"top_p": 0}
            return _response(status="incomplete")
        return _response()

    provider.client.responses.create = AsyncMock(side_effect=create)
    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request()))

    assert provider.client.responses.create.await_count == 1


def test_astra_continuation_rejects_logprobs_after_final_extra_merge() -> None:
    provider = _provider(extra_request_params={"seed": 1})
    call_count = 0

    async def create(**_: object) -> SimpleNamespace:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            provider.extra_request_params = {"logprobs": True}
            return _response(status="incomplete")
        return _response()

    provider.client.responses.create = AsyncMock(side_effect=create)
    with pytest.raises(kernel_errors.InvalidRequestError, match="logprobs"):
        asyncio.run(provider.complete(_request()))

    assert provider.client.responses.create.await_count == 1


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("default", Decimal("0.001")),
        ("flex", Decimal("0.0005")),
        ("priority", Decimal("0.002")),
        ("fast", Decimal("0.002")),
    ],
)
def test_astra_pricing_uses_actual_service_tier(tier: str, expected: Decimal) -> None:
    assert compute_cost(
        "gpt-6-astra", prompt_tokens=100, service_tier=tier
    ) == expected


def test_astra_pricing_long_boundary_and_unknown_tier() -> None:
    assert compute_cost("gpt-6-astra", prompt_tokens=272_000) == Decimal("2.72")
    assert compute_cost("gpt-6-astra", prompt_tokens=272_001) == Decimal("5.44002")
    assert (
        compute_cost("gpt-6-astra", prompt_tokens=100, service_tier=None) is None
    )
    assert (
        compute_cost("gpt-6-astra", prompt_tokens=100, service_tier="scale") is None
    )
    assert compute_cost("gpt-6-astra-2099-01-01", prompt_tokens=100) is None


def test_astra_fast_request_costs_response_default_when_downgraded() -> None:
    provider = _provider(extra_request_params={"service_tier": "fast"})
    provider.client.responses.create = AsyncMock(return_value=_response(service_tier="default"))

    response = asyncio.run(provider.complete(_request()))

    assert response.usage.cost_usd == Decimal("0.0015")


def test_astra_computer_raw_fallback_emits_raw_response_with_coordinator() -> None:
    """The raw computer fallback must support ``raw: true`` response emission."""
    provider = _provider(raw=True)
    coordinator = SimpleNamespace(
        hooks=SimpleNamespace(emit=AsyncMock()),
        get_capability=lambda _: None,
    )
    provider.coordinator = coordinator
    raw_body = {
        "id": "resp_astra_computer",
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [
            {
                "type": "computer_call",
                "call_id": "call_astra_screenshot",
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
    client.responses.with_raw_response.create = AsyncMock(return_value=_RawSDKResponse())
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
                model="gpt-6-astra",
                messages=[Message(role="user", content="take a screenshot")],
                tools=[computer],
            )
        )
    )

    assert result.tool_calls[0].arguments == {"actions": [{"type": "screenshot"}]}
    response_events = [
        call.args[1]
        for call in coordinator.hooks.emit.await_args_list
        if call.args[0] == "llm:response"
    ]
    assert response_events[0]["raw"] == raw_body


def test_astra_stream_failed_response_usage_is_billed_once_after_retry() -> None:
    """A failed SSE terminal is billed once even when the SDK then retries."""
    provider = _provider(
        use_streaming=True,
        max_retries=1,
        min_retry_delay=0,
        max_retry_delay=0,
        retry_jitter=False,
    )
    failed = _response(
        status="failed",
        service_tier="priority",
        input_tokens=272_001,
        output_tokens=1,
    )
    succeeded = _response(
        service_tier="flex",
        input_tokens=272_000,
        output_tokens=1,
    )
    provider.client.responses.stream = MagicMock(
        side_effect=[
            _StreamContext(_TerminalFailedStream(failed)),
            _StreamContext(_CompletedStream(succeeded)),
        ]
    )

    result = asyncio.run(provider.complete(_request()))

    assert provider.client.responses.stream.call_count == 2
    assert result.usage.input_tokens == 544_001
    assert result.usage.output_tokens == 2
    assert result.usage.cost_usd == Decimal("12.240215")


def test_astra_stream_failed_response_without_priceable_usage_leaves_cost_unknown() -> None:
    provider = _provider(
        use_streaming=True,
        max_retries=1,
        min_retry_delay=0,
        max_retry_delay=0,
        retry_jitter=False,
    )
    failed = _response(status="failed", service_tier=None)
    succeeded = _response(service_tier="default")
    provider.client.responses.stream = MagicMock(
        side_effect=[
            _StreamContext(_TerminalFailedStream(failed)),
            _StreamContext(_CompletedStream(succeeded)),
        ]
    )

    result = asyncio.run(provider.complete(_request()))

    assert result.usage.cost_usd is None


def test_astra_stream_failed_response_without_usage_leaves_cost_unknown() -> None:
    provider = _provider(
        use_streaming=True,
        max_retries=1,
        min_retry_delay=0,
        max_retry_delay=0,
        retry_jitter=False,
    )
    failed = _response(status="failed", service_tier="default")
    del failed.usage
    succeeded = _response(service_tier="default")
    provider.client.responses.stream = MagicMock(
        side_effect=[
            _StreamContext(_TerminalFailedStream(failed)),
            _StreamContext(_CompletedStream(succeeded)),
        ]
    )

    result = asyncio.run(provider.complete(_request()))

    assert result.usage.cost_usd is None


def test_astra_continuation_sums_each_actual_response_tier() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(
        side_effect=[
            _response(status="incomplete", service_tier="flex"),
            _response(service_tier="priority"),
        ]
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.usage.cost_usd == Decimal("0.00375")


def test_astra_multiple_continuations_bill_every_response() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(
        side_effect=[
            _response(status="incomplete", service_tier="default"),
            _response(status="incomplete", service_tier="flex"),
            _response(service_tier="priority"),
        ]
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.usage.cost_usd == Decimal("0.00525")
    assert response.usage.input_tokens == 300
    assert response.usage.output_tokens == 30


def test_astra_cached_write_mix_and_reasoning_tokens_are_not_double_charged() -> None:
    assert compute_cost(
        "gpt-6-astra",
        prompt_tokens=100_000,
        cached_tokens=10_000,
        cache_write_tokens=20_000,
        completion_tokens=20_000,
        service_tier="default",
    ) == Decimal("1.96")


def test_astra_unknown_continuation_tier_does_not_report_partial_total() -> None:
    provider = _provider()
    provider.client.responses.create = AsyncMock(
        side_effect=[
            _response(status="incomplete", service_tier="default"),
            _response(service_tier=None),
        ]
    )

    response = asyncio.run(provider.complete(_request()))

    assert response.usage.cost_usd is None