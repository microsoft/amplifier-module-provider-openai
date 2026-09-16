"""Focused, deterministic coverage for OpenAI request budget preflight.

These tests use only assembled payloads and fake responses.  They deliberately do
not contact an API: parent DTU validation owns execution of this file.
"""

import asyncio
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider, _tool_search


def _provider(**config):
    return OpenAIProvider(
        api_key="test-key",
        config={
            "use_streaming": False,
            "max_retries": 0,
            "base_url": "https://test.openai.invalid/v1",
            **config,
        },
    )


def _request(content="hello", *, system=None, output=None):
    messages = []
    if system is not None:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=content))
    kwargs = {"messages": messages}
    if output is not None:
        kwargs["max_output_tokens"] = output
    return ChatRequest(**kwargs)


def _response(input_tokens, *, status="completed"):
    return SimpleNamespace(status=status, usage=SimpleNamespace(input_tokens=input_tokens))


class _StreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_response(self):
        return self.response


class _RetryableFailingStreamContext:
    """Stream request that fails before returning any generation response."""

    async def __aenter__(self):
        raise kernel_errors.LLMError(
            "retryable stream failure", provider="openai", retryable=True
        )

    async def __aexit__(self, *args):
        return False


def _native_provider(*, input_tokens=10, **config):
    counter = AsyncMock(return_value=SimpleNamespace(input_tokens=input_tokens))
    responses = SimpleNamespace(
        input_tokens=SimpleNamespace(count=counter),
        create=AsyncMock(),
        stream=MagicMock(),
    )
    provider = OpenAIProvider(
        api_key="test-key",
        client=SimpleNamespace(
            base_url="https://api.openai.com/v1/",
            responses=responses,
        ),
        config={"use_streaming": False, "max_retries": 0, **config},
    )
    return provider, counter


def _count_projection(params):
    return {
        key: params[key]
        for key in OpenAIProvider._COUNT_FIELDS
        if key in params
    }


def test_native_count_returns_official_measurement_for_the_finalized_projection():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    request = ChatRequest(
        messages=[
            Message(role="system", content="be concise"),
            Message(role="user", content="hello"),
        ],
        tools=[
            ToolSpec(
                name="weather",
                description="get weather",
                parameters={"type": "object", "properties": {}},
            )
        ],
        max_output_tokens=1_000,
    )

    decision = asyncio.run(
        provider.request_budget(
            request,
            context_estimate=123,
            request_options={"temperature": 0.1, "reasoning_effort": "low"},
            temperature=0.2,
        )
    )
    expected = provider._budget_params(
        request, temperature=0.2, reasoning_effort="low"
    )

    assert decision["estimated_input_tokens"] == 10
    assert decision["context_token_budget"] == 123
    assert decision["measurement"] == {
        "kind": "provider_count",
        "source": "official.operation",
        "input_tokens": 10,
    }
    assert "request_budget:provider_count" in provider.get_info().capabilities
    assert counter.call_args.kwargs == _count_projection(expected)
    assert "max_output_tokens" not in counter.call_args.kwargs
    assert "store" not in counter.call_args.kwargs
    assert "request_options" not in counter.call_args.kwargs


@pytest.mark.parametrize("use_streaming", [False, True])
def test_native_count_and_generation_share_final_assembly_for_both_dispatch_paths(
    use_streaming,
):
    provider, counter = _native_provider(
        default_model="gpt-5-mini", use_streaming=use_streaming
    )
    request = _request("hello", system="be concise", output=1_000)
    response = _response(10)
    response.output = []
    expected = provider._budget_params(
        request,
        temperature=0.2,
        reasoning_effort="low",
        tools=[{"type": "web_search_preview"}],
    )
    if use_streaming:
        provider.client.responses.stream.return_value = _StreamContext(response)
    else:
        provider.client.responses.create.return_value = response

    asyncio.run(
        provider.complete(
            request,
            request_options={"temperature": 0.1, "reasoning_effort": "low"},
            temperature=0.2,
            tools=[{"type": "web_search_preview"}],
        )
    )

    sdk_call = (
        provider.client.responses.stream
        if use_streaming
        else provider.client.responses.create
    )
    assert counter.call_args.kwargs == _count_projection(expected)
    assert sdk_call.call_args.kwargs == expected
    assert "request_options" not in sdk_call.call_args.kwargs


def test_native_count_overrides_a_stale_calibrated_estimate_before_dispatch():
    provider, counter = _native_provider(default_model="gpt-5-mini", input_tokens=10)
    provider._budget_calibration["gpt-5-mini"] = (100.0, 1, 100)
    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response

    asyncio.run(provider.complete(_request("native count wins")))

    assert counter.await_count == 1
    assert provider.client.responses.create.await_count == 1


def test_native_preflight_is_pure_and_final_dispatch_uses_a_fresh_count():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    request = _request("native count")
    state_before = (
        provider._client,
        provider._pending_additional_tools_item,
        provider._tool_search_roster,
        dict(provider._tool_search_extra),
        set(provider._native_call_ids),
        dict(provider._native_call_types),
        dict(provider._budget_calibration),
    )

    decision = asyncio.run(provider.request_budget(request, context_estimate=10))

    assert decision["measurement"]["input_tokens"] == 10
    assert (
        provider._client,
        provider._pending_additional_tools_item,
        provider._tool_search_roster,
        provider._tool_search_extra,
        provider._native_call_ids,
        provider._native_call_types,
        provider._budget_calibration,
    ) == state_before

    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    asyncio.run(provider.complete(request))

    # One count makes the Context-facing decision and a separate fresh count
    # protects the exact payload immediately before the create dispatch.
    assert counter.await_count == 2
    assert provider.client.responses.create.await_count == 1


def test_native_count_over_allowance_blocks_generation_dispatch():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request())
    counter.return_value = SimpleNamespace(
        input_tokens=provider._budget_input_limit(params) + 1
    )

    with pytest.raises(kernel_errors.ContextLengthError, match="native input allowance"):
        asyncio.run(provider.complete(_request()))

    assert counter.await_count == 1
    provider.client.responses.create.assert_not_called()


@pytest.mark.parametrize(
    ("reported", "available"),
    [(0, True), (None, False), (-1, False), (True, False), ("10", False)],
)
def test_native_count_validates_counter_response(reported, available):
    provider, _ = _native_provider(default_model="gpt-5-mini", input_tokens=reported)

    decision = asyncio.run(provider.request_budget(_request(), context_estimate=10))

    if available:
        assert decision["measurement"]["input_tokens"] == reported
    else:
        assert decision is None


def test_native_count_failure_and_cancellation_do_not_become_measurements():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    counter.side_effect = RuntimeError("count response malformed")
    assert asyncio.run(provider.request_budget(_request(), context_estimate=10)) is None
    assert "request_budget:provider_count" in provider.get_info().capabilities

    counter.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(provider.request_budget(_request(), context_estimate=10))


def test_native_count_is_unavailable_for_unknown_fields_custom_routes_and_subclasses():
    provider, counter = _native_provider(
        default_model="gpt-5-mini",
        extra_request_params={"unproved_input_option": "value"},
    )
    assert asyncio.run(provider.request_budget(_request(), context_estimate=10)) is None
    counter.assert_not_called()
    assert "request_budget:provider_count" in provider.get_info().capabilities

    custom = _provider(default_model="gpt-5-mini")
    assert "request_budget:provider_count" not in custom.get_info().capabilities
    assert isinstance(custom.request_budget(_request(), context_estimate=10), dict)

    class DerivedOpenAIProvider(OpenAIProvider):
        pass

    subclass = DerivedOpenAIProvider(
        api_key="test-key",
        config={"base_url": "https://api.openai.com/v1"},
    )
    assert "request_budget:provider_count" not in subclass.get_info().capabilities


def test_injected_standard_client_endpoint_wins_over_custom_environment(monkeypatch):
    provider, _ = _native_provider(default_model="gpt-5-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.invalid/v1")

    assert "request_budget:provider_count" in provider.get_info().capabilities


@pytest.mark.parametrize("injected_base_url", ["https://proxy.invalid/v1", None])
def test_injected_client_endpoint_overrides_standard_config_and_environment(
    monkeypatch, injected_base_url
):
    provider, counter = _native_provider(
        default_model="gpt-5-mini",
        base_url="https://api.openai.com/v1",
    )
    provider.client.base_url = injected_base_url
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    assert "request_budget:provider_count" not in provider.get_info().capabilities
    budget = provider.request_budget(_request(), context_estimate=10)
    assert isinstance(budget, dict)
    assert "measurement" not in budget
    counter.assert_not_called()

    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    asyncio.run(provider.complete(_request()))

    assert provider.client.responses.create.await_count == 1
    counter.assert_not_called()


@pytest.mark.parametrize("use_streaming", [False, True])
def test_native_final_count_refresh_blocks_retry_before_second_generation_dispatch(
    use_streaming,
):
    provider, counter = _native_provider(
        default_model="gpt-5-mini",
        use_streaming=use_streaming,
        max_retries=1,
        retry_jitter=False,
    )
    request = _request("retry count")
    allowance = provider._budget_input_limit(provider._budget_params(request))
    counter.side_effect = [
        SimpleNamespace(input_tokens=10),
        SimpleNamespace(input_tokens=allowance + 1),
    ]
    retryable_failure = kernel_errors.LLMError(
        "retryable generation failure", provider="openai", retryable=True
    )
    if use_streaming:
        provider.client.responses.stream = MagicMock(
            return_value=_RetryableFailingStreamContext()
        )
    else:
        provider.client.responses.create = AsyncMock(side_effect=retryable_failure)

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(kernel_errors.ContextLengthError, match="native input allowance"),
    ):
        asyncio.run(provider.complete(request))

    # One pre-event count covers the first attempt. The retry obtains a fresh
    # count and is rejected before it can make a second physical dispatch.
    assert counter.await_count == 2
    if use_streaming:
        assert provider.client.responses.stream.call_count == 1
    else:
        assert provider.client.responses.create.await_count == 1


def test_missing_native_helper_keeps_legacy_budget_behavior():
    provider = OpenAIProvider(
        api_key="test-key",
        client=SimpleNamespace(
            base_url="https://api.openai.com/v1/",
            responses=SimpleNamespace(input_tokens=SimpleNamespace()),
        ),
        config={"use_streaming": False, "max_retries": 0},
    )

    decision = provider.request_budget(_request(), context_estimate=10)

    assert isinstance(decision, dict)
    assert "measurement" not in decision
    assert "request_budget:provider_count" not in provider.get_info().capabilities


@pytest.mark.parametrize(
    ("reported_version", "available"),
    [
        ("2.5.9", False),
        ("2.6.0", True),
        ("3.5.0", True),
        ("3.5.0rc1", False),
        ("malformed", False),
        ("3.5", False),
    ],
)
def test_uninitialized_standard_route_requires_a_supported_stable_sdk_version(
    monkeypatch, reported_version, available
):
    provider = OpenAIProvider(
        api_key="test-key",
        config={"base_url": "https://api.openai.com/v1"},
    )
    monkeypatch.setattr(
        "amplifier_module_provider_openai.installed_package_version",
        lambda _distribution: reported_version,
    )

    assert (
        "request_budget:provider_count" in provider.get_info().capabilities
    ) is available


def test_uninitialized_standard_route_fails_closed_when_sdk_metadata_is_missing(
    monkeypatch,
):
    provider = OpenAIProvider(
        api_key="test-key",
        config={"base_url": "https://api.openai.com/v1"},
    )

    def missing_sdk(_distribution):
        raise PackageNotFoundError("openai")

    monkeypatch.setattr(
        "amplifier_module_provider_openai.installed_package_version", missing_sdk
    )

    assert "request_budget:provider_count" not in provider.get_info().capabilities


def test_injected_callable_counter_remains_authoritative_over_sdk_version(monkeypatch):
    provider, _ = _native_provider(default_model="gpt-5-mini")
    monkeypatch.setattr(
        "amplifier_module_provider_openai.installed_package_version",
        lambda _distribution: "2.5.9",
    )

    assert "request_budget:provider_count" in provider.get_info().capabilities


def test_custom_route_does_not_advertise_even_with_supported_sdk_version(monkeypatch):
    provider = _provider(default_model="gpt-5-mini")
    monkeypatch.setattr(
        "amplifier_module_provider_openai.installed_package_version",
        lambda _distribution: "3.5.0",
    )

    assert "request_budget:provider_count" not in provider.get_info().capabilities


def test_preflight_uses_complete_assembled_payload_and_full_output_reserve():
    provider = _provider(default_model="gpt-5-mini")
    request = _request("hello", system="be concise", output=1_000)

    params = provider._budget_params(request, temperature=0.2)
    budget = provider.request_budget(request, context_estimate=123, temperature=0.2)

    assert params["instructions"] == "be concise"
    assert params["temperature"] == 0.2
    assert budget["estimated_input_tokens"] == provider._serialized_input_bytes(params)
    assert budget["input_limit_tokens"] == 128_000 - 1_000 - 4_096
    assert budget["context_token_budget"] == 123
    assert budget["max_output_tokens"] == 1_000


def test_cold_oversize_preflight_returns_none_without_state_or_log_side_effects(caplog):
    provider = _provider(default_model="gpt-5-mini")
    request = _request("cold-preflight-private-content-" * 5_000, output=1_000)
    state_before = (
        dict(provider._budget_calibration),
        set(provider._budget_uncalibrated_warned_models),
    )

    caplog.clear()
    assert provider.request_budget(request, context_estimate=100_000) is None

    assert (
        provider._budget_calibration,
        provider._budget_uncalibrated_warned_models,
    ) == state_before
    assert not caplog.records


def test_request_output_cap_overrides_extra_request_params() -> None:
    provider = _provider(
        default_model="gpt-5-mini",
        extra_request_params={"max_output_tokens": 64_000},
    )
    request = _request("hello", output=1_000)

    assert provider._budget_params(request)["max_output_tokens"] == 1_000


def test_dispatch_keeps_request_output_cap_over_extra_request_params() -> None:
    provider = _provider(
        default_model="gpt-5-mini",
        extra_request_params={"max_output_tokens": 64_000},
    )
    response = _response(10)
    response.output = []
    provider.client.responses.create = AsyncMock(return_value=response)

    asyncio.run(provider.complete(_request("hello", output=1_000)))

    assert provider.client.responses.create.call_args.kwargs["max_output_tokens"] == 1_000


def test_preflight_assembly_matches_nonstream_sdk_payload():
    provider = _provider(default_model="gpt-5-mini")
    request = _request("hello", system="be concise")
    expected = provider._budget_params(
        request, temperature=0.2, tools=[{"type": "web_search_preview"}]
    )
    response = _response(10)
    response.output = []
    provider.client.responses.create = AsyncMock(return_value=response)

    asyncio.run(
        provider.complete(
            request, temperature=0.2, tools=[{"type": "web_search_preview"}]
        )
    )

    assert provider.client.responses.create.call_args.kwargs == expected


def test_negative_output_reserve_cannot_enlarge_the_input_allowance():
    provider = _provider(default_model="gpt-5.6-terra", enable_long_context=True)
    provider.client.responses.create = AsyncMock()
    request = _request("x" * 1_000_000)

    with pytest.raises(kernel_errors.ContextLengthError, match="nonnegative integer"):
        asyncio.run(provider.complete(request, max_tokens=-1_000_000))

    provider.client.responses.create.assert_not_called()


def test_omitted_direct_output_cap_reserves_model_maximum_without_mutation():
    provider = _provider(default_model="gpt-5.6-terra", enable_long_context=True)
    params = {"model": "gpt-5.6-terra", "input": []}
    assert provider._budget_input_limit(params) == 900_000 - 128_000 - 4_096
    assert "max_output_tokens" not in params


def test_preflight_assembly_matches_streaming_sdk_payload_and_final_usage():
    provider = _provider(default_model="gpt-5-mini", use_streaming=True)
    request = _request("hello")
    expected = provider._budget_params(request)
    response = _response(10)
    response.output = []
    provider.client.responses.stream = MagicMock(return_value=_StreamContext(response))

    asyncio.run(provider.complete(request))

    assert provider.client.responses.stream.call_args.kwargs == expected
    assert "gpt-5-mini" in provider._budget_calibration


def test_preflight_has_no_request_mutation_events_or_calibration_side_effects():
    provider = _provider(default_model="gpt-5-mini")
    request = _request("hello", system="system")
    before = request.model_dump()
    state_before = (
        provider._pending_additional_tools_item,
        provider._tool_search_roster,
        dict(provider._tool_search_extra),
        set(provider._native_call_ids),
        dict(provider._native_call_types),
        dict(provider._budget_calibration),
    )

    provider.request_budget(request, context_estimate=10)

    assert request.model_dump() == before
    assert (
        provider._pending_additional_tools_item,
        provider._tool_search_roster,
        provider._tool_search_extra,
        provider._native_call_ids,
        provider._native_call_types,
        provider._budget_calibration,
    ) == state_before


def test_namespace_preflight_defers_warning_until_accepted_dispatch(monkeypatch, caplog):
    monkeypatch.setattr(_tool_search, "_WARNED_UNLISTED", set())
    provider = _provider(
        default_model="gpt-5.6-terra",
        enable_long_context=True,
        tool_search={"mode": "namespaced"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        tools=[
            ToolSpec(
                name="budget_unlisted_probe",
                parameters={"type": "object", "properties": {}},
            )
        ],
    )
    caplog.clear()
    provider.request_budget(request, context_estimate=10)
    provider.request_budget(request, context_estimate=10)
    assert _tool_search._WARNED_UNLISTED == set()
    assert not any("namespace table" in record.message for record in caplog.records)

    response = _response(10)
    response.output = []
    provider.client.responses.create = AsyncMock(return_value=response)
    asyncio.run(provider.complete(request))
    assert ("budget_unlisted_probe",) in _tool_search._WARNED_UNLISTED
    assert sum("namespace table" in record.message for record in caplog.records) == 1


def test_matched_raw_usage_calibration_ignores_cache_fields_and_is_model_scoped():
    provider = _provider(default_model="gpt-5-mini")
    request = _request("small")
    params = provider._budget_params(request)
    response = _response(200)
    response.usage.input_tokens_details = SimpleNamespace(
        cached_tokens=190, cache_write_tokens=190
    )

    provider._record_budget_calibration(params, response)

    assert provider._budget_calibration["gpt-5-mini"][2] == 200
    other = provider.request_budget(request, context_estimate=10, model="gpt-5.6-terra")
    other_params = provider._budget_params(request, model="gpt-5.6-terra")
    assert other["estimated_input_tokens"] == provider._serialized_input_bytes(other_params)


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, 1.5, float("inf")])
def test_bad_or_missing_usage_never_calibrates(bad):
    provider = _provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request())
    provider._record_budget_calibration(params, _response(bad))
    assert provider._budget_calibration == {}


def test_incomplete_or_failed_response_never_calibrates():
    provider = _provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request())

    provider._record_budget_calibration(params, _response(1, status="incomplete"))
    provider._record_budget_calibration(params, _response(1, status="failed"))

    assert provider._budget_calibration == {}


def test_calibration_grows_conservatively_but_smaller_payload_uses_rate():
    provider = _provider(default_model="gpt-5-mini")
    small = _request("small")
    small_params = provider._budget_params(small)
    small_bytes = provider._serialized_input_bytes(small_params)
    provider._record_budget_calibration(small_params, _response(small_bytes * 2))

    large = _request("x" * 20_000)
    large_params = provider._budget_params(large)
    large_bytes = provider._serialized_input_bytes(large_params)
    grown = provider.request_budget(large, context_estimate=10_000)
    reduced = provider.request_budget(small, context_estimate=10)

    assert grown["estimated_input_tokens"] >= small_bytes * 2 + (large_bytes - small_bytes)
    assert reduced["estimated_input_tokens"] < grown["estimated_input_tokens"]


def test_oversize_returns_a_strictly_smaller_positive_context_target():
    provider = _provider(default_model="gpt-5-mini")
    request = _request("x" * 8_000)
    provider._budget_calibration["gpt-5-mini"] = (100.0, 1, 100)

    budget = provider.request_budget(request, context_estimate=100_000)

    assert budget["estimated_input_tokens"] > budget["input_limit_tokens"]
    assert 0 < budget["context_token_budget"] < 100_000


def test_final_guard_uses_same_growth_bound_and_has_scalar_attribution():
    provider = _provider(default_model="gpt-5-mini")
    base = provider._budget_params(_request("small"))
    base_bytes = provider._serialized_input_bytes(base)
    provider._budget_calibration["gpt-5-mini"] = (0.25, base_bytes, 50_000)
    params = provider._budget_params(_request("x" * 20_000))

    with pytest.raises(kernel_errors.ContextLengthError, match="input_items=.*tools="):
        provider._guard_assembled_params(params)


@pytest.mark.parametrize("use_streaming", [False, True])
def test_cold_oversize_dispatches_once_warns_once_and_calibrates(use_streaming, caplog):
    private_content = "cold-dispatch-private-content-" * 5_000
    provider = _provider(
        default_model="gpt-5-mini",
        use_streaming=use_streaming,
        extra_request_params={"max_output_tokens": 64_000},
    )
    request = _request(private_content, output=1_000)
    params = provider._budget_params(request)
    response = _response(100_000)
    response.output = []
    if use_streaming:
        provider.client.responses.stream = MagicMock(
            return_value=_StreamContext(response)
        )
    else:
        provider.client.responses.create = AsyncMock(return_value=response)

    caplog.clear()
    asyncio.run(provider.complete(request))

    sdk_call = (
        provider.client.responses.stream
        if use_streaming
        else provider.client.responses.create
    )
    assert sdk_call.call_count == 1
    assert sdk_call.call_args.kwargs["max_output_tokens"] == 1_000
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Local budget estimate is uncalibrated" in record.getMessage()
    ]
    expected_warning = (
        "[PROVIDER] Local budget estimate is uncalibrated; sending request for API "
        "validation and API may reject it "
        f"(model=gpt-5-mini, serialized_bytes={provider._serialized_input_bytes(params)}, "
        f"input_limit_tokens={provider._budget_input_limit(params)})."
    )
    assert len(warnings) == 1
    assert warnings == [expected_warning]
    assert private_content not in warnings[0]
    assert "gpt-5-mini" in provider._budget_calibration


def test_cold_oversize_repeat_without_usage_warns_once_per_model(caplog):
    provider = _provider(default_model="gpt-5-mini")
    response = _response(0)
    response.output = []
    provider.client.responses.create = AsyncMock(return_value=response)
    request = _request("cold-repeat-private-content-" * 5_000, output=1_000)

    caplog.clear()
    asyncio.run(provider.complete(request))
    asyncio.run(provider.complete(request))

    assert provider.client.responses.create.await_count == 2
    assert provider._budget_calibration == {}
    assert (
        sum(
            "Local budget estimate is uncalibrated" in record.getMessage()
            for record in caplog.records
        )
        == 1
    )


def test_cold_oversize_warning_is_isolated_by_model(caplog):
    provider = _provider(default_model="gpt-5-mini")
    response = _response(0)
    response.output = []
    provider.client.responses.create = AsyncMock(return_value=response)
    request = _request("cold-model-private-content-" * 5_000, output=1_000)

    caplog.clear()
    asyncio.run(provider.complete(request))
    asyncio.run(provider.complete(request, model="gpt-5-mini-2025-08-07"))

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Local budget estimate is uncalibrated" in record.getMessage()
    ]
    assert provider.client.responses.create.await_count == 2
    assert len(warnings) == 2
    assert any("model=gpt-5-mini" in message for message in warnings)
    assert any("model=gpt-5-mini-2025-08-07" in message for message in warnings)


def test_calibrated_oversize_rejects_before_sdk_dispatch():
    provider = _provider(default_model="gpt-5-mini")
    provider._budget_calibration["gpt-5-mini"] = (100.0, 1, 100)
    provider.client.responses.create = AsyncMock()

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(
            provider.complete(_request("calibrated-overflow-" * 5_000, output=1_000))
        )

    provider.client.responses.create.assert_not_called()


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (
            {"extra_request_params": {"metadata": {"nonserializable": object()}}},
            kernel_errors.InvalidRequestError,
        ),
        (
            {"extra_request_params": {"max_output_tokens": "not-an-integer"}},
            kernel_errors.ContextLengthError,
        ),
        ({"extra_request_params": {"model": ""}}, kernel_errors.ContextLengthError),
    ],
)
def test_preflight_rejects_invalid_serialization_output_or_model_locally(config, error):
    provider = _provider(default_model="gpt-5-mini", **config)

    with pytest.raises(error):
        provider.request_budget(_request(), context_estimate=1)

    assert provider._budget_uncalibrated_warned_models == set()


def test_low_level_direct_dispatch_uses_the_local_guard():
    provider = _provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request())
    calls = []

    def reject(assembled, **_kwargs):
        calls.append(assembled)
        raise kernel_errors.ContextLengthError("local", provider="openai")

    provider._guard_assembled_params = reject

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider._create_response(params))

    # _create_response is the common direct/native/continuation dispatch seam.
    assert calls == [params]


def test_continuation_rechecks_and_calibrates_its_exact_final_params():
    provider = _provider(default_model="gpt-5-mini")
    request = _request("hello", system="system")
    incomplete = _response(10, status="incomplete")
    incomplete.id = "resp_incomplete"
    incomplete.output = []
    completed = _response(20)
    completed.output = []
    provider.client.responses.create = AsyncMock(side_effect=[incomplete, completed])

    asyncio.run(provider.complete(request))

    assert provider.client.responses.create.await_count == 2
    continuation_params = provider.client.responses.create.call_args_list[1].kwargs
    assert continuation_params["input"] == provider._build_continuation_input(
        provider.client.responses.create.call_args_list[0].kwargs["input"], []
    )
    assert provider._budget_calibration["gpt-5-mini"][1] == provider._serialized_input_bytes(
        continuation_params
    )


def test_over_budget_continuation_raises_without_a_second_sdk_dispatch():
    provider = _provider(default_model="gpt-5-mini")
    incomplete = _response(10, status="incomplete")
    incomplete.id = "resp_incomplete"
    incomplete.output = []
    provider.client.responses.create = AsyncMock(return_value=incomplete)
    guard_calls = 0

    def guard_then_reject_continuation(_params, **_kwargs):
        nonlocal guard_calls
        guard_calls += 1
        # Initial payload is checked before llm:request and again at its direct
        # SDK seam. The continuation's first check must fail locally.
        if guard_calls == 3:
            raise kernel_errors.ContextLengthError("local", provider="openai")

    provider._guard_assembled_params = guard_then_reject_continuation

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_request("hello")))

    assert provider.client.responses.create.await_count == 1
