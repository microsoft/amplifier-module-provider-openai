"""Focused, deterministic coverage for OpenAI request budget preflight.

These tests use only assembled payloads and fake responses.  They deliberately do
not contact an API: parent DTU validation owns execution of this file.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


def _provider(**config):
    return OpenAIProvider(
        api_key="test-key", config={"use_streaming": False, "max_retries": 0, **config}
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
    params = provider._budget_params(request)
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


def test_low_level_direct_dispatch_uses_the_local_guard():
    provider = _provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request())
    calls = []

    def reject(assembled):
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

    def guard_then_reject_continuation(_params):
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
