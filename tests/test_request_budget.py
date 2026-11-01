"""Focused, deterministic coverage for OpenAI request budget preflight.

These tests use only assembled payloads and fake responses.  They deliberately do
not contact an API: parent DTU validation owns execution of this file.
"""

import asyncio
import base64
import copy
import json
import re
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider, _tool_search
from amplifier_module_provider_openai._token_count import TokenCountError


def _wire_params(sdk_kwargs):
    # The SDK consumes timeout locally; it is not a JSON body field and must
    # not change input-token measurements or the exact assembled wire payload.
    result = dict(sdk_kwargs)
    timeout = result.pop("timeout")
    assert timeout.read is None and timeout.write is None
    return result


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


def _json_bytes(value):
    """Independent JSON-byte oracle, never a token estimate."""
    return len(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    )


def _assert_hex(value, length):
    assert type(value) is str
    assert re.fullmatch(rf"[0-9a-f]{{{length}}}", value)


def _assert_descriptor(descriptor, params):
    # Exact keys also prevent exporting content, credentials, or an HMAC key.
    assert type(descriptor) is dict
    assert set(descriptor) == {
        "payload_tag", "serialized_bytes", "input_items", "input_types",
        "input_bytes", "instructions_bytes", "tools", "tools_bytes",
    }
    _assert_hex(descriptor["payload_tag"], 64)
    items = params.get("input", [])
    tools = params.get("tools", [])
    expected_sizes = {
        "serialized_bytes": _json_bytes(params),
        "input_items": len(items) if isinstance(items, list) else 1,
        "input_bytes": _json_bytes(items),
        "instructions_bytes": _json_bytes(params.get("instructions")),
        "tools": len(tools) if isinstance(tools, list) else 0,
        "tools_bytes": _json_bytes(tools),
    }
    for key, expected in expected_sizes.items():
        assert type(descriptor[key]) is int
        assert descriptor[key] == expected
    # Version 1 uses a bounded enum, not caller-controlled type names.
    known_types = {
        "message", "function_call", "function_call_output", "compaction",
        "reasoning", "computer_call", "computer_call_output",
        "apply_patch_call", "apply_patch_call_output", "item_reference",
    }
    expected_types = {}
    for item in items if isinstance(items, list) else []:
        kind = item.get("type", "message") if isinstance(item, dict) else "other"
        kind = kind if isinstance(kind, str) and kind in known_types else "other"
        expected_types[kind] = expected_types.get(kind, 0) + 1
    assert type(descriptor["input_types"]) is dict
    assert descriptor["input_types"] == expected_types
    assert all(type(count) is int for count in descriptor["input_types"].values())


def _assert_attribution(attribution, params):
    assert type(attribution) is dict
    assert set(attribution) == {
        "version", "scope_id", "count_id", "request", "compact_input",
    }
    assert type(attribution["version"]) is int
    assert attribution["version"] == 1
    _assert_hex(attribution["scope_id"], 32)
    _assert_hex(attribution["count_id"], 32)
    _assert_descriptor(attribution["request"], params)
    _assert_descriptor(
        attribution["compact_input"],
        {key: params[key] for key in ("model", "input", "instructions") if key in params},
    )


def _capture_events(provider, callback=None):
    events = []

    async def emit(name, payload):
        # Freeze the observation before any intentionally hostile callback.
        events.append((name, copy.deepcopy(payload)))
        if callback is not None:
            await callback(name, payload)

    provider.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=AsyncMock(side_effect=emit)))
    return events


def _attribution_events(events):
    return [payload for name, payload in events if name == "provider:request_attribution"]


def _assert_dispatch_event(event, *, operation, stage, input_tokens, binding):
    assert type(event) is dict
    assert set(event) == {
        "provider", "operation", "stage", "dispatch_id", "input_tokens",
        "attribution", "dispatch", "binding",
        "implementation", "measurement_scope", "dispatch_scope",
    }
    assert event["provider"] == "openai"
    assert event["operation"] == operation
    assert event["stage"] == stage
    _assert_hex(event["dispatch_id"], 32)
    assert event["input_tokens"] == input_tokens
    if input_tokens is not None:
        assert type(event["input_tokens"]) is int
    assert event["binding"] == binding
    assert event["measurement_scope"] == "count_projection"
    assert event["dispatch_scope"] == "count_projection"
    assert event["implementation"]["imported_bytes_verified"] is False


@pytest.mark.parametrize("estimate", [True, False, 1.5, "1", None, -1])
def test_invalid_context_estimate_keeps_value_error_without_dispatch(estimate):
    provider, counter = _native_provider()
    request = _request()
    original = request.model_dump()

    with pytest.raises(ValueError, match="context_estimate must be a nonnegative integer"):
        provider.request_budget(request, context_estimate=estimate)

    counter.assert_not_called()
    provider.client.responses.create.assert_not_called()
    assert request.model_dump() == original


def test_gpt_5_mini_native_budget_uses_published_limits():
    """A native count above the old 59,904 allowance still fits mini."""
    provider, counter = _native_provider(input_tokens=100_000, default_model="gpt-5-mini")
    decision = asyncio.run(provider.request_budget(_request(), context_estimate=80_000))

    assert decision["max_output_tokens"] == 128_000
    assert decision["input_limit_tokens"] == 267_904  # 400k - 128k - 4096 reserve
    assert decision["estimated_input_tokens"] == 100_000
    assert decision["context_token_budget"] == 80_000
    counter.assert_awaited_once()


def test_gpt_5_mini_reduced_output_cannot_exceed_documented_input_ceiling():
    provider, counter = _native_provider(input_tokens=300_000, default_model="gpt-5-mini")
    request = _request(output=1_000)
    decision = asyncio.run(provider.request_budget(request, context_estimate=300_000))

    assert decision["max_output_tokens"] == 1_000
    assert decision["input_limit_tokens"] == 272_000 - 4_096
    assert decision["estimated_input_tokens"] > decision["input_limit_tokens"]
    with pytest.raises(kernel_errors.ContextLengthError, match="native input allowance"):
        asyncio.run(provider.complete(request))
    provider.client.responses.create.assert_not_called()
    assert counter.await_count == 2  # preflight plus the independent dispatch guard


def test_models_without_an_input_ceiling_keep_existing_output_reserve_policy():
    provider = _provider(default_model="gpt-5.3")
    params = provider._budget_params(_request(output=1_000))
    assert provider._budget_input_limit(params) == 400_000 - 1_000 - 4_096


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
    attribution = decision["measurement"]["attribution"]
    assert decision["measurement"] == {
        "kind": "provider_count",
        "source": "official.operation",
        "input_tokens": 10,
        "attribution": attribution,
        "implementation": decision["measurement"]["implementation"],
    }
    assert type(decision["measurement"]["input_tokens"]) is int
    _assert_attribution(attribution, _count_projection(expected))
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
    events = _capture_events(provider)
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
    assert _wire_params(sdk_call.call_args.kwargs) == expected
    assert "request_options" not in sdk_call.call_args.kwargs
    [event] = _attribution_events(events)
    _assert_dispatch_event(
        event,
        operation="responses.stream" if use_streaming else "responses.create",
        stage="dispatch_ready", input_tokens=10, binding="matched",
    )
    _assert_attribution(event["attribution"], _count_projection(expected))
    _assert_descriptor(event["dispatch"], _count_projection(expected))
    assert event["dispatch"] == event["attribution"]["request"]
    assert counter.await_count == 1
    assert sdk_call.call_count == 1


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
    events = _capture_events(provider)
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
    assert events == []  # Successful preflight emits no attribution/request event.
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
    [event] = _attribution_events(events)
    before = decision["measurement"]["attribution"]
    after = event["attribution"]
    assert before["scope_id"] == after["scope_id"]
    assert before["count_id"] != after["count_id"]
    assert before["request"] == after["request"] == event["dispatch"]


@pytest.mark.parametrize("observer_fails", [False, True])
def test_native_count_over_allowance_blocks_generation_dispatch(observer_fails, caplog):
    provider, counter = _native_provider(default_model="gpt-5-mini")

    async def observer(name, _payload):
        if observer_fails and name == "provider:request_attribution":
            raise RuntimeError("private-rejection-observer-error")

    events = _capture_events(provider, observer)
    params = provider._budget_params(_request())
    counter.return_value = SimpleNamespace(
        input_tokens=provider._budget_input_limit(params) + 1
    )

    with pytest.raises(kernel_errors.ContextLengthError, match="native input allowance"):
        asyncio.run(provider.complete(_request()))

    assert counter.await_count == 1
    provider.client.responses.create.assert_not_called()
    provider.client.responses.stream.assert_not_called()
    [event] = _attribution_events(events)
    _assert_dispatch_event(
        event, operation="context_preparation", stage="rejected",
        input_tokens=counter.return_value.input_tokens, binding="matched",
    )
    _assert_attribution(event["attribution"], _count_projection(params))
    assert event["dispatch"] == event["attribution"]["request"]
    assert not any(name == "llm:request" for name, _ in events)
    assert "private-rejection-observer-error" not in caplog.text


def _attribution_request(*, content="réponse-a", instructions="policy-a", tool="tool_a"):
    return ChatRequest(
        messages=[
            Message(role="system", content=instructions),
            Message(role="user", content=content),
        ],
        tools=[
            ToolSpec(
                name=tool,
                description="A deterministic test tool.",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            )
        ],
        max_output_tokens=1_000,
    )


@pytest.mark.parametrize(
    "change",
    [{"content": "réponse-b"}, {"instructions": "policy-b"}, {"tool": "tool_b"}],
    ids=["content", "instructions", "tools"],
)
def test_equal_native_counts_do_not_identify_different_same_length_requests(change):
    # This reproduces equal numbers, not the historical incident's cause.
    # All requests use real assembly/count/dispatch; only the SDK is fake.
    provider, counter = _native_provider(
        default_model="gpt-5.6-terra", enable_long_context=True, input_tokens=499_947,
    )
    events = _capture_events(provider)
    original = _attribution_request()
    changed = _attribution_request(**change)
    decisions = [
        asyncio.run(provider.request_budget(request, context_estimate=400_000))
        for request in (original, original, changed)
    ]
    assert events == []
    projections = [call.kwargs for call in counter.call_args_list]
    assert projections[0] == projections[1]
    assert projections[0] != projections[2]
    assert _json_bytes(projections[0]) == _json_bytes(projections[2])
    receipts = []
    for decision, projection in zip(decisions, projections, strict=True):
        measurement = decision["measurement"]
        receipt = measurement["attribution"]
        assert measurement == {
            "kind": "provider_count", "source": "official.operation",
            "input_tokens": 499_947, "attribution": receipt,
            "implementation": measurement["implementation"],
        }
        assert type(measurement["input_tokens"]) is int
        _assert_attribution(receipt, projection)
        assert receipt["request"]["serialized_bytes"] != measurement["input_tokens"]
        receipts.append(receipt)
    first, identical, different = receipts
    assert len({receipt["scope_id"] for receipt in receipts}) == 1
    assert len({receipt["count_id"] for receipt in receipts}) == 3
    assert first["request"] == identical["request"]
    assert first["compact_input"] == identical["compact_input"]
    assert first["request"]["payload_tag"] != different["request"]["payload_tag"]
    # Sizes/type counts alone cannot distinguish this pair.
    assert {k: v for k, v in first["request"].items() if k != "payload_tag"} == {
        k: v for k, v in different["request"].items() if k != "payload_tag"
    }
    if "tool" in change:
        assert first["compact_input"] == different["compact_input"]
    else:
        assert first["compact_input"]["payload_tag"] != different["compact_input"]["payload_tag"]

    response = _response(499_947)
    response.output = []
    provider.client.responses.create.return_value = response
    for request in (original, changed):
        asyncio.run(provider.complete(request))
    dispatches = _attribution_events(events)
    assert len(dispatches) == 2
    for event, receipt in zip(dispatches, (first, different), strict=True):
        _assert_dispatch_event(
            event, operation="responses.create", stage="dispatch_ready",
            input_tokens=499_947, binding="matched",
        )
        assert event["dispatch"] == receipt["request"]
        assert event["attribution"]["count_id"] not in {r["count_id"] for r in receipts}
    assert dispatches[0]["dispatch_id"] != dispatches[1]["dispatch_id"]
    assert dispatches[0]["attribution"]["count_id"] != dispatches[1]["attribution"]["count_id"]
    assert counter.await_count == 5
    assert provider.client.responses.create.await_count == 2


def test_attribution_scope_and_payload_tags_are_provider_local():
    request = _attribution_request()
    first, first_counter = _native_provider(default_model="gpt-5-mini")
    second, second_counter = _native_provider(default_model="gpt-5-mini")
    first_receipt = asyncio.run(
        first.request_budget(request, context_estimate=10)
    )["measurement"]["attribution"]
    second_receipt = asyncio.run(
        second.request_budget(request, context_estimate=10)
    )["measurement"]["attribution"]

    assert first_counter.call_args.kwargs == second_counter.call_args.kwargs
    for receipt in (first_receipt, second_receipt):
        _assert_attribution(receipt, first_counter.call_args.kwargs)
    assert first_receipt["scope_id"] != second_receipt["scope_id"]
    assert first_receipt["count_id"] != second_receipt["count_id"]
    for key in ("request", "compact_input"):
        assert first_receipt[key]["serialized_bytes"] == second_receipt[key]["serialized_bytes"]
        assert first_receipt[key]["payload_tag"] != second_receipt[key]["payload_tag"]


def test_concurrent_measurements_keep_their_own_counts_and_payload_attribution():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    events = _capture_events(provider)
    requests = [_request("first", system="policy-a"), _request("second-longer", system="policy-b")]
    projections = []

    async def measure_concurrently():
        first_started = asyncio.Event()
        second_finished = asyncio.Event()

        async def count(**params):
            index = len(projections)
            projections.append(copy.deepcopy(params))
            if index == 0:
                first_started.set()
                await second_finished.wait()
            else:
                second_finished.set()
            return SimpleNamespace(input_tokens=(101, 202)[index])

        counter.side_effect = count
        async with asyncio.timeout(5):
            first = asyncio.create_task(provider.request_budget(requests[0], context_estimate=1))
            try:
                await first_started.wait()
                second = await provider.request_budget(requests[1], context_estimate=2)
                return [await first, second]
            finally:
                if not first.done():
                    first.cancel()
                    await asyncio.gather(first, return_exceptions=True)

    decisions = asyncio.run(measure_concurrently())
    counter.side_effect = None
    for index, decision in enumerate(decisions):
        tokens = (101, 202)[index]
        assert type(decision["measurement"]["input_tokens"]) is int
        assert decision["measurement"]["input_tokens"] == tokens
        receipt = decision["measurement"]["attribution"]
        _assert_attribution(receipt, projections[index])
        counter.return_value = SimpleNamespace(input_tokens=tokens)
        control = asyncio.run(
            provider.request_budget(requests[index], context_estimate=index + 1)
        )["measurement"]["attribution"]
        assert receipt["request"] == control["request"]
        assert receipt["count_id"] != control["count_id"]
    first_receipt, second_receipt = [d["measurement"]["attribution"] for d in decisions]
    assert first_receipt["scope_id"] == second_receipt["scope_id"]
    assert first_receipt["count_id"] != second_receipt["count_id"]
    assert first_receipt["request"]["payload_tag"] != second_receipt["request"]["payload_tag"]
    assert counter.await_count == 4
    assert events == []
    provider.client.responses.create.assert_not_called()
    provider.client.responses.stream.assert_not_called()


@pytest.mark.parametrize("use_streaming", [False, True])
@pytest.mark.parametrize("mutation_event", ["llm:request", "provider:request_attribution"])
def test_post_count_hook_mutation_fails_without_recount_retry_or_foreground_dispatch(
    monkeypatch, use_streaming, mutation_event,
):
    provider, counter = _native_provider(
        default_model="gpt-5-mini", use_streaming=use_streaming, max_retries=2,
    )
    retained_params = []
    real_count = provider._native_input_token_count

    async def count_and_retain(params):
        result = await real_count(params)
        retained_params.append(params)
        return result

    monkeypatch.setattr(provider, "_native_input_token_count", count_and_retain)

    async def mutate(name, _payload):
        if name == mutation_event:
            assert counter.await_count == 1
            assert len(retained_params) == 1
            # Mutate nested input so this remains observable even when the
            # low-level create seam makes a shallow copy of the parameter dict.
            retained_params[0]["input"][0]["content"] = "changed-after-count"

    events = _capture_events(provider, mutate)
    with pytest.raises(kernel_errors.LLMError) as raised:
        asyncio.run(provider.complete(_request("hello", system="policy-a")))

    assert raised.value.retryable is False
    assert counter.await_count == 1
    provider.client.responses.create.assert_not_called()
    provider.client.responses.stream.assert_not_called()
    event, rejected = _attribution_events(events)
    assert rejected["stage"] == "rejected" and rejected["binding"] == "mismatch"
    # A callback can mutate after the ready observation itself. That earlier
    # matched observation must not be interpreted as proof of an SDK/server send.
    _assert_dispatch_event(
        event, operation="responses.stream" if use_streaming else "responses.create",
        stage="dispatch_ready", input_tokens=10,
        binding="mismatch" if mutation_event == "llm:request" else "matched",
    )


@pytest.mark.parametrize("mutate", [False, True], ids=["unchanged-control", "changed"])
def test_direct_create_binds_a_retained_count_without_recount(mutate):
    provider, counter = _native_provider(default_model="gpt-5-mini", max_retries=2)
    events = _capture_events(provider)
    params = provider._budget_params(_request("hello", system="policy-a"))
    response = _response(10)
    provider.client.responses.create.return_value = response

    async def dispatch():
        # Reuse the existing count/dispatch seams, not new attribution helpers.
        count = await provider._native_input_token_count(params)
        if mutate:
            params["instructions"] = "policy-b"
        return await provider._create_response(params, native_input_tokens=count)

    if mutate:
        with pytest.raises(kernel_errors.LLMError) as raised:
            asyncio.run(dispatch())
        assert raised.value.retryable is False
        provider.client.responses.create.assert_not_called()
    else:
        assert asyncio.run(dispatch()) is response
        provider.client.responses.create.assert_awaited_once()
    assert counter.await_count == 1
    observations = _attribution_events(events)
    assert len(observations) == (2 if mutate else 1)
    event = observations[0]
    if mutate:
        assert observations[-1]["stage"] == "rejected"
    _assert_dispatch_event(
        event, operation="responses.create", stage="dispatch_ready",
        input_tokens=10, binding="mismatch" if mutate else "matched",
    )


@pytest.mark.parametrize("use_streaming", [False, True])
def test_plain_integer_count_never_claims_matched_attribution(monkeypatch, use_streaming):
    provider, counter = _native_provider(
        default_model="gpt-5-mini", use_streaming=use_streaming,
    )
    plain_counter = AsyncMock(return_value=10)
    monkeypatch.setattr(provider, "_native_input_token_count", plain_counter)
    events = _capture_events(provider)
    request = _request()
    decision = asyncio.run(provider.request_budget(request, context_estimate=10))
    assert decision["measurement"] == {
        "kind": "provider_count", "source": "official.operation",
        "input_tokens": 10, "attribution": None,
        "implementation": decision["measurement"]["implementation"],
    }
    assert type(decision["measurement"]["input_tokens"]) is int
    assert events == []
    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    provider.client.responses.stream.return_value = _StreamContext(response)
    asyncio.run(provider.complete(request))

    [event] = _attribution_events(events)
    _assert_dispatch_event(
        event, operation="responses.stream" if use_streaming else "responses.create",
        stage="dispatch_ready", input_tokens=10, binding="unavailable",
    )
    assert event["attribution"] is None
    _assert_descriptor(event["dispatch"], _count_projection(provider._budget_params(request)))
    assert plain_counter.await_count == 2
    counter.assert_not_called()
    assert provider.client.responses.create.call_count == (0 if use_streaming else 1)
    assert provider.client.responses.stream.call_count == (1 if use_streaming else 0)


@pytest.mark.parametrize("mutation", ["lost-projection", "output-reserve"])
def test_post_count_projection_loss_and_output_reserve_change_cannot_dispatch(mutation):
    provider, counter = _native_provider(default_model="gpt-5-mini", input_tokens=100_000)
    params = provider._budget_params(_request(output=1000))

    async def change(name, payload):
        if name == "provider:request_attribution" and payload["stage"] == "dispatch_ready":
            if mutation == "lost-projection":
                params["input"][0]["content"] = "changed"
                params["extra_body"] = {"input": "changed"}
            else:
                params["max_output_tokens"] = 390_000

    _capture_events(provider, change)

    async def dispatch():
        counted = await provider._native_input_token_count(params)
        # Exercise the dispatch helper's exact mutable parameter instance.
        await provider._before_counted_dispatch(params, counted, operation="responses.create")

    with pytest.raises(kernel_errors.LLMError):
        asyncio.run(dispatch())
    assert counter.await_count == 1
    provider.client.responses.create.assert_not_called()


@pytest.mark.parametrize("use_streaming", [False, True])
def test_slow_attribution_observer_is_bounded_without_generation_retry(use_streaming):
    provider, counter = _native_provider(default_model="gpt-5-mini",
                                         use_streaming=use_streaming, max_retries=2)
    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    provider.client.responses.stream.return_value = _StreamContext(response)
    async def slow(name, payload):
        if name == "provider:request_attribution":
            await asyncio.sleep(10)
    _capture_events(provider, slow)
    async def run():
        async with asyncio.timeout(1):
            await provider.complete(_request(), timeout=0.01)
    asyncio.run(run())
    assert counter.await_count == 1
    assert provider.client.responses.create.call_count == (0 if use_streaming else 1)
    assert provider.client.responses.stream.call_count == (1 if use_streaming else 0)


def test_scheduled_observer_mutation_cannot_change_frozen_sdk_payload():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request("counted-original"))
    expected = copy.deepcopy(params["input"])

    async def observer(name, payload):
        if name == "provider:request_attribution":
            asyncio.get_running_loop().call_soon(
                lambda: params["input"][0].update(content="later mutation"))
    _capture_events(provider, observer)
    async def sdk(**kwargs):
        await asyncio.sleep(0)
        assert kwargs["input"] == expected
        return _response(10)
    provider.client.responses.create.side_effect = sdk
    async def dispatch():
        count = await provider._native_input_token_count(params)
        await provider._create_response(params, native_input_tokens=count, timeout=1)
    asyncio.run(dispatch())
    assert counter.await_count == provider.client.responses.create.await_count == 1
    assert params["input"][0]["content"] == "later mutation"


@pytest.mark.parametrize("unknown_projection", [False, True])
def test_unavailable_native_measurement_is_not_fabricated_at_dispatch(unknown_projection):
    provider, counter = _native_provider(
        default_model="gpt-5-mini",
        extra_request_params={"unproved_input_option": "value"} if unknown_projection else {},
    )
    if not unknown_projection:
        provider.client.responses.input_tokens = SimpleNamespace()
    events = _capture_events(provider)
    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    asyncio.run(provider.complete(_request()))

    [event] = _attribution_events(events)
    _assert_dispatch_event(
        event, operation="responses.create", stage="dispatch_ready",
        input_tokens=None, binding="unavailable",
    )
    assert event["attribution"] is None
    if unknown_projection:
        assert event["dispatch"] is None
    else:
        _assert_descriptor(event["dispatch"], _count_projection(provider._budget_params(_request())))
    counter.assert_not_called()
    provider.client.responses.create.assert_awaited_once()


@pytest.mark.parametrize("use_streaming", [False, True])
def test_attribution_observer_failure_is_best_effort_without_model_or_count_replay(
    use_streaming, caplog,
):
    provider, counter = _native_provider(
        default_model="gpt-5-mini", use_streaming=use_streaming, max_retries=2,
    )
    private_error = "private-observer-exception-do-not-export"

    async def fail_observer(name, _payload):
        if name == "provider:request_attribution":
            raise RuntimeError(private_error)

    events = _capture_events(provider, fail_observer)
    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    provider.client.responses.stream.return_value = _StreamContext(response)
    asyncio.run(provider.complete(_request()))

    [event] = _attribution_events(events)
    assert event["binding"] == "matched"
    assert counter.await_count == 1
    assert provider.client.responses.create.call_count == (0 if use_streaming else 1)
    assert provider.client.responses.stream.call_count == (1 if use_streaming else 0)
    assert private_error not in caplog.text


def test_attribution_diagnostics_never_export_prompt_tool_arguments_url_or_unknown_type(caplog):
    sentinels = {
        "prompt": "private-prompt-sentinel",
        "instructions": "private-instructions-sentinel",
        "tool": "private_tool_name_sentinel",
        "argument": "private-argument-sentinel",
        "url": "https://private.invalid/secret-path?q=private-url-sentinel",
        "unknown_type": "private-unknown-type-sentinel",
    }
    wire_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": sentinels["prompt"]},
                {"type": "input_image", "image_url": sentinels["url"]},
            ],
        },
        {
            "type": "function_call", "call_id": "call_private",
            "name": sentinels["tool"], "arguments": json.dumps({"arg": sentinels["argument"]}),
        },
        {"type": "function_call_output", "call_id": "call_private", "output": "done"},
        {"type": sentinels["unknown_type"], "content": sentinels["prompt"]},
        {"type": "another-private-unknown-type", "content": sentinels["argument"]},
    ]
    provider, counter = _native_provider(
        default_model="gpt-5-mini", raw=False,
        extra_request_params={"input": wire_input},
    )
    events = _capture_events(provider)
    request = _attribution_request(
        content=sentinels["prompt"], instructions=sentinels["instructions"],
        tool=sentinels["tool"],
    )
    decision = asyncio.run(provider.request_budget(request, context_estimate=10))
    assert events == []
    receipt = decision["measurement"]["attribution"]
    projection = counter.call_args.kwargs
    assert projection["input"] == wire_input
    _assert_attribution(receipt, projection)
    assert receipt["request"]["input_types"] == {
        "message": 1, "function_call": 1, "function_call_output": 1, "other": 2,
    }
    response = _response(10)
    response.output = []
    provider.client.responses.create.return_value = response
    asyncio.run(provider.complete(request))
    [event] = _attribution_events(events)
    _assert_dispatch_event(
        event, operation="responses.create", stage="dispatch_ready",
        input_tokens=10, binding="matched",
    )
    _assert_attribution(event["attribution"], projection)
    _assert_descriptor(event["dispatch"], projection)
    # Round-trip the external contract: no private helper object/key is exported.
    exported = json.dumps({"measurement": decision["measurement"], "event": event})
    assert json.loads(exported)["measurement"] == decision["measurement"]
    for sentinel in (*sentinels.values(), "another-private-unknown-type", "test-key"):
        assert sentinel not in exported
        assert sentinel not in caplog.text
    assert counter.await_count == 2
    provider.client.responses.create.assert_awaited_once()


@pytest.mark.parametrize(
    ("reported", "available"),
    [(0, True), (None, False), (-1, False), (True, False), ("10", False)],
)
def test_native_count_validates_counter_response(reported, available):
    provider, _ = _native_provider(default_model="gpt-5-mini", input_tokens=reported)

    if available:
        decision = asyncio.run(provider.request_budget(_request(), context_estimate=10))
        assert decision["measurement"]["input_tokens"] == reported
    else:
        with pytest.raises(TokenCountError):
            asyncio.run(provider.request_budget(_request(), context_estimate=10))


def test_native_count_failure_and_cancellation_do_not_become_measurements():
    provider, counter = _native_provider(default_model="gpt-5-mini")
    counter.side_effect = RuntimeError("count response malformed")
    with pytest.raises(TokenCountError):
        asyncio.run(provider.request_budget(_request(), context_estimate=10))
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
    assert budget["input_limit_tokens"] == 272_000 - 4_096
    assert budget["context_token_budget"] == 123
    assert budget["max_output_tokens"] == 1_000


def test_cold_oversize_preflight_returns_none_without_state_or_log_side_effects(caplog):
    provider = _provider(default_model="gpt-5-mini")
    request = _request("cold-preflight-private-content-" * 15_000, output=1_000)
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

    assert _wire_params(provider.client.responses.create.call_args.kwargs) == expected


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

    assert _wire_params(provider.client.responses.stream.call_args.kwargs) == expected
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
    provider._budget_calibration["gpt-5-mini"] = (
        0.25,
        base_bytes,
        provider._budget_input_limit(base) - 10_000,
    )
    params = provider._budget_params(_request("x" * 20_000))

    with pytest.raises(kernel_errors.ContextLengthError, match="input_items=.*tools="):
        provider._guard_assembled_params(params)


@pytest.mark.parametrize("use_streaming", [False, True])
def test_cold_oversize_dispatches_once_warns_once_and_calibrates(use_streaming, caplog):
    private_content = "cold-dispatch-private-content-" * 15_000
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
    request = _request("cold-repeat-private-content-" * 15_000, output=1_000)

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
    request = _request("cold-model-private-content-" * 15_000, output=1_000)

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
    continuation_params = _wire_params(provider.client.responses.create.call_args_list[1].kwargs)
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


# Synthetic transport bytes exercise budgeting without image decoding or API calls.
_IMAGE_URL = (
    "data:image/png;base64,"
    + base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + b"synthetic-image-budget-fixture" * 40_000
    ).decode()
)


def _image_request():
    return _request(
        [
            {"type": "text", "text": "Compare these two images."},
            *[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _IMAGE_URL.split(",", 1)[1],
                    },
                }
                for _ in range(2)
            ],
        ]
    )


def _calibrate_text(provider):
    params = provider._budget_params(_request("small text calibration"))
    provider._record_budget_calibration(params, _response(100))
    return dict(provider._budget_calibration)


@pytest.mark.parametrize("use_streaming", [False, True])
def test_typed_images_without_native_count_dispatch_unchanged_without_calibration(
    use_streaming,
    caplog,
):
    provider = _provider(
        default_model="gpt-5.6-terra",
        enable_long_context=True,
        use_streaming=use_streaming,
    )
    calibration = _calibrate_text(provider)
    request = _image_request()
    expected = provider._budget_params(request)
    before = request.model_dump()
    estimate, _ = provider._estimated_input_tokens(expected)
    assert estimate > provider._budget_input_limit(expected)
    response = _response(50_000)
    response.output = []
    if use_streaming:
        provider.client.responses.stream = MagicMock(
            return_value=_StreamContext(response)
        )
    else:
        provider.client.responses.create = AsyncMock(return_value=response)

    caplog.clear()
    assert provider.request_budget(request, context_estimate=12_000) is None
    assert not caplog.records
    result = asyncio.run(provider.complete(request))
    sdk_call = (
        provider.client.responses.stream
        if use_streaming
        else provider.client.responses.create
    )
    assert sdk_call.call_count == 1
    assert _wire_params(sdk_call.call_args.kwargs) == expected
    assert request.model_dump() == before
    assert provider._budget_calibration == calibration
    assert result.usage.input_tokens == 50_000
    assert _IMAGE_URL not in caplog.text
    assert "non-text input" in caplog.text


@pytest.mark.parametrize("input_tokens", [10, 900_000])
def test_typed_images_native_count_remains_authoritative(input_tokens):
    provider, counter = _native_provider(
        default_model="gpt-5.6-terra",
        enable_long_context=True,
        input_tokens=input_tokens,
    )
    calibration = _calibrate_text(provider)
    request = _image_request()
    params = provider._budget_params(request)
    decision = asyncio.run(provider.request_budget(request, context_estimate=12_000))
    assert decision["measurement"]["input_tokens"] == input_tokens
    assert decision["input_limit_tokens"] == 767_904
    assert counter.call_args.kwargs == _count_projection(params)
    response = _response(input_tokens)
    response.output = []
    provider.client.responses.create.return_value = response
    if input_tokens > decision["input_limit_tokens"]:
        with pytest.raises(
            kernel_errors.ContextLengthError, match="native input allowance"
        ):
            asyncio.run(provider.complete(request))
        provider.client.responses.create.assert_not_called()
    else:
        asyncio.run(provider.complete(request))
        assert _wire_params(provider.client.responses.create.call_args.kwargs) == params
    assert counter.await_count == 2
    assert provider._budget_calibration == calibration


def test_failed_image_counter_stops_before_generation_without_usage_poisoning():
    provider, counter = _native_provider(
        default_model="gpt-5.6-terra", enable_long_context=True
    )
    calibration = _calibrate_text(provider)
    counter.side_effect = RuntimeError("synthetic counter unavailable")
    response = _response(50_000)
    response.output = []
    provider.client.responses.create.return_value = response
    request = _image_request()
    with pytest.raises(TokenCountError):
        asyncio.run(provider.request_budget(request, context_estimate=12_000))
    with pytest.raises(TokenCountError):
        asyncio.run(provider.complete(request))
    provider.client.responses.create.assert_not_called()
    assert provider._budget_calibration == calibration


@pytest.mark.parametrize(
    "item",
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://example.invalid/image.png",
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "file_id": "synthetic-file"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_file", "file_data": "synthetic-file-data"}],
        },
        {
            "type": "function_call_output",
            "call_id": "call",
            "output": [
                {
                    "type": "input_image",
                    "image_url": "https://example.invalid/image.png",
                }
            ],
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "call",
            "output": [{"type": "input_file", "file_id": "synthetic-file"}],
        },
        {
            "type": "computer_call_output",
            "call_id": "call",
            "output": {
                "type": "computer_screenshot",
                "image_url": "https://example.invalid/image.png",
            },
        },
    ],
)
def test_nontext_wire_positions_do_not_create_or_poison_text_calibration(item):
    provider = _provider(default_model="gpt-5-mini")
    params = {"model": "gpt-5-mini", "max_output_tokens": 1000, "input": [item]}
    before = copy.deepcopy(params)
    provider._record_budget_calibration(params, _response(100_000))
    assert provider._budget_calibration == {}
    calibration = _calibrate_text(provider)
    provider._record_budget_calibration(params, _response(100_000))
    assert provider._budget_calibration == calibration
    provider._budget_calibration["gpt-5-mini"] = (100_000.0, 1, 100_000)
    provider._guard_assembled_params(params)
    assert params == before


@pytest.mark.parametrize(
    "content",
    [
        _IMAGE_URL,
        json.dumps({"type": "input_image", "image_url": _IMAGE_URL}),
    ],
)
def test_textual_image_urls_and_quoted_image_json_keep_text_budget_enforcement(content):
    provider = _provider(default_model="gpt-5-mini")
    _calibrate_text(provider)
    request = _request(content)
    decision = provider.request_budget(request, context_estimate=100_000)
    assert decision["context_token_budget"] < 100_000
    with pytest.raises(kernel_errors.ContextLengthError, match="local input allowance"):
        provider._guard_assembled_params(provider._budget_params(request))


def test_tool_schema_image_examples_are_text_not_multimodal_input():
    provider = _provider(default_model="gpt-5-mini")
    params = provider._budget_params(_request())
    params["tools"] = [
        {
            "type": "function",
            "name": "example",
            "parameters": {
                "type": "object",
                "example": {"type": "input_image", "image_url": _IMAGE_URL},
            },
        }
    ]
    _calibrate_text(provider)
    with pytest.raises(kernel_errors.ContextLengthError, match="local input allowance"):
        provider._guard_assembled_params(params)


@pytest.mark.parametrize(
    "invalid",
    [
        {"max_output_tokens": -1},
        {"max_output_tokens": 900_000},
        {"metadata": {"invalid": object()}},
    ],
)
def test_typed_images_do_not_bypass_output_or_serialization_guards(invalid):
    provider = _provider(default_model="gpt-5.6-terra", enable_long_context=True)
    params = provider._budget_params(_image_request())
    params.update(invalid)
    with pytest.raises(
        (kernel_errors.ContextLengthError, kernel_errors.InvalidRequestError)
    ):
        provider._guard_assembled_params(params)


def test_tuple_typed_content_uses_the_same_multimodal_fallback():
    provider = _provider(default_model="gpt-5-mini")
    provider._budget_calibration["gpt-5-mini"] = (100_000.0, 1, 100_000)
    params = {
        "model": "gpt-5-mini",
        "input": (
            {
                "role": "user",
                "content": (
                    {
                        "type": "input_image",
                        "image_url": "https://example.invalid/image.png",
                    },
                ),
            },
        ),
    }
    provider._guard_assembled_params(params)


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "function_call_output",
            "call_id": "call",
            "output": json.dumps({"type": "input_image", "image_url": _IMAGE_URL}),
        },
        {
            "type": "function_call",
            "call_id": "call",
            "name": "example",
            "arguments": json.dumps({"type": "input_image", "image_url": _IMAGE_URL}),
        },
    ],
)
def test_tool_output_text_and_arguments_keep_the_text_guard(item):
    provider = _provider(default_model="gpt-5-mini")
    _calibrate_text(provider)
    params = {"model": "gpt-5-mini", "input": [item]}
    with pytest.raises(kernel_errors.ContextLengthError, match="local input allowance"):
        provider._guard_assembled_params(params)


def test_nontext_warning_is_scalar_only_once_per_model(caplog):
    provider = _provider(default_model="gpt-5-mini")
    params = {
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.invalid/private-image",
                    },
                ],
            }
        ],
    }
    caplog.clear()
    provider._guard_assembled_params(params)
    provider._guard_assembled_params(params)
    assert len(caplog.records) == 1
    assert "private-image" not in caplog.text
    assert "non-text input" in caplog.text
