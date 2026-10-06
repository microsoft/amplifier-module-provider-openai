"""Usage-only settlement: measured failure is not a delivered response."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._generation_errors import RequestOutcomeUnknownError


def vendor_usage(inputs=100, outputs=10, *, writes=0, reads=0):
    return SimpleNamespace(
        input_tokens=inputs, output_tokens=outputs,
        input_tokens_details=SimpleNamespace(cache_write_tokens=writes, cached_tokens=reads),
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )


def response(usage, tier="default", **extra):
    return SimpleNamespace(model="gpt-6-astra", service_tier=tier, usage=usage, **extra)


def provider():
    return OpenAIProvider(
        api_key="offline-placeholder",
        config={"default_model": "gpt-6-astra", "use_streaming": False, "max_retries": 1},
    )


def test_absent_usage_and_measured_zero_are_distinct():
    p = provider()
    p._add_cost = MagicMock()
    absent = p._account_failed_responses([response(None)])
    p._add_cost.assert_not_called()
    zero = p._account_failed_responses([response(vendor_usage(0, 0))])
    assert absent["input_tokens"] is absent["output_tokens"] is absent["cost_usd"] is None
    assert zero["input_tokens"] == zero["output_tokens"] == zero["total_tokens"] == 0
    assert Decimal(zero["cost_usd"]) == 0
    assert zero["cost_complete"] is True and absent["cost_complete"] is False
    p._add_cost.assert_called_once_with(Decimal(0))


def test_usage_only_cache_normalization_does_not_mutate_tool_state():
    p = provider()
    p._add_cost = MagicMock()
    failed = response(vendor_usage(1000, 100, writes=300, reads=200))
    # Any full-response conversion would parse or fail on this private object.
    failed.output = object()
    failed.id = "private-response-id"
    usage = p._account_failed_responses([failed])
    assert usage["input_tokens"] == 700
    assert usage["cache_write_tokens"] == 300
    assert usage["cache_read_tokens"] == 200
    assert usage["output_tokens"] == 100 and usage["total_tokens"] == 800
    assert usage["reasoning_tokens"] == 2
    p._add_cost.assert_called_once_with(Decimal("0.013950"))
    assert p._native_call_ids == set()
    assert "private" not in repr(usage)


@pytest.mark.parametrize("inputs,outputs", [(None, 10), (100, None), (-1, 10), (True, 10)])
def test_missing_or_invalid_counts_never_price_as_zero(inputs, outputs):
    p = provider()
    p._add_cost = MagicMock()
    usage = p._account_failed_responses([response(vendor_usage(inputs, outputs))])
    assert usage["cost_usd"] is None and usage["cost_complete"] is False
    assert usage["total_tokens"] is None
    p._add_cost.assert_not_called()


def test_mixed_known_unknown_attempts_preserve_subtotal_not_complete_total():
    p = provider()
    p._add_cost = MagicMock()
    usage = p._account_failed_responses([
        response(vendor_usage()), response(None), response(vendor_usage(), tier=None),
    ])
    assert len(usage["attempts"]) == 3
    assert usage["input_tokens"] is usage["output_tokens"] is None
    assert usage["attempts"][0]["input_tokens"] == 100
    assert usage["attempts"][2]["input_tokens"] == 100
    assert usage["cost_usd"] is None and usage["cost_complete"] is False
    assert usage["cost_known_subtotal_usd"] == "0.0015"
    assert usage["cost_scope"] == "known_attempts"
    p._add_cost.assert_called_once_with(Decimal("0.0015"))


def test_failed_continuation_preserves_prior_usage_without_committing_content():
    p = provider()
    p._add_cost = MagicMock()
    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=AsyncMock()))
    initial = response(
        vendor_usage(), status="incomplete", id="fixture-incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[],  # No accepted canonical content is invented.
    )
    p.client.responses.create = AsyncMock(side_effect=[initial, RuntimeError("private-error")])
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=100)
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        asyncio.run(p.complete(ChatRequest(messages=[Message(role="user", content="Hello")])))
    assert p.client.responses.create.await_count == 2  # Original + continuation, no replacement.
    usage = caught.value.usage
    assert usage["input_tokens"] is None
    assert usage["attempts"][0]["input_tokens"] == 100
    assert usage["attempts"][1]["input_tokens"] is None
    assert usage["cost_usd"] is None and usage["cost_complete"] is False
    assert usage["cost_known_subtotal_usd"] == "0.0015"
    p._add_cost.assert_called_once_with(Decimal("0.0015"))
    events = [call.args for call in p.coordinator.hooks.emit.await_args_list]
    results = [data for name, data in events if name == "llm:response"]
    assert len(results) == 1 and results[0]["status"] == "error"
    assert results[0]["usage"] == usage
    assert "private-error" not in repr(results)


@pytest.mark.parametrize("writes,reads", [(-1, 0), ("10", 0), (True, 0), (800, 300)])
def test_invalid_cache_buckets_keep_cost_and_derived_input_unknown(writes, reads):
    p = provider()
    p._add_cost = MagicMock()
    usage = p._account_failed_responses([response(vendor_usage(1000, 10, writes=writes, reads=reads))])
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] == 10
    assert usage["cost_usd"] is None
    p._add_cost.assert_not_called()


def test_terminal_hook_failure_does_not_add_success_cost_twice():
    p = provider()
    p._add_cost = MagicMock()
    p._create_response = AsyncMock(return_value=response(vendor_usage(), status="completed", output=[]))
    events = []

    async def emit(name, payload):
        if name == "llm:response" and payload["status"] == "ok":
            raise RuntimeError("private-hook-failure")
        events.append((name, payload))

    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        asyncio.run(p.complete(ChatRequest(messages=[Message(role="user", content="Hello")])))
    p._add_cost.assert_called_once_with(Decimal("0.0015"))
    assert caught.value.usage["cost_usd"] == "0.0015"
    assert len(caught.value.usage["attempts"]) == 1
    assert [data for name, data in events if name == "llm:response"][0]["status"] == "error"


@pytest.mark.parametrize("initial_count", [100, None])
def test_continuation_count_rejection_does_not_invent_a_generation(initial_count):
    p = provider()
    p._add_cost = MagicMock()
    p.client.responses.create = AsyncMock(return_value=response(
        vendor_usage(), status="incomplete", id="fixture-incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"), output=[],
    ))
    count_results = [initial_count]
    if initial_count is None:
        count_results.append(None)  # Original create helper's unavailable-count fallback.
    count_results.append(llm_errors.ContextLengthError("fixture count rejection", retryable=False))
    p._guard_assembled_params_with_provider_count = AsyncMock(side_effect=count_results)
    with pytest.raises(llm_errors.ContextLengthError) as caught:
        asyncio.run(p.complete(ChatRequest(messages=[Message(role="user", content="Hello")])))
    assert p.client.responses.create.await_count == 1
    assert len(caught.value.usage["attempts"]) == 1
    assert caught.value.usage["input_tokens"] == 100
    assert caught.value.usage["cost_complete"] is True
    p._add_cost.assert_called_once_with(Decimal("0.0015"))


def test_slow_continuation_count_preserves_explicit_attempt_timeout():
    p = provider()
    p._add_cost = MagicMock()
    p.client.responses.create = AsyncMock(return_value=response(
        vendor_usage(), status="incomplete", id="fixture-incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"), output=[],
    ))
    counts = []

    async def count(_params):
        counts.append(1)
        if len(counts) == 1:
            return 100
        await asyncio.sleep(999)

    p._guard_assembled_params_with_provider_count = count
    with pytest.raises(llm_errors.LLMTimeoutError) as caught:
        asyncio.run(p.complete(ChatRequest(
            messages=[Message(role="user", content="Hello")], timeout=0.02,
        )))
    assert len(counts) == 2
    assert p.client.responses.create.await_count == 1
    assert len(caught.value.usage["attempts"]) == 1
    assert caught.value.usage["input_tokens"] == 100
    assert caught.value.usage["cost_complete"] is True
    p._add_cost.assert_called_once_with(Decimal("0.0015"))


@pytest.mark.parametrize("hooks_enabled", [False, True])
def test_duplicate_failed_terminals_never_charge_or_convert_twice(hooks_enabled):
    p = provider()
    p.use_streaming = True
    p._add_cost = MagicMock()
    if hooks_enabled:
        p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=AsyncMock()))
    failed = response(vendor_usage(), status="failed", output=object())
    duplicate = response(vendor_usage(), status="failed", output=object())
    completed = response(vendor_usage(), status="completed", output=object())

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def __aiter__(self):
            async def events():
                for item in [failed, duplicate, completed]:
                    yield SimpleNamespace(type="response." + item.status, response=item)
            return events()

        get_final_response = AsyncMock(return_value=completed)

    stream = Stream()
    p.client.responses.stream = MagicMock(return_value=stream)
    p._convert_to_chat_response = MagicMock(side_effect=AssertionError("must not convert"))
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        asyncio.run(p.complete(ChatRequest(messages=[Message(role="user", content="Hello")])))
    assert p.client.responses.stream.call_count == 1
    assert len(caught.value.usage["attempts"]) == 1
    p._add_cost.assert_called_once_with(Decimal("0.0015"))
    p._convert_to_chat_response.assert_not_called()
    stream.get_final_response.assert_not_called()