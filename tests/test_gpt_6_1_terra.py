"""Exact Terra compatibility and per-attempt pricing; no live credentials."""

from __future__ import annotations

import asyncio
import json
import re
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import llm_errors as errors
from amplifier_core.message_models import ChatRequest, Message
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_capabilities
from amplifier_module_provider_openai._cost import compute_cost
from amplifier_module_provider_openai.image_operation import PROTOCOL

MODEL = "gpt-6.1-terra"
NEAR_MISSES = ["gpt-6-terra", "gpt-6.1", "gpt-6.1-terra-2099-01-01", "gpt-6.1-terra-pro"]


def _provider(**config):
    return OpenAIProvider(
        api_key="offline-fixture",
        config={
            "default_model": MODEL,
            "base_url": "https://offline.invalid/v1",
            "max_retries": 0,
            "use_streaming": False,
            "reasoning_summary": "auto",
            **config,
        },
    )


def _request(**kwargs):
    return ChatRequest(messages=[Message(role="user", content="Hello")], **kwargs)


def _response(*, input_tokens=272_000, status="completed", tier="default", breakdown=True):
    details = {"cached_tokens": 100_000, "cache_write_tokens": 50_000}
    if breakdown:
        details["cache_write_tokens_details"] = {
            "ttl_30m_tokens": 20_000,
            "ttl_12h_tokens": 30_000,
        }
    return {
        "id": "resp_terra",
        "object": "response",
        "created_at": 0,
        "model": MODEL,
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "service_tier": tier,
        "output": [{
            "id": "msg_terra", "type": "message", "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
        }],
        "usage": {
            "input_tokens": input_tokens, "output_tokens": 20_000,
            "total_tokens": input_tokens + 20_000,
            "input_tokens_details": details,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": False, "tools": [], "tool_choice": "auto",
        "temperature": 1, "top_p": 1,
    }


def _mock_response():
    # Lightweight response for parameter-only tests. Transport tests below use
    # real SDK decoding, including the newer cache-write usage breakdown.
    from openai.types.responses import Response

    return Response.model_construct(**{
        **_response(),
        "output": [],
        "usage": SimpleNamespace(
            input_tokens=100, output_tokens=10,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        ),
    })


def test_capabilities_are_exact_and_default_is_unchanged():
    caps = get_capabilities(MODEL)
    assert caps.family == MODEL
    assert (caps.context_window, caps.max_input_tokens, caps.max_output_tokens) == (
        1_050_000, 922_000, 128_000,
    )
    assert caps.long_context_pricing_threshold == 272_000
    assert caps.default_reasoning_effort is None
    assert caps.supports_reasoning and caps.supports_vision and caps.supports_streaming
    assert caps.supports_native_apply_patch and caps.supports_native_computer_use
    assert set(caps.capability_tags) == {"tools", "reasoning", "streaming", "json_mode", "vision"}
    assert not caps.supports_in_memory_retention
    assert OpenAIProvider(api_key="offline-fixture").default_model == "gpt-6.1-sol"
    assert _provider(default_model="gpt-5.4").default_model == "gpt-5.4"
    for model in NEAR_MISSES:
        assert get_capabilities(model).family == "unknown"
        assert compute_cost(model, prompt_tokens=100) is None


@pytest.mark.parametrize("long_context,window", [(False, 272_000), (True, 922_000)])
def test_discovery_and_config_gate(long_context, window):
    provider = _provider(enable_long_context=long_context, hide_dated_models=False)
    provider.client.models.list = AsyncMock(return_value=SimpleNamespace(
        data=[SimpleNamespace(id=model) for model in [MODEL, *NEAR_MISSES]]
    ))
    models = asyncio.run(provider.list_models())
    assert [(m.id, m.display_name, m.context_window, m.max_output_tokens) for m in models] == [
        (MODEL, "GPT 6.1 Terra", window, 128_000)
    ]
    assert provider.get_info().defaults["context_window"] == window
    field = next(f for f in provider.get_info().config_fields if f.id == "enable_long_context")
    pattern = field.show_when["default_model"].removeprefix("matches:")
    assert re.fullmatch(pattern, MODEL)
    assert all(re.fullmatch(pattern, model) is None for model in NEAR_MISSES)


@pytest.mark.parametrize("long_context,allowance", [(False, 267_904), (True, 917_904)])
def test_native_input_admission_reserves_output_and_safety_budget(long_context, allowance):
    provider = _provider(enable_long_context=long_context)
    params = {"model": MODEL, "input": [], "max_output_tokens": 128_000}
    # min(total minus output, independent input ceiling, optional price threshold)
    # minus the existing 4,096 reserve. Output is not subtracted twice.
    provider._guard_assembled_params(params, native_input_tokens=allowance)
    with pytest.raises(errors.ContextLengthError):
        provider._guard_assembled_params(params, native_input_tokens=allowance + 1)


@pytest.mark.parametrize("source", ["config", "nested", "request", "kwargs", "extras"])
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_effort_sources_preserve_literal_none_and_config_sentinel(source, effort):
    config, kwargs = {}, {}
    request = _request()
    if source == "config":
        config["reasoning_effort"] = effort
    elif source == "nested":
        config["reasoning"] = {"effort": effort}
    elif source == "request":
        request = _request(reasoning_effort=effort)
    elif source == "kwargs":
        kwargs["reasoning"] = {"effort": effort}
    else:
        config["extra_request_params"] = {"reasoning": {"effort": effort}}
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(return_value=_mock_response())
    asyncio.run(provider.complete(request, **kwargs))
    params = provider.client.responses.create.call_args.kwargs
    if source == "config" and effort == "none":
        assert "reasoning" not in params
        assert params["include"] == ["reasoning.encrypted_content"]
    else:
        assert params["reasoning"]["effort"] == effort
        if effort == "none":
            assert "include" not in params
        else:
            assert params["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("source", ["config", "nested", "request", "kwargs", "extras"])
def test_minimal_rejected_after_final_merge(source):
    config, kwargs = {}, {}
    request = _request(reasoning_effort="high")
    if source == "config":
        config["reasoning_effort"] = "minimal"
        request = _request()
    elif source == "nested":
        config["reasoning"] = {"effort": "minimal"}
        request = _request()
    elif source == "request":
        request = _request(reasoning_effort="minimal")
    elif source == "kwargs":
        request = _request()
        kwargs["reasoning"] = {"effort": "minimal"}
    else:
        config["extra_request_params"] = {"reasoning": {"effort": "minimal"}}
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock()
    with pytest.raises(errors.InvalidRequestError, match="minimal"):
        asyncio.run(provider.complete(request, **kwargs))
    provider.client.responses.create.assert_not_awaited()


def test_final_model_and_effort_override_are_validated_together():
    provider = _provider(
        default_model="gpt-6.1-sol",
        extra_request_params={"model": MODEL, "reasoning": {"effort": "none"}, "temperature": 0},
    )
    provider.client.responses.create = AsyncMock(return_value=_mock_response())
    asyncio.run(provider.complete(_request(reasoning_effort="high")))
    assert provider.client.responses.create.call_args.kwargs["model"] == MODEL
    assert provider.client.responses.create.call_args.kwargs["reasoning"]["effort"] == "none"


@pytest.mark.parametrize("effort", [None, "low"])
@pytest.mark.parametrize("extra", [
    {"temperature": 0}, {"top_p": 0.5}, {"top_logprobs": 3},
    {"include": ["message.output_text.logprobs"]},
])
def test_sampling_requires_explicit_none(effort, extra):
    provider = _provider(extra_request_params=extra)
    provider.client.responses.create = AsyncMock()
    with pytest.raises(errors.InvalidRequestError):
        asyncio.run(provider.complete(_request(reasoning_effort=effort)))
    provider.client.responses.create.assert_not_awaited()


def test_literal_none_allows_sampling_but_not_top_level_logprobs():
    extra = {
        "temperature": 0.2, "top_p": 0.5, "top_logprobs": 3,
        "include": ["reasoning.encrypted_content", "message.output_text.logprobs"],
    }
    provider = _provider(extra_request_params=extra)
    provider.client.responses.create = AsyncMock(return_value=_mock_response())
    asyncio.run(provider.complete(_request(reasoning_effort="none")))
    params = provider.client.responses.create.call_args.kwargs
    assert {key: params[key] for key in ("temperature", "top_p", "top_logprobs")} == {
        "temperature": 0.2, "top_p": 0.5, "top_logprobs": 3,
    }
    assert params["include"] == ["message.output_text.logprobs"]
    provider.extra_request_params["logprobs"] = True
    provider.client.responses.create.reset_mock()
    with pytest.raises(errors.InvalidRequestError, match="logprobs"):
        asyncio.run(provider.complete(_request(reasoning_effort="none")))
    provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("source", ["config", "kwargs", "extras"])
@pytest.mark.parametrize("ttl", ["30m", "12h", "24h", "1h", "", None])
def test_ttl_validation_and_legacy_retention_removal(source, ttl):
    config, kwargs = {"prompt_cache_retention": "24h"}, {}
    if source == "config":
        config["prompt_cache_options"] = {"ttl": ttl}
    elif source == "kwargs":
        kwargs["prompt_cache_options"] = {"ttl": ttl}
    else:
        config["extra_request_params"] = {
            "prompt_cache_options": {"ttl": ttl}, "prompt_cache_retention": "in_memory",
        }
    provider = _provider(**config)
    provider.client.responses.create = AsyncMock(return_value=_mock_response())
    if ttl in {"30m", "12h"}:
        asyncio.run(provider.complete(_request(), **kwargs))
        params = provider.client.responses.create.call_args.kwargs
        assert params["prompt_cache_options"] == {"ttl": ttl}
        assert "prompt_cache_retention" not in params
    else:
        with pytest.raises(errors.InvalidRequestError, match="ttl"):
            asyncio.run(provider.complete(_request(), **kwargs))
        provider.client.responses.create.assert_not_awaited()


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"])
def test_all_documented_gpt_6_models_accept_extended_ttl(model):
    provider = _provider(default_model=model, prompt_cache_options={"ttl": "12h"})
    provider.client.responses.create = AsyncMock(return_value=_mock_response())
    asyncio.run(provider.complete(_request()))
    assert provider.client.responses.create.call_args.kwargs["prompt_cache_options"] == {
        "ttl": "12h"
    }


@pytest.mark.parametrize("tokens,standard", [
    # Fresh=122000, cached=100000, 30m=20000, 12h=30000, output=20000.
    # Short: .122 + .005 + .025 + .075 + .100 = .327
    (272_000, Decimal("0.327")),
    # Long: .244002 + .010 + .050 + .150 + .150 = .604002
    (272_001, Decimal("0.604002")),
])
@pytest.mark.parametrize("tier,multiplier", [
    ("default", Decimal(1)), ("flex", Decimal(".5")),
    ("fast", Decimal(2)), ("priority", Decimal(2)),
])
def test_independent_mixed_lifetime_price_oracles(tokens, standard, tier, multiplier):
    assert compute_cost(
        MODEL, prompt_tokens=tokens, completion_tokens=20_000,
        cached_tokens=100_000, cache_write_tokens=50_000,
        cache_write_tokens_details={"ttl_30m_tokens": 20_000, "ttl_12h_tokens": 30_000},
        service_tier=tier,
    ) == standard * multiplier


@pytest.mark.parametrize("details", [
    None, {}, {"ttl_30m_tokens": 1}, {"ttl_12h_tokens": 50_001},
    {"ttl_30m_tokens": -1, "ttl_12h_tokens": 50_001},
    {"ttl_30m_tokens": True, "ttl_12h_tokens": 49_999},
    {"ttl_30m_tokens": "20000", "ttl_12h_tokens": 30_000},
])
def test_ambiguous_or_malformed_write_lifetimes_have_unknown_cost(details):
    assert compute_cost(
        MODEL, prompt_tokens=272_000, cache_write_tokens=50_000,
        cache_write_tokens_details=details,
    ) is None


@pytest.mark.parametrize("details,expected", [
    ({"ttl_30m_tokens": 1000}, Decimal(".00125")),
    ({"ttl_12h_tokens": 1000}, Decimal(".0025")),
])
def test_single_lifetime_writes_are_not_charged_as_fresh_input(details, expected):
    assert compute_cost(
        MODEL, prompt_tokens=1000, cache_write_tokens=1000,
        cache_write_tokens_details=details,
    ) == expected


@pytest.mark.parametrize("tier", [None, "auto", "unpriced"])
def test_unknown_tiers_stay_unknown_even_without_writes(tier):
    assert compute_cost(MODEL, prompt_tokens=1, service_tier=tier) is None
    assert compute_cost(MODEL, prompt_tokens=1000) == Decimal(".001")


def test_typed_sdk_lifetime_breakdown_preserves_normalized_usage_and_cost():
    response = _mock_response()
    response.usage = SimpleNamespace(
        input_tokens=1000, output_tokens=100,
        input_tokens_details=SimpleNamespace(
            cached_tokens=200, cache_write_tokens=300,
            cache_write_tokens_details=SimpleNamespace(ttl_30m_tokens=100, ttl_12h_tokens=200),
        ),
    )
    converted = _provider()._convert_to_chat_response(response)
    assert converted.usage.input_tokens == 700
    assert converted.usage.cache_write_tokens == 300
    # .0005 fresh + .00001 cached + .000125 30m + .0005 12h + .0005 output.
    assert converted.usage.cost_usd == Decimal(".001635")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("unknown", [
    None, "lifetime", "tier", "missing_count", "null_count", "missing_cached",
])
async def test_real_sdk_usage_events_and_per_attempt_accumulation(streaming, continuation, unknown):
    sent, reported = [], []
    responses = [_response(status="incomplete" if continuation else "completed")]
    if continuation:
        responses.append(_response(input_tokens=272_001, tier="flex"))
    if unknown == "lifetime":
        responses[0]["usage"]["input_tokens_details"].pop("cache_write_tokens_details")
    elif unknown == "tier":
        responses[0]["service_tier"] = None
    elif unknown in {"missing_count", "null_count"}:
        details = responses[0]["usage"]["input_tokens_details"]
        details.pop("cache_write_tokens_details")
        if unknown == "missing_count":
            details.pop("cache_write_tokens")
        else:
            details["cache_write_tokens"] = None
    elif unknown == "missing_cached":
        responses[0]["usage"]["input_tokens_details"].pop("cached_tokens")

    def respond(req):
        body = json.loads(req.content)
        sent.append(body)
        payload = responses[len(sent) - 1]
        assert body["model"] == MODEL
        assert body["prompt_cache_options"] == {"ttl": "12h"}
        assert "prompt_cache_retention" not in body
        if body.get("stream"):
            events = [
                {"type": "response.created", "sequence_number": 0, "response": payload},
                {"type": f"response.{payload['status']}", "sequence_number": 1, "response": payload},
            ]
            return httpx.Response(200, text="".join(
                f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
            ), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=payload)

    coordinator = SimpleNamespace(
        hooks=SimpleNamespace(emit=AsyncMock()), get_capability=lambda _: None,
    )
    async with AsyncOpenAI(
        api_key="offline-fixture", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        provider = _provider(use_streaming=streaming, prompt_cache_options={"ttl": "12h"})
        provider._client = client
        # Native count is a separate boundary, irrelevant to these SDK usage checks.
        provider._provider_count_available = lambda: False
        provider.coordinator = coordinator
        provider._add_cost = reported.append
        result = await provider.complete(_request())

    # .327 short Standard + (.604002 * .5) long Flex. Never price merged
    # token totals as one long request or infer lifetime from the requested TTL.
    expected = None if unknown else Decimal(".629001") if continuation else Decimal(".327")
    assert len(sent) == (2 if continuation else 1)
    assert result.usage.cost_usd == expected
    missing_count = unknown in {"missing_count", "null_count"}
    assert result.usage.input_tokens == (
        (494_001 if continuation else 272_000) if missing_count
        else (444_001 if continuation else 222_000)
    )
    assert result.usage.cache_write_tokens == (
        (50_000 if continuation else None) if missing_count
        else (100_000 if continuation else 50_000)
    )
    assert result.usage.cache_read_tokens == (
        (100_000 if continuation else None) if unknown == "missing_cached"
        else (200_000 if continuation else 100_000)
    )
    assert reported == ([] if expected is None else [expected])
    event = next(call.args[1] for call in coordinator.hooks.emit.await_args_list
                 if call.args[0] == "llm:response")
    assert event["usage"]["cost_usd"] == (
        str(result.usage.cost_usd) if expected is not None else None
    )
    assert event["usage"].get("cache_write_tokens") == result.usage.cache_write_tokens
    assert event["usage"]["input_tokens"] == result.usage.input_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("model,supported", [(MODEL, True), *[(m, False) for m in NEAR_MISSES]])
async def test_image_controller_prepare_is_exact_and_never_generates(model, supported):
    request = {
        "protocol": PROTOCOL, "request_id": "terra-image-prepare",
        "operation": "generate", "prompt": "A simple blue square", "inputs": [],
        "selection": {"model": model, "config": {"image_generation": {
            "model": "gpt-image-2.5-flare", "transport": "responses",
        }}},
        "output": {"count": 1, "mime_type": "image/png", "size": "auto",
                   "quality": "auto", "background": "auto"},
    }
    sent = []
    async with AsyncOpenAI(api_key="offline-fixture", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: sent.append(req)),
    )) as client:
        provider = OpenAIProvider(api_key="offline-fixture", client=client)
        prepared = await provider.prepare_image_generation(request)
    assert prepared["supported"] is supported
    assert sent == []