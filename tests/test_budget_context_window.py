"""Exact effective-model window evidence, independent of budget constraints."""

import copy
import inspect
import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import pytest
from amplifier_core.message_models import ChatRequest, Message
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import get_known_context_window


@asynccontextmanager
async def provider_fixture(*, native, endpoint="https://api.openai.com/v1", provider_type=OpenAIProvider, **config):
    requests = []

    def receive(request):
        # Any added catalog, count or generation request fails the call-count oracle.
        assert request.method == "POST"
        assert str(request.url) == endpoint.rstrip("/") + "/responses/input_tokens"
        assert request.headers["authorization"] == "Bearer offline-window-fixture"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"input_tokens": 42})

    async with AsyncOpenAI(
        api_key="offline-window-fixture", base_url=endpoint, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive), trust_env=False),
    ) as client:
        if not native:
            client.responses.input_tokens.count = None
        provider = provider_type(client=client, config={
            "default_model": "gpt-6.1-sol", "max_retries": 0, **config,
        })
        yield provider, requests


async def decision(provider, request, *, native, **options):
    result = provider.request_budget(request, context_estimate=20, **options)
    assert inspect.isawaitable(result) is native
    return await result if native else result


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("configured,selected,options,extras,effective,window", [
    ("gpt-6.1-sol", "gpt-6-astra", {}, {}, "gpt-6-astra", 1_050_000),
    ("gpt-6.1-sol", "gpt-5.2", {}, {}, "gpt-5.2", 400_000),
    ("gpt-5.2", None, {}, {}, "gpt-5.2", 400_000),
    ("gpt-5.2", "gpt-5.4-mini", {"model": "gpt-6-astra"}, {}, "gpt-6-astra", 1_050_000),
    ("gpt-6.1-sol", "gpt-6-astra", {"request_options": {"model": "gpt-5.2"}}, {}, "gpt-5.2", 400_000),
    ("gpt-5.2", "gpt-5.2", {"request_options": {"model": "gpt-5.4-mini"}, "model": "gpt-6-astra"}, {}, "gpt-6-astra", 1_050_000),
    ("gpt-6.1-sol", "gpt-6-astra", {"model": "gpt-6-sol"}, {"model": "gpt-5.2"}, "gpt-5.2", 400_000),
    ("gpt-5.2", "gpt-5.6", {}, {}, "gpt-5.6", 1_050_000),
])
async def test_window_follows_final_wire_model_without_changing_existing_budget(
    native, configured, selected, options, extras, effective, window,
):
    async with provider_fixture(native=native, default_model=configured, extra_request_params=extras) as (p, calls):
        request = ChatRequest(messages=[Message(role="user", content="hello")], model=selected, max_output_tokens=1_000)
        before = request.model_dump()
        config_before = copy.deepcopy(p.config)
        assert p._budget_params(request, **p._merge_request_options(options.get("request_options"),
               {k: v for k, v in options.items() if k != "request_options"}))["model"] == effective
        with patch.object(p, "get_info", side_effect=AssertionError("default metadata is not evidence")):
            actual = await decision(p, request, native=native, **options)
            assert actual.pop("context_window_tokens") == window
            # The old envelope is bit-for-bit unchanged with evidence omitted.
            with patch.object(p, "_budget_window_metadata", return_value={}):
                original = await decision(p, request, native=native, **options)
        assert actual == original
        assert request.model_dump() == before and p.config == config_before
        assert len(calls) == (2 if native else 0)
        assert all(call["model"] == effective for call in calls)
        assert all("context_window_tokens" not in call for call in calls)
        assert p.default_model == configured


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("long_context,input_limit", [(False, 267_904), (True, 917_904)])
@pytest.mark.parametrize("output", [1_000, 128_000])
async def test_full_astra_window_is_not_input_ceiling_price_threshold_or_output_reserve(
    native, long_context, input_limit, output,
):
    async with provider_fixture(native=native, enable_long_context=long_context) as (p, calls):
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-astra", max_output_tokens=output)
        result = await decision(p, request, native=native)
        assert result["context_window_tokens"] == 1_050_000
        assert result["input_limit_tokens"] == input_limit
        assert result["max_output_tokens"] == output
        assert result["context_token_budget"] == 20
        assert result["context_window_tokens"] != input_limit + output
        assert len(calls) == (1 if native else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("model", [
    "gpt-6-new-tier", "gpt-6-astra-2099-01-01", "gpt-5.2-2099-01-01",
    "gpt-5.5-deployment", "gpt-5-future", "unknown-deployment", "GPT-6-ASTRA",
])
async def test_unknown_or_inferred_model_has_no_window_evidence(native, model):
    async with provider_fixture(native=native) as (p, calls):
        request = ChatRequest(messages=[Message(role="user", content="hello")], model=model, max_output_tokens=1_000)
        result = await decision(p, request, native=native)
        assert "context_window_tokens" not in result
        assert len(calls) == (1 if native else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [
    "https://gateway.invalid/v1", "https://api.openai.com/custom", "http://api.openai.com/v1",
])
async def test_custom_actual_endpoint_overrides_standard_config_and_omits_window(endpoint):
    async with provider_fixture(native=True, endpoint=endpoint, base_url="https://api.openai.com/v1") as (p, calls):
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-astra")
        result = await decision(p, request, native=False)
        assert "context_window_tokens" not in result
        # Both decision constructors enforce the same route constraint.
        assert "context_window_tokens" not in p._budget_decision(
            p._budget_params(request), context_estimate=20, estimated_input_tokens=42,
        )
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("body,wire_model,window", [
    ({"model": "gpt-5.2"}, "gpt-5.2", None),
    ({"model": "unknown-deployment"}, "unknown-deployment", None),
    ({"model": None}, None, None),
    ({"model": 42}, 42, None),
    ({"model": "gpt-6-astra"}, "gpt-6-astra", 1_050_000),
    ({"metadata": {"fixture": "overlay"}}, "gpt-6-astra", 1_050_000),
])
async def test_sdk_overlay_window_matches_wire_identity_or_is_omitted(body, wire_model, window):
    wires = []

    def receive(request):
        assert request.method == "POST" and request.url.path == "/v1/responses"
        wires.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_offline", "object": "response", "created_at": 0,
            "status": "completed", "model": "offline-fixture", "output": [],
        })

    async with AsyncOpenAI(
        api_key="offline-overlay", max_retries=0, base_url="https://api.openai.com/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive), trust_env=False),
    ) as client:
        p = OpenAIProvider(client=client, config={"extra_request_params": {"extra_body": body}})
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-astra",
                              max_output_tokens=1_000)
        params = p._budget_params(request)
        before = copy.deepcopy(params)
        # Direct native counting already refuses this unsupported SDK envelope.
        assert await p.request_budget(request, context_estimate=20) is None
        client.responses.input_tokens.count = None
        result = p.request_budget(request, context_estimate=20)
        assert result.get("context_window_tokens") == window
        assert p._budget_decision(params, context_estimate=20, estimated_input_tokens=42).get(
            "context_window_tokens"
        ) == window
        with patch.object(p, "_budget_window_metadata", return_value={}):
            prior = p.request_budget(request, context_estimate=20)
        assert {k: v for k, v in result.items() if k != "context_window_tokens"} == prior
        assert result["input_limit_tokens"] == 267_904  # Existing pre-overlay fitting is unchanged.
        assert result["max_output_tokens"] == 1_000
        assert wires == []  # Metadata/preflight adds no IO.
        # Exercise the real SDK overlay, not a hand-written JSON merge.
        await client.responses.create(**{**params, "stream": False})
        assert len(wires) == 1 and wires[0]["model"] == wire_model
        assert "context_window_tokens" not in wires[0]
        assert params == before and p.config["extra_request_params"]["extra_body"] == body


@pytest.mark.parametrize("body", [None, [], "unsupported"])
def test_sdk_overlay_non_object_has_no_window_evidence(body):
    p = OpenAIProvider(api_key="unused")
    assert p._budget_window_metadata({"model": "gpt-6-astra", "extra_body": body}) == {}
    assert p._client is None


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_present,window", [(False, None), (True, 1_050_000)])
async def test_native_sdk_overlay_folded_and_fallback_window(owner_present, window):
    from types import SimpleNamespace

    from amplifier_module_provider_openai.native import NativeResponsesProvider

    async with provider_fixture(native=True, extra_request_params={"extra_body": {"model": "gpt-6-astra"}}) as (p, calls):
        owner = SimpleNamespace() if owner_present else None
        wrapper = NativeResponsesProvider.wrap(p, owner_getter=lambda: owner)
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-5.2",
                              max_output_tokens=1_000)
        result = await wrapper.request_budget(request, context_estimate=20)
        assert result.get("context_window_tokens") == window
        assert result["input_limit_tokens"] == (267_904 if owner_present else 394_904)
        with patch.object(wrapper, "_budget_window_metadata", return_value={}):
            prior = await wrapper.request_budget(request, context_estimate=20)
        assert {k: v for k, v in result.items() if k != "context_window_tokens"} == prior
        assert calls == []


@pytest.mark.asyncio
async def test_non_openai_subclass_cannot_inherit_public_model_window_claim():
    class DeploymentProvider(OpenAIProvider):
        name = "deployment"

    async with provider_fixture(native=True, provider_type=DeploymentProvider) as (p, calls):
        result = await decision(p, ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-astra"), native=False)
        assert "context_window_tokens" not in result
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("selected,window", [("gpt-6-astra", 1_050_000), ("gpt-5.2", 400_000)])
async def test_native_transport_wrapper_uses_effective_model_without_extra_count(selected, window):
    from amplifier_module_provider_openai.native import NativeResponsesProvider

    async with provider_fixture(native=True) as (original, calls):
        provider = NativeResponsesProvider.wrap(original, owner_getter=lambda: None)
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-sol")
        result = await provider.request_budget(
            request, context_estimate=20, request_options={"model": selected},
        )
        assert result["context_window_tokens"] == window
        assert provider.default_model == "gpt-6.1-sol"
        assert "measurement" not in result  # No change to this wrapper's count support.
        assert calls == []


def test_local_window_lookup_does_not_create_client_or_use_family_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    p = OpenAIProvider(api_key="unused")
    assert p._budget_window_metadata({"model": "gpt-6-astra"}) == {"context_window_tokens": 1_050_000}
    assert p._client is None
    assert get_known_context_window("gpt-5.2") == 400_000
    assert get_known_context_window("gpt-5.2-2099-01-01") is None
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    assert p._budget_window_metadata({"model": "gpt-6-astra"}) == {}
    assert p._client is None


@pytest.mark.asyncio
async def test_unknown_injected_endpoint_omits_window():
    from types import SimpleNamespace

    p = OpenAIProvider(client=SimpleNamespace(), config={"base_url": "https://api.openai.com/v1"})
    result = await decision(p, ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-astra"), native=False)
    assert "context_window_tokens" not in result


@pytest.mark.asyncio
async def test_unavailable_budget_stays_none_not_window_only_success():
    async with provider_fixture(native=True) as (p, calls):
        request = ChatRequest(messages=[Message(role="user", content="hello")], model="gpt-6-astra")
        with patch.object(p, "_native_count_params", return_value=None):
            assert await decision(p, request, native=True) is None
        assert calls == []
