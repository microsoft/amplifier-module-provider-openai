"""Exact GPT-6.2 policy and final SDK/native wires; entirely offline."""

import copy
import json
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ThinkingBlock,
    ToolCallBlock,
)
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._capabilities import (
    LONG_CONTEXT_MODEL_PATTERN,
    get_capabilities,
    get_known_context_window,
    get_request_policy,
)
from amplifier_module_provider_openai._cost import compute_cost
from amplifier_module_provider_openai._model_listing import MODEL_SUPERSESSION
from amplifier_module_provider_openai.native import NativeResponsesProvider
from tests.test_native_transport import Owner, Socket, created, finished

MODELS = ("gpt-6.2-astra", "gpt-6.2-sol")


def request(**kwargs):
    return ChatRequest(messages=[Message(role="user", content="hello")], **kwargs)


def provider(model, **config):
    return OpenAIProvider(api_key="offline", config={
        "default_model": model, "max_retries": 0, "use_streaming": False,
        "base_url": "https://offline.invalid/v1", **config,
    })


@pytest.mark.parametrize("model", MODELS)
def test_exact_facts_registry_and_no_snapshot_invention(model):
    import re

    caps = get_capabilities(model)
    assert caps.family == model
    assert caps.context_window == get_known_context_window(model) == 1_050_000
    assert caps.max_input_tokens == 922_000 and caps.max_output_tokens == 128_000
    assert caps.long_context_pricing_threshold == 272_000
    assert caps.supports_reasoning and caps.supports_vision and caps.supports_streaming
    assert caps.supports_native_apply_patch and caps.supports_native_computer_use
    assert caps.default_reasoning_effort is None  # recommendation is not API default evidence
    policy = get_request_policy(model)
    assert policy.allowed_efforts == frozenset({"low", "medium"})
    assert policy.cache_ttls == {"30m", "12h"}
    assert policy.supports_steering and policy.supports_async_tools
    assert policy.supports_configuration_update
    assert re.search(LONG_CONTEXT_MODEL_PATTERN.removeprefix("matches:"), model)
    assert provider(model).get_info().defaults["context_window"] == 272_000
    assert provider(model, enable_long_context=True).get_info().defaults["context_window"] == 922_000
    for unknown in (model + "-2099-01-01", model + "-preview"):
        assert get_capabilities(unknown).family == "unknown"
        assert get_request_policy(unknown).allowed_efforts is None
        assert get_known_context_window(unknown) is None
        assert compute_cost(unknown, prompt_tokens=1) is None


def native_provider(model, **config):
    owner = Owner()
    base = provider(model, **config)
    p = NativeResponsesProvider.wrap(base, owner_getter=lambda: owner)
    p.socket = Socket()
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    return p, owner


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", [None, "none", "low", "medium"])
@pytest.mark.parametrize("overlay", [False, True])
async def test_actual_http_sdk_wire_policy_and_omission(model, effort, overlay):
    wires = []

    def receive(req):
        assert req.url.path == "/v1/responses"
        wires.append(json.loads(req.content))
        return httpx.Response(200, json={
            "id": "r", "object": "response", "created_at": 0,
            "model": model, "status": "completed", "output": [],
        })

    options = {"model": model, "prompt_cache_options": {"ttl": "12h"}}
    if effort not in (None, "none"):
        options["reasoning"] = {"effort": effort}
    config = {"default_model": "gpt-5.4" if overlay else model,
              "reasoning_effort": effort, "use_streaming": False, "max_retries": 0}
    if overlay:
        config["extra_request_params"] = {"extra_body": options}
    else:
        config["prompt_cache_options"] = options["prompt_cache_options"]
    async with AsyncOpenAI(api_key="offline", base_url="https://offline.invalid/v1",
                           http_client=httpx.AsyncClient(
                               transport=httpx.MockTransport(receive), trust_env=False),
                           max_retries=0) as client:
        p = OpenAIProvider(client=client, config=config)
        await p.complete(request())
        assert len(wires) == 1
        wire = wires[0]
        assert wire["model"] == model
        assert wire["prompt_cache_options"] == {"ttl": "12h"}
        assert "prompt_cache_retention" not in wire
        if effort in (None, "none"):
            assert "reasoning" not in wire
        else:
            assert wire["reasoning"]["effort"] == effort
        assert wire["include"] == ["reasoning.encrypted_content"]
        assert not set(wire) & {"temperature", "top_p", "top_logprobs", "logprobs"}


@pytest.mark.parametrize("layer", ["top", "body"])
def test_overlay_to_old_literal_none_preserves_include_normalization(layer):
    extras = {"extra_body": {
        "model": "gpt-6-sol", "reasoning": {"effort": "none"},
    }}
    target = extras if layer == "top" else extras["extra_body"]
    target["include"] = ["reasoning.encrypted_content", "file_search_call.results"]
    p = provider("gpt-6.2-sol", extra_request_params=extras)
    params = p._budget_params(request())
    effective = {**params, **params["extra_body"]}
    assert effective["include"] == ["file_search_call.results"]
    assert p.extra_request_params == extras  # assembly owns its normalization


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("overlay", [False, True])
@pytest.mark.parametrize("sentinel", [None, "none"])
def test_extended_thinking_implicit_effort_is_documented_without_coercing_explicit(model, overlay, sentinel):
    extras = {"extra_body": {"model": model}} if overlay else {}
    p = provider("gpt-5.4" if overlay else model, extra_request_params=extras,
                 reasoning_effort=sentinel)
    params = p._budget_params(request(), extended_thinking=True)
    assert params["reasoning"]["effort"] == "medium"
    with pytest.raises(llm_errors.InvalidRequestError, match="not established"):
        p._budget_params(request(), extended_thinking=True, reasoning={"effort": "high"})


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("transport", ["http", "native"])
@pytest.mark.parametrize("overlay", [False, True])
@pytest.mark.parametrize("invalid", [
    {"reasoning": {"effort": "none"}},
    {"reasoning": {"effort": "minimal"}},
    {"reasoning": {"effort": "high"}},
    {"reasoning": {"effort": "xhigh"}},
    {"reasoning": {"effort": "max"}},
    {"temperature": 0},
    {"top_p": 0},
    {"top_logprobs": 0},
    {"logprobs": True},
    {"include": ["reasoning.encrypted_content", "message.output_text.logprobs"]},
    {"prompt_cache_options": {"ttl": "24h"}},
])
async def test_final_invalid_fields_never_reach_any_transport(model, transport, overlay, invalid):
    values = {"model": model, **copy.deepcopy(invalid)}
    config = {"extra_request_params": {"extra_body": values} if overlay else values}
    if transport == "native":
        p, _ = native_provider("gpt-5.4" if overlay else model, **config)
    else:
        p = provider("gpt-5.4" if overlay else model, **config)
    p.client.responses.create = AsyncMock()
    with pytest.raises(llm_errors.InvalidRequestError):
        await p.complete(request())
    p.client.responses.create.assert_not_awaited()
    if transport == "native":
        assert p.socket.sent == []


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("long_context,limit", [(False, 267_904), (True, 917_904)])
@pytest.mark.parametrize("output", [1_000, 128_000])
def test_budget_keeps_independent_input_and_reserve(model, long_context, limit, output):
    p = provider(model, enable_long_context=long_context)
    params = p._budget_params(request(max_output_tokens=output))
    assert params["model"] == model
    assert p._budget_input_limit(params) == limit
    assert p.default_model == model


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_real_sdk_count_projection_uses_exact_registry(model):
    wires = []

    def receive(req):
        assert req.url.path == "/v1/responses/input_tokens"
        wires.append(json.loads(req.content))
        return httpx.Response(200, json={"input_tokens": 272_001})

    async with AsyncOpenAI(api_key="offline", base_url="https://api.openai.com/v1",
                           http_client=httpx.AsyncClient(
                               transport=httpx.MockTransport(receive), trust_env=False),
                           max_retries=0) as client:
        p = OpenAIProvider(client=client, config={"default_model": model, "max_retries": 0})
        result = await p.request_budget(request(reasoning_effort="low"), context_estimate=42)
        assert result["input_limit_tokens"] == 267_904
        assert result["context_window_tokens"] == 1_050_000
        assert result["estimated_input_tokens"] == 272_001
        assert len(wires) == 1 and wires[0]["model"] == model
        assert wires[0]["reasoning"]["effort"] == "low"
        assert "prompt_cache_retention" not in wires[0]


@pytest.mark.parametrize("model,scale", [("gpt-6.2-astra", 4), ("gpt-6.2-sol", 1)])
@pytest.mark.parametrize("tokens,expected", [
    (272_000, "0.654"), (272_001, "1.208004"),
])
@pytest.mark.parametrize("tier,multiplier", [("default", 1), ("fast", 2), ("priority", 2)])
def test_whole_request_prices_at_boundary_all_cache_lifetimes(model, scale, tokens, expected, tier, multiplier):
    # Independent Sol oracle: 122k fresh, 100k cached, 20k 30m/30k 12h writes,
    # 20k output. At 272001 re-rate EVERYTHING: 4/.20/5/10/15, not marginal.
    actual = compute_cost(
        model, prompt_tokens=tokens, cached_tokens=100_000,
        cache_write_tokens=50_000, completion_tokens=20_000,
        cache_write_tokens_details={"ttl_30m_tokens": 20_000, "ttl_12h_tokens": 30_000},
        service_tier=tier,
    )
    assert actual == Decimal(expected) * scale * multiplier


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tier", [None, "flex", "ultrafast", "unknown", "batch"])
def test_unknown_or_unimplemented_tier_is_not_standard(model, tier):
    assert compute_cost(model, prompt_tokens=100, service_tier=tier) is None


@pytest.mark.parametrize("model", MODELS)
def test_write_lifetime_evidence_is_required_and_old_rates_retained(model):
    assert compute_cost(model, prompt_tokens=100, cache_write_tokens=20) is None
    assert compute_cost(model, prompt_tokens=100, cache_write_tokens=20,
                        cache_write_tokens_details={"ttl_30m_tokens": 21}) is None
    assert compute_cost("gpt-6-astra", prompt_tokens=100) == Decimal(".001")
    assert compute_cost("gpt-6-sol", prompt_tokens=100) == Decimal(".0002")
    assert compute_cost("gpt-6.1-sol", prompt_tokens=100, service_tier="flex") == Decimal(".0001")


INVENTORY = [*MODEL_SUPERSESSION, *MODELS, "gpt-6.1-terra", "gpt-6-luna",
             "gpt-6-future-preview", "gpt-5.6"]


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", [None, {}, {"include": ["gpt-6*"], "hide_dated": True}])
@pytest.mark.parametrize("configured", [None, {}, {"include": ["gpt-6*"]}])
@pytest.mark.parametrize("direct", [False, True])
async def test_recommendations_are_separate_from_raw_and_custom_inventory(selector, configured, direct):
    p = provider(MODELS[1], base_url=None if direct else "https://gateway.invalid/v1",
                 model_filters=configured)
    p._fetch_model_catalog = AsyncMock(return_value=dict.fromkeys(INVENTORY))
    rows = await p.list_models(filters=selector)
    expected = set(INVENTORY)
    if selector is None and configured is None or selector and selector.get("include") == ["gpt-6*"] or (
        selector is None and configured and configured.get("include") == ["gpt-6*"]
    ):
        expected.remove("gpt-5.6")
    if direct and selector is None and configured is None:
        expected -= MODEL_SUPERSESSION.keys()
    assert {row.id for row in rows} == expected
    readback = p.get_model_list_info()
    assert readback["default_supersession"] == (MODEL_SUPERSESSION if direct and configured is None else {})
    readback["default_supersession"].clear()
    assert p.get_model_list_info()["default_supersession"] == (
        MODEL_SUPERSESSION if direct and configured is None else {})


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [None, *MODELS])
async def test_absent_successor_never_hides_pin_or_synthesizes_replacement(replacement):
    p = provider("gpt-6-astra", base_url=None)
    inventory = list(MODEL_SUPERSESSION) + ([replacement] if replacement else [])
    p._fetch_model_catalog = AsyncMock(return_value=dict.fromkeys(inventory))
    expected = {mid for mid in inventory if MODEL_SUPERSESSION.get(mid) != replacement or replacement is None}
    assert {row.id for row in await p.list_models()} == expected
    assert p.default_model == "gpt-6-astra"
    p._fetch_model_catalog = AsyncMock(return_value={})
    assert await p.list_models() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_model_switch_and_append_only_preserve_reasoning_bytes_and_order(model):
    p = provider("gpt-6-astra", reasoning_replay_scope="all")
    messages = [
        Message(role="user", content="task"),
        Message(role="assistant", content=[
            ThinkingBlock(thinking="opaque", content=[{
                "id": "rs_1", "encrypted_content": "ENC+/= preserved",
                "summary": "opaque",
            }]),
            ToolCallBlock(id="call_1", name="read", input={"path": "a"}),
        ]),
        Message(role="tool", tool_call_id="call_1", content="result"),
    ]
    before = copy.deepcopy([m.model_dump() for m in messages])
    original = p._budget_params(ChatRequest(messages=messages), model="gpt-6-astra")
    switched = p._budget_params(ChatRequest(messages=messages), model=model)
    assert switched["input"] == original["input"]
    assert next(i for i in switched["input"] if i.get("type") == "reasoning")["encrypted_content"] == "ENC+/= preserved"
    appended = p._budget_params(ChatRequest(messages=[
        *messages, Message(role="user", content="continue"),
    ]), model=model)
    assert appended["input"][:len(switched["input"])] == switched["input"]
    assert [m.model_dump() for m in messages] == before


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("overlay", [False, True])
async def test_native_final_wire_and_model_switch_preserve_history(model, overlay):
    p, owner = native_provider("gpt-6-astra")
    socket = p.socket
    # Seed a successful old-model lineage, then switch to either new exact ID.
    for step, selected in enumerate(("gpt-6-astra", model, model)):
        if overlay:
            p.extra_request_params = {"extra_body": {
                "model": selected, "reasoning": {"effort": "low"},
                "prompt_cache_options": {"ttl": "30m"},
            }}
        owner.messages.append({"role": "user", "content": f"append {step}"})
        snapshot = copy.deepcopy(owner.messages)
        socket.feed(created(f"r{step}"), finished(f"r{step}"))
        await p.complete(ChatRequest(messages=[Message(**m) for m in owner.messages]),
                         model=selected, reasoning={"effort": "low"})
        wire = socket.sent[-1]
        assert wire["model"] == selected
        assert wire["reasoning"]["effort"] == "low"
        assert "prompt_cache_retention" not in wire
        assert owner.messages == snapshot
        if step == 1:
            assert "previous_response_id" not in wire  # old connection cache not a transcript
            assert len(wire["input"]) == 2
        if step == 2:
            assert wire["previous_response_id"] == "r1"
    assert len(socket.sent) == 3


@pytest.mark.parametrize("model", MODELS)
def test_configuration_update_uses_bounded_efforts_and_existing_guards(model):
    from amplifier_module_provider_openai.configuration_updates import validate_request

    update = {"type": "configuration_update", "reasoning": {"effort": "medium"}}
    params = {"model": model, "reasoning": {"effort": "low"}, "input": [update]}
    assert validate_request(params) == [update]
    for effort in ("none", "high", "xhigh", "max"):
        with pytest.raises(ValueError, match="not supported"):
            validate_request({**params, "input": [{
                "type": "configuration_update", "reasoning": {"effort": effort},
            }]})
    with pytest.raises(ValueError, match="single-agent"):
        validate_request({**params, "reasoning": {"effort": "low", "mode": "pro"}})