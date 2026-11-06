"""Offline native qualification: source eligibility is not live account access."""

import asyncio
import copy
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.native import (
    NATIVE_REQUEST,
    NativeAsyncTool,
    NativeResponsesProvider,
    NativeSendAdmission,
)
from tests.test_native_transport import created, finished, make, wait_created

MODELS = ("gpt-6-astra", "gpt-6-sol", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna")


def request(model=None):
    return ChatRequest(messages=[Message(role="user", content="fixture")], model=model, max_output_tokens=32)


def public_provider():
    p, owner = make()
    p.owner = None
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    return p, owner


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("source", ["request", "options", "extras", "extra_body"])
async def test_final_effective_model_uses_native_and_counted_plan(model, source):
    p, owner = public_provider()
    req = request(model if source == "request" else "gpt-5.6")
    options = {"request_options": {"model": model}} if source == "options" else {}
    if source == "extras":
        p.extra_request_params["model"] = model
    if source == "extra_body":
        p.extra_request_params["extra_body"] = {"model": model}
    owner.messages = [m.model_dump() for m in req.messages]
    end = finished("one")
    end["response"]["model"] = model
    p.socket.feed(created("one"), end)
    result = await p.complete(req, **options)
    assert p.socket.sent[0]["model"] == model
    assert p.native_status()["model"] == model
    assert result.metadata["converge_live_epoch"] == p.epoch
    counted = p._guard_assembled_params_with_provider_count.call_args_list[-1].args[0]
    assert counted["model"] == model
    assert counted["input"] == p.socket.sent[0]["input"]
    assert p.owner is None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6-unknown", "gpt-6-astra-2099-01-01", "gpt-5.6", "compatible-deployment"])
async def test_unsupported_final_model_falls_back_only_when_settled(model):
    p, _ = public_provider()
    with patch.object(OpenAIProvider, "complete", new_callable=AsyncMock, return_value="ordinary") as ordinary:
        assert await p.complete(request(model)) == "ordinary"
        ordinary.assert_awaited_once()
    assert not p.socket.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending_parent", "request_uncertain", "_native_send_refused", "awaiting_successor"])
async def test_unsupported_model_cannot_escape_pending_or_unknown_state(state):
    p, _ = public_provider()
    setattr(p, state, True)
    with patch.object(OpenAIProvider, "complete", new_callable=AsyncMock) as ordinary:
        with pytest.raises(LLMError, match="explicit recovery"):
            await p.complete(request("gpt-5.6"))
        ordinary.assert_not_awaited()
    assert not p.socket.sent


@pytest.mark.asyncio
async def test_owned_generation_cannot_fall_back_or_steal_owner():
    p, owner = make()
    with patch.object(OpenAIProvider, "complete", new_callable=AsyncMock) as ordinary:
        with pytest.raises(LLMError, match="already owns"):
            await p.complete(request("gpt-5.6"))
        ordinary.assert_not_awaited()
    assert p.owner is owner


@pytest.mark.asyncio
async def test_config_change_during_host_preparation_is_known_unsent():
    p, owner = public_provider()
    def prepare(kind, *, input_id):
        p.config["reasoning_effort"] = "changed-after-plan"
        return NativeSendAdmission("attempt", lambda envelope: True)
    p.prepare_send = prepare
    with pytest.raises(LLMError):
        await p.complete(request())
    assert not p.socket.sent
    assert p._native_send_refused
    assert any(e["type"] == "native.not_sent" for e in owner.events)


def billed(identity, model="gpt-6-astra", tier="default", tokens=200_000, **details):
    end = finished(identity)
    end["response"].update(model=model, service_tier=tier, status="completed")
    end["response"]["usage"] = {
        "input_tokens": tokens, "output_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, **details},
    }
    return end


async def two_terminals(p, first, second):
    p.socket.feed(created("parent"))
    task = asyncio.create_task(p._native_response({"model": "gpt-6-astra", "input": []}))
    await wait_created(p)
    await p.steer_live(SimpleNamespace(id="direction", text="update"))
    first = copy.deepcopy(first)
    first["type"] = "response.incomplete"
    first["response"]["incomplete_details"] = {"reason": "steered"}
    p.socket.feed(
        {"type": "response.steer.accepted", "steer": {"id": "steer"}},
        first, created("next"), second,
    )
    return await task


@pytest.mark.asyncio
async def test_each_terminal_priced_before_long_context_threshold():
    p, _ = make()
    raw = await two_terminals(p, billed("parent"), billed("next"))
    assert raw.usage.input_tokens == 400_000
    assert p._compute_attempt_cost(raw) == Decimal(4)
    assert len(raw.native_terminals) == 2
    # A naive merged 400K Astra request costs twice the actual two 200K sends.
    assert p._terminal_cost(raw) == Decimal(8)


@pytest.mark.asyncio
async def test_terminal_actual_models_tiers_and_terra_ttls_preserved():
    p, _ = make()
    first = billed("parent", tier="priority", tokens=10)
    second = billed(
        "next", model="gpt-6.1-terra", tokens=100,
        cached_tokens=10, cache_write_tokens=30,
        cache_write_tokens_details={"ttl_30m_tokens": 10, "ttl_12h_tokens": 20},
    )
    raw = await two_terminals(p, first, second)
    expected = sum(p._terminal_cost(type(raw).__bases__[0](x["response"])) for x in (first, second))
    assert p._compute_attempt_cost(raw) == expected
    assert raw.native_terminals[0]["service_tier"] == "priority"
    assert raw.native_terminals[1]["model"] == "gpt-6.1-terra"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["usage", "model", "output_tokens", "terra_cache", "terra_ttl"])
async def test_any_unknown_terminal_keeps_aggregate_cost_unknown(missing):
    p, _ = make()
    end = billed("parent")
    if missing in {"usage", "model"}:
        end["response"].pop(missing)
    elif missing == "output_tokens":
        end["response"]["usage"].pop("output_tokens")
    else:
        end["response"]["model"] = "gpt-6.1-terra"
        if missing == "terra_cache":
            end["response"]["usage"]["input_tokens_details"].pop("cache_write_tokens")
        else:
            end["response"]["usage"]["input_tokens_details"]["cache_write_tokens"] = 20
    raw = await two_terminals(p, end, billed("next"))
    assert p._compute_attempt_cost(raw) is None


@pytest.mark.asyncio
async def test_endpoint_pricing_hook_sees_each_actual_terminal_once_not_merged():
    endpoint = OpenAIProvider(api_key="fixture")
    observed = []
    def price(raw):
        observed.append((raw.model, raw.service_tier, raw.usage.input_tokens))
        return Decimal(7)
    endpoint._compute_attempt_cost = price
    p, owner = make()
    wrapped = NativeResponsesProvider.wrap(endpoint, owner_getter=lambda: owner)
    wrapped.owner, wrapped.socket = owner, p.socket
    raw = await two_terminals(wrapped, billed("parent"), billed("next", model="gpt-6-sol", tier="flex"))
    assert wrapped._compute_attempt_cost(raw) == Decimal(14)
    assert wrapped._compute_attempt_cost(raw) == Decimal(14)  # no repricing
    assert observed == [("gpt-6-astra", "default", 200_000), ("gpt-6-sol", "flex", 200_000)]


@pytest.mark.asyncio
async def test_public_complete_publishes_per_terminal_cost_without_double_count():
    p, _ = public_provider()
    p.socket.feed(created("parent"))
    task = asyncio.create_task(p.complete(request()))
    await wait_created(p)
    await p.steer_live(SimpleNamespace(id="direction", text="update"))
    first, second = billed("parent"), billed("next")
    first["type"] = "response.incomplete"
    first["response"]["incomplete_details"] = {"reason": "steered"}
    p.socket.feed(
        {"type": "response.steer.accepted", "steer": {"id": "steer"}},
        first, created("next"), second,
    )
    result = await task
    assert result.usage.cost_usd == Decimal(4)
    assert len(p.socket.sent) == 2  # one create, one steer, no extra generation


@pytest.mark.asyncio
async def test_optional_progress_has_successor_safe_identity_and_no_final_delta_replay():
    p, _ = make()
    p.progressive_output = True
    hook = AsyncMock()
    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=hook))
    def events():
        return [
            {"type": "response.output_item.added", "output_index": 0, "sequence_number": 1,
             "item": {"type": "message", "id": "text"}},
            {"type": "response.output_text.delta", "output_index": 0, "sequence_number": 2, "delta": "hello"},
            {"type": "response.output_text.delta", "output_index": 0, "sequence_number": 2, "delta": "hello"},
            {"type": "response.output_item.done", "output_index": 0, "sequence_number": 3,
             "item": {"type": "message", "id": "text"}},
        ]
    p.socket.feed(created("parent"), *events())
    task = asyncio.create_task(p._native_response({"model": "gpt-6-astra", "input": []}))
    await wait_created(p)
    await p.steer_live(SimpleNamespace(id="direction", text="update"))
    p.socket.feed(
        {"type": "response.steer.accepted", "steer": {"id": "steer"}},
        billed("parent"), created("next"), *events(), billed("next"),
    )
    await task
    deltas = [c.args[1] for c in hook.await_args_list if c.args[0] == "llm:stream_block_delta"]
    assert [d["text"] for d in deltas] == ["hello", "hello"]
    assert [d["block_index"] for d in deltas] == [0, 1]
    assert len({d["request_id"] for d in deltas}) == 1
    assert [d["native_response_id"] for d in deltas] == ["parent", "next"]


@pytest.mark.asyncio
async def test_progress_is_opt_in_and_callback_failure_does_not_replay():
    p, _ = make()
    hook = AsyncMock(side_effect=OSError("sink failed"))
    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=hook))
    p.socket.feed(created("p"), {"type": "response.output_text.delta", "output_index": 0, "delta": "x"}, billed("p"))
    await p._native_response({"model": "gpt-6-astra", "input": []})
    hook.assert_not_awaited()
    p.progressive_output = True
    socket = p.socket
    socket.feed(created("q"), {"type": "response.output_text.delta", "output_index": 0, "delta": "x"})
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    token = NATIVE_REQUEST.set(p)
    try:
        with pytest.raises(LLMError):
            await p._create_response({"model": "gpt-6-astra", "input": []})
    finally:
        NATIVE_REQUEST.reset(token)
    assert p.request_uncertain and socket.closed and len(socket.sent) == 2


def async_call(identity, *, name="lookup", kind="function", namespace=None):
    return {
        "type": "function_call" if kind == "function" else "custom_tool_call",
        "call_id": identity, "name": name, "async": True, "status": "completed",
        **({"arguments": '{"task_handle":"handle-' + identity + '"}'} if kind == "function" else {"input": "opaque custom text"}),
        **({"namespace": namespace} if namespace else {}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["function", "custom"])
async def test_explicit_async_handler_gets_complete_original_item_once(kind):
    p, owner = make()
    start = AsyncMock(return_value={"status": "queued"})  # not a tool result
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", start, kind=kind)]
    item = async_call("original", kind=kind)
    p.socket.feed(
        created("p"),
        {"type": "response.output_item.done", "item": item},
        {"type": "response.output_item.done", "item": item},
        finished("p"),
    )
    await p._native_response({"model": "gpt-6-astra", "input": [], "tools": [{"type": kind, "name": "lookup"}]})
    start.assert_awaited_once_with(item)
    assert p.socket.sent[0]["tools"][0]["async"] is True
    assert p.async_calls["original"] == item
    assert not p.delivered_results  # acknowledging job launch is NOT output


@pytest.mark.asyncio
async def test_async_namespaces_exact_and_wait_tool_stays_synchronous():
    p, owner = make()
    start = AsyncMock()
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", start, namespace="approved")]
    p.socket.feed(created("p"), finished("p"))
    await p._native_response({
        "model": "gpt-6-astra", "input": [], "tools": [
            {"type": "namespace", "name": "approved", "tools": [{"type": "function", "name": "lookup"}]},
            {"type": "namespace", "name": "other", "tools": [{"type": "function", "name": "lookup", "async": True}]},
            {"type": "function", "name": "wait_for_tasks", "async": True},
        ],
    })
    approved, other, wait = p.socket.sent[0]["tools"]
    assert approved["tools"][0]["async"]
    assert "async" not in other["tools"][0] and "async" not in wait


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["approval_denied", "budget_exhausted", "stop", "unknown"])
async def test_host_denial_or_unknown_dispatch_never_restarts_original_call(reason):
    p, owner = make()
    start = AsyncMock(side_effect=RuntimeError(reason))
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", start)]
    item = async_call("same")
    p.socket.feed(created("p"), {"type": "response.output_item.done", "item": item})
    with pytest.raises(RuntimeError):
        await p._native_response({"model": "gpt-6-astra", "input": [], "tools": [{"type": "function", "name": "lookup"}]})
    assert p.async_calls["same"] == item and p.request_uncertain
    with pytest.raises(LLMError, match="uncertain"):
        await p._renew_idle_connection()
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_cancellation_is_unknown_no_automatic_replay():
    p, owner = make()
    start = AsyncMock(side_effect=asyncio.CancelledError())
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", start)]
    item = async_call("same")
    p.socket.feed(created("p"), {"type": "response.output_item.done", "item": item})
    with pytest.raises(asyncio.CancelledError):
        await p._native_response({"model": "gpt-6-astra", "input": [], "tools": [{"type": "function", "name": "lookup"}]})
    assert p.request_uncertain and "same" in p.async_calls
    start.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["computer", "unqualified", "incomplete", "bad_json", "identity_reuse"])
async def test_async_negative_controls_do_not_dispatch(bad):
    p, owner = make()
    start = AsyncMock()
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", start)]
    item = async_call("one")
    if bad in {"computer", "unqualified"}:
        item["name"] = bad
    elif bad == "incomplete":
        item["status"] = "in_progress"
    elif bad == "bad_json":
        item["arguments"] = "{"
    else:
        p.async_calls["one"] = {**item, "arguments": '{"different":true}'}
    p.socket.feed(created("p"), {"type": "response.output_item.done", "item": item})
    with pytest.raises((ValueError, LLMError)):
        await p._native_response({"model": "gpt-6-astra", "input": [], "tools": [{"type": "function", "name": "lookup"}]})
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_out_of_order_jobs_intervening_user_and_restart_keep_original_bindings():
    p, owner = make()
    calls = [async_call("first"), async_call("second", kind="custom")]
    p.async_calls = {c["call_id"]: c for c in calls}
    jobs = {"first": {"result": None}, "second": {"result": "real second"}}
    owner.native_job = jobs.get
    p.previous_response_id = "latest-after-user-turn"
    messages = [
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "jobs running", "metadata": {"converge_native_async_calls": calls}},
        {"role": "user", "content": "new direction"},
    ]
    p.full_messages = copy.deepcopy(messages)
    token = NATIVE_REQUEST.set(p)
    try:
        wire = p._convert_messages(messages)
        outputs = [x for x in wire if x.get("type", "").endswith("_call_output")]
        assert outputs == [{"type": "custom_tool_call_output", "call_id": "second", "output": "real second"}]
        assert not any(x.get("call_id") == "first" and "output" in x for x in wire)
        assert p._convert_messages(messages) == []  # results delivered once on chain
        # A restored unknown job is not replayed or given a synthetic result.
        jobs["first"] = {"restored": True, "recovered": False, "result": "unqualified stale"}
        p.previous_response_id = None
        wire = p._convert_messages(messages)
        assert not any(x.get("call_id") == "first" and "output" in x for x in wire)
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
async def test_legacy_delegate_remains_supported_only_with_real_handler():
    p, owner = make()
    owner.config["background_delegate"] = True
    p.socket.feed(created("p"), finished("p"))
    await p._native_response({"model": "gpt-6-astra", "input": [], "tools": [{"type": "function", "name": "delegate"}]})
    assert "async" not in p.socket.sent[0]["tools"][0]
    owner.start_native_job = AsyncMock()
    item = async_call("delegate-1", name="delegate")
    p.socket.feed(created("q"), {"type": "response.output_item.done", "item": item}, finished("q"))
    await p._native_response({"model": "gpt-6-astra", "input": [], "tools": [{"type": "function", "name": "delegate"}]})
    assert owner.start_native_job.await_args.args[0].id == "delegate-1"


@pytest.mark.asyncio
async def test_ordinary_fallback_retains_base_missing_result_repair_behavior():
    p, _ = public_provider()
    req = ChatRequest(model="gpt-5.6", messages=[
        Message(role="user", content="lookup"),
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "interrupted", "name": "read", "input": {},
        }]),
    ])
    with patch.object(OpenAIProvider, "complete", new_callable=AsyncMock, return_value="ordinary") as ordinary:
        assert await p.complete(req) == "ordinary"
        ordinary.assert_awaited_once()
    assert not p.socket.sent


@pytest.mark.asyncio
async def test_budget_and_completion_overlay_model_parity_without_checkpoint():
    p, _ = public_provider()
    p.extra_request_params["extra_body"] = {"model": "gpt-6.1-terra"}
    p._provider_count_available = lambda: True
    p._native_input_token_count = AsyncMock(return_value=1)
    budget = await p.request_budget(request("gpt-4.1"), context_estimate=1)
    counted = p._native_input_token_count.call_args.args[0]
    assert counted["model"] == "gpt-6.1-terra" and "extra_body" not in counted
    assert budget["input_limit_tokens"] == p._budget_input_limit(counted)
    before = copy.deepcopy(counted)
    p.socket.feed(created("one"), billed("one", model="gpt-6.1-terra"))
    await p.complete(request("gpt-4.1"))
    assert p._guard_assembled_params_with_provider_count.call_args.args[0] == before


@pytest.mark.asyncio
async def test_large_deferred_output_is_recounted_before_public_dispatch():
    p, owner = public_provider()
    item = async_call("original", kind="custom")
    p.async_calls["original"] = item
    owner.native_job = lambda identity: {"result": "R" * 12000}
    req = ChatRequest(messages=[
        Message(role="user", content="original"),
        Message(role="assistant", content="running", metadata={"converge_native_async_calls": [item]}),
        Message(role="user", content="later direction"),
    ], max_output_tokens=32)
    owner.messages = [m.model_dump() for m in req.messages]
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=1)
    p.socket.feed(created("one"), billed("one"))
    await p.complete(req)
    counted = p._guard_assembled_params_with_provider_count.await_args_list
    assert len(counted) == 2
    first, final = counted[0].args[0], counted[1].args[0]
    assert not any(x.get("call_id") == "original" for x in first["input"])
    results = [x for x in final["input"] if x.get("type") == "custom_tool_call_output"]
    assert results == [{"type": "custom_tool_call_output", "call_id": "original", "output": "R" * 12000}]
    assert any(x == results[0] for x in p.socket.sent[0]["input"])


@pytest.mark.asyncio
async def test_canonical_result_delivered_without_live_job_after_epoch_skip():
    p, owner = make()
    item = async_call("original", kind="custom")
    p.async_calls["original"] = item
    owner.native_job = lambda identity: None
    p.previous_response_id = "current"
    messages = [
        {"role": "user", "content": "original"},
        {"role": "assistant", "content": "running", "metadata": {
            "converge_native_async_calls": [item], "converge_live_epoch": p.epoch,
        }},
        {"role": "tool", "tool_call_id": "original", "content": "actual saved result"},
        {"role": "user", "content": "later"},
    ]
    token = NATIVE_REQUEST.set(p)
    try:
        wire = p._convert_messages(messages)
        assert {"type": "custom_tool_call_output", "call_id": "original", "output": "actual saved result"} in wire
        assert p._convert_messages(messages) == []
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
async def test_pending_model_change_refuses_before_send():
    p, _ = public_provider()
    p._last_native_model = "gpt-6-astra"
    p.pending_parent = True
    p.pending_steer = {"command": SimpleNamespace(id="steer"), "required_input": []}
    with pytest.raises(LLMError, match="cannot change model"):
        await p.complete(request("gpt-6.1-terra"))
    assert not p.socket.sent and p.owner is None and p.pending_parent


@pytest.mark.asyncio
async def test_settled_model_change_resets_lineage_and_checkpoint():
    p, _ = public_provider()
    p._last_native_model = "gpt-6-astra"
    p.previous_response_id = "old"
    p.socket.feed(created("new"), billed("new", model="gpt-6-sol"))
    await p.complete(request("gpt-6-sol"))
    assert "previous_response_id" not in p.socket.sent[0]
    assert p._checkpoint_status == "discarded:model_changed"


@pytest.mark.asyncio
async def test_ready_async_results_precede_synchronous_wait_status():
    p, owner = make()
    item = async_call("original")
    p.async_calls["original"] = item
    p.previous_response_id = "latest"
    owner.native_job = lambda identity: {"result": "real lookup"}
    messages = [
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "running", "metadata": {
            "converge_native_async_calls": [item], "converge_live_epoch": p.epoch,
        }},
        {"role": "assistant", "content": [{
            "type": "tool_call", "id": "wait", "name": "wait_for_tasks",
            "input": {"task_handles": ["handle-original"]},
        }]},
        {"role": "tool", "tool_call_id": "wait", "content": "completed status only"},
    ]
    token = NATIVE_REQUEST.set(p)
    try:
        wire = p._convert_messages(messages)
    finally:
        NATIVE_REQUEST.reset(token)
    assert [x["call_id"] for x in wire if x.get("type") == "function_call_output"] == ["original", "wait"]


@pytest.mark.asyncio
async def test_async_multiagent_parallel_combination_rejected_before_send():
    p, owner = make()
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", AsyncMock())]
    with pytest.raises(ValueError, match="multi-agent parallel"):
        await p._native_response({
            "model": "gpt-6.1-sol", "input": [], "multi_agent": {"enabled": True},
            "tools": [{"type": "function", "name": "lookup"}],
        })
    assert not p.socket.sent


@pytest.mark.asyncio
async def test_general_async_public_completion_retains_custom_call_metadata():
    p, owner = public_provider()
    start = AsyncMock(return_value="queued acknowledgement")
    owner.native_async_tools = lambda: [NativeAsyncTool("lookup", start, kind="custom")]
    # Existing raw-tool escape hatch preserves custom definitions; ordinary
    # ToolSpec conversion currently models JSON-schema function tools only.
    p.extra_request_params["tools"] = [{"type": "custom", "name": "lookup"}]
    item = async_call("custom-original", kind="custom")
    p.socket.feed(
        created("one"), {"type": "response.output_item.done", "item": item},
        billed("one"),
    )
    result = await p.complete(request())
    assert result.metadata["converge_native_async_calls"] == [item]
    assert not p.delivered_results
    start.assert_awaited_once_with(item)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [False, True])
async def test_checkpoint_never_erases_ready_async_output_from_wire(mismatch):
    from amplifier_module_provider_openai.native_checkpoint import binding, digest

    p, owner = public_provider()
    initial = request()
    params, _, _ = OpenAIProvider._assemble_initial_responses_params(p, initial)
    canonical = [m.model_dump() for m in initial.messages]
    p._checkpoint = {
        "sourceCount": len(canonical), "sourceRevision": digest(canonical),
        "wireCount": len(params["input"]), "wireRevision": digest(params["input"]),
        "configDigest": p._config_digest(),
        "requestBinding": "mismatch" if mismatch else binding(params),
        "output": [{"type": "compaction", "encrypted_content": "OPAQUE"}],
    }
    item = async_call("saved", kind="custom")
    p.async_calls["saved"] = item
    owner.native_job = lambda identity: {"result": "REAL READY OUTPUT"}
    req = ChatRequest(messages=[
        *initial.messages,
        Message(role="assistant", content="running", metadata={"converge_native_async_calls": [item]}),
        Message(role="user", content="later"),
    ], max_output_tokens=32)
    owner.messages = [m.model_dump() for m in req.messages]
    p.socket.feed(created("next"), billed("next"))
    await p.complete(req)
    expected = {"type": "custom_tool_call_output", "call_id": "saved", "output": "REAL READY OUTPUT"}
    assert expected in p.socket.sent[0]["input"]
    assert expected in p._guard_assembled_params_with_provider_count.call_args.args[0]["input"]
    assert p._checkpoint is None and "saved" in p.delivered_results


@pytest.mark.asyncio
async def test_job_result_snapshot_cannot_change_between_count_and_wire():
    p, owner = public_provider()
    item = async_call("saved", kind="custom")
    p.async_calls["saved"] = item
    job = {"result": "result observed before count"}
    owner.native_job = lambda identity: job
    req = request()
    owner.messages = [m.model_dump() for m in req.messages]
    counts = []
    async def count(params):
        counts.append(copy.deepcopy(params))
        if len(counts) == 2:
            job["result"] = "different after count"
        return 1
    p._guard_assembled_params_with_provider_count = count
    p.socket.feed(created("next"), billed("next"))
    await p.complete(req)
    outputs = [x for x in p.socket.sent[0]["input"] if x.get("type") == "custom_tool_call_output"]
    assert outputs[0]["output"] == "result observed before count"
    assert outputs[0] in counts[-1]["input"]


@pytest.mark.asyncio
async def test_async_wait_order_never_crosses_original_call_after_earlier_sync_pair():
    p, owner = public_provider()
    item = async_call("async-later")
    p.async_calls[item["call_id"]] = item
    owner.native_job = lambda identity: {"result": "real later output"}
    req = ChatRequest(messages=[
        Message(role="user", content="initial"),
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "read-old", "name": "read", "input": {},
        }]),
        Message(role="tool", tool_call_id="read-old", content="old synchronous result"),
        Message(role="user", content="next"),
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "async-later", "name": "lookup", "input": {},
        }], metadata={"converge_native_async_calls": [item]}),
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "wait", "name": "wait_for_tasks",
            "input": {"task_handles": ["handle-async-later"]},
        }]),
        Message(role="tool", tool_call_id="wait", content="completed status only"),
    ], max_output_tokens=32)
    owner.messages = [m.model_dump() for m in req.messages]
    p.socket.feed(created("next"), billed("next"))
    await p.complete(req)
    wire = p.socket.sent[0]["input"]
    def position(kind, identity):
        return next(i for i, value in enumerate(wire) if value.get("type") == kind and value.get("call_id") == identity)
    assert (
        position("function_call", "read-old")
        < position("function_call_output", "read-old")
        < position("function_call", "async-later")
        < position("function_call_output", "async-later")
        < position("function_call_output", "wait")
    )