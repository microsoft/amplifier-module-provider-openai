"""Synthetic, offline tests at the actual native send and early-dispatch boundary."""

import asyncio
import copy
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider, native
from tests.test_native_transport import Owner, Socket, created, finished, wait_created


class WireSocket(Socket):
    def __init__(self):
        super().__init__()
        self.wire = []
        self.failure = None

    async def send(self, value):
        self.wire.append(value)
        if self.failure:
            raise self.failure
        await super().send(value)


def make(prepare_send):
    owner = Owner()
    provider = native.NativeResponsesProvider.wrap(
        OpenAIProvider(
            api_key="fixture", config={"default_model": "gpt-6-astra", "max_retries": 0}
        ),
        owner_getter=lambda: owner,
        prepare_send=prepare_send,
    )
    provider.owner = owner
    provider.socket = WireSocket()
    provider._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    return provider, owner


def admission(records, *, attempt_id="attempt-1", decision=True):
    def admit(envelope):
        records.append(envelope)
        return decision

    return native.NativeSendAdmission(attempt_id=attempt_id, admit=admit)


@pytest.mark.asyncio
async def test_final_bytes_follow_awaited_preparation_and_cannot_be_mutated():
    records, order, held = [], [], {}

    async def prepare(kind, *, input_id):
        assert (kind, input_id) == ("response.create", None)
        order.append("prepare")
        await asyncio.sleep(0)
        held["payload"]["input"][0]["content"] = "after preparation"

        def admit(envelope):
            order.append("admit")
            records.append(envelope)
            assert "after preparation" in envelope.payload
            with pytest.raises(FrozenInstanceError):
                envelope.payload = "changed"
            held["payload"]["input"][0]["content"] = "after freeze"
            asyncio.get_running_loop().call_soon(order.append, "later callback")
            return True

        return native.NativeSendAdmission(attempt_id="attempt-1", admit=admit)

    p, _ = make(prepare)
    socket = p.socket
    original_send = socket.send

    async def send(value):
        order.append("send")
        assert order[-2] == "admit"  # no scheduling gap after admission
        assert value is records[-1].payload
        await original_send(value)

    socket.send = send
    p.trace_context = lambda payload: held.update(payload=payload)

    async def lifecycle(kind, data):
        if kind == "native.submitting":
            order.append("lifecycle")
            await asyncio.sleep(0)
            held["payload"]["input"][0]["content"] = "after lifecycle"

    p.lifecycle = lifecycle
    socket.feed(created("r1"), finished("r1"))
    await p._native_response({"input": [{"role": "user", "content": "original"}]})
    assert order[:4] == ["lifecycle", "prepare", "admit", "send"]
    assert socket.wire == [records[0].payload]
    assert records[0].attempt_id == "attempt-1"
    assert records[0].kind == "response.create" and records[0].input_id is None
    assert json.loads(socket.wire[0])["input"][0]["content"] == "after preparation"
    assert "after preparation" not in repr(records[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "steer"])
@pytest.mark.parametrize("decision", [False, None, 1, "true"])
async def test_only_literal_true_admits_and_refusal_never_sends(operation, decision):
    records = []
    p, o = make(lambda *args, **kwargs: admission(records, decision=decision))
    socket = p.socket
    if operation == "steer":
        p.response_id = "parent"
        p.request_uncertain = True  # an independently running create
        with pytest.raises(LLMError) as error:
            await p.steer_live(SimpleNamespace(id="local-1", text="direction"))
        assert p.inflight is None and not o.pending
        assert not socket.closed  # no uncertainty invented for an unsent steer
    else:
        token = native.NATIVE_REQUEST.set(p)
        try:
            with pytest.raises(LLMError) as error:
                await p._create_response({"input": []})
        finally:
            native.NATIVE_REQUEST.reset(token)
        assert not p.request_uncertain
    assert error.value.retryable is False
    assert not socket.wire and len(records) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "steer"])
@pytest.mark.parametrize("invalid", ["missing", "identity", "async-admit", "prepare-raises", "admit-raises"])
async def test_malformed_admission_or_callback_exception_fails_closed(operation, invalid):
    async def async_admit(envelope):
        raise AssertionError("an async admission must not run")

    def raises(*args):
        raise ValueError("synthetic callback failure")

    def prepare(*args, **kwargs):
        if invalid == "missing":
            return None
        if invalid == "prepare-raises":
            raises()
        return native.NativeSendAdmission(
            attempt_id="" if invalid == "identity" else "attempt-1",
            admit=async_admit if invalid == "async-admit" else (
                raises if invalid == "admit-raises" else lambda e: True
            ),
        )

    p, o = make(prepare)
    socket = p.socket
    token = native.NATIVE_REQUEST.set(p)
    try:
        if operation == "steer":
            p.response_id = "parent"
            call = p.steer_live(SimpleNamespace(id="local-1", text="direction"))
        else:
            call = p._create_response({"input": []})
        with pytest.raises(LLMError) as error:
            await call
    finally:
        native.NATIVE_REQUEST.reset(token)
    assert error.value.retryable is False
    assert not socket.wire and not o.pending and p.inflight is None


@pytest.mark.asyncio
async def test_final_admission_sees_revocation_during_async_preparation():
    current = [True]
    observed = []

    async def prepare(*args, **kwargs):
        asyncio.get_running_loop().call_soon(current.__setitem__, 0, False)
        await asyncio.sleep(0)

        def admit(envelope):
            observed.append(envelope)
            return current[0]

        return native.NativeSendAdmission(attempt_id="attempt-1", admit=admit)

    p, _ = make(prepare)
    with pytest.raises(LLMError):
        await p._native_response({"input": []})
    assert len(observed) == 1 and not p.socket.wire and not p.request_uncertain


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "steer"])
async def test_cancel_during_preparation_never_sends(operation):
    entered = asyncio.Event()

    async def prepare(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    p, _ = make(prepare)
    socket = p.socket
    if operation == "steer":
        p.response_id = "parent"
        p.request_uncertain = True
        call = p.steer_live(SimpleNamespace(id="local-1", text="direction"))
    else:
        call = p._native_response({"input": []})
    task = asyncio.create_task(call)
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not socket.wire and not socket.closed and p.inflight is None
    assert p.request_uncertain is (operation == "steer")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "steer"])
@pytest.mark.parametrize("failure", [ConnectionError("synthetic"), asyncio.CancelledError()])
async def test_send_failure_or_cancel_keeps_exact_observation_unknown_and_no_replay(operation, failure):
    records = []
    p, o = make(lambda *args, **kwargs: admission(records))
    socket = p.socket
    socket.failure = failure
    token = native.NATIVE_REQUEST.set(p)
    try:
        if operation == "steer":
            p.response_id = "parent"
            p.request_uncertain = True
            call = p.steer_live(SimpleNamespace(id="local-1", text="direction"))
        else:
            call = p._create_response({"input": []})
        with pytest.raises((LLMError, ConnectionError, asyncio.CancelledError)):
            await call
        with pytest.raises(LLMError, match="uncertain"):
            await p._renew_idle_connection()
    finally:
        native.NATIVE_REQUEST.reset(token)
    assert socket.wire == [records[0].payload]
    assert socket.closed and p.request_uncertain
    assert any(e["type"] == "native.outcome_unknown" for e in o.events)
    assert not o.pending


@pytest.mark.asyncio
async def test_steer_parent_changes_during_preparation_refuses_stale_send():
    records = []

    async def prepare(*args, **kwargs):
        await asyncio.sleep(0)
        p.response_id = None  # receive loop has completed the parent
        return admission(records)

    p, _ = make(prepare)
    p.response_id = "parent"
    with pytest.raises(LLMError):
        await p.steer_live(SimpleNamespace(id="local-1", text="direction"))
    assert not p.socket.wire and not records and p.inflight is None


@pytest.mark.asyncio
async def test_real_complete_observes_delta_not_full_preflight_input():
    records = []
    p, o = make(lambda *args, **kwargs: admission(records))
    p.owner = None
    request = ChatRequest(messages=[Message(role="user", content="one")], max_output_tokens=32)
    o.messages = [m.model_dump() for m in request.messages]
    for identity in ("first", "second"):
        end = finished(identity)
        end["response"].update(model="gpt-6-astra", status="completed")
        p.socket.feed(created(identity), end)
        await p.complete(request)
    assert [e.payload for e in records] == p.socket.wire
    assert json.loads(records[0].payload)["input"]
    assert json.loads(records[1].payload)["input"] == []
    assert json.loads(records[1].payload)["previous_response_id"] == "first"


@pytest.mark.asyncio
async def test_real_complete_observes_entire_checkpoint_window_and_suffix():
    from tests.test_native_checkpoint import make as checkpoint_fixture

    p, messages, opaque, _ = checkpoint_fixture()
    await p.native_compact(canonical=messages, identity={"instance": "one"})
    records = []
    p.prepare_send = lambda *args, **kwargs: admission(records)
    o = Owner()
    o.messages = messages + [Message(role="user", content="suffix").model_dump()]
    p.owner_getter = lambda: o
    p.socket = WireSocket()
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    end = finished("checkpoint-response")
    end["response"].update(model="gpt-6-astra", status="completed")
    p.socket.feed(created("checkpoint-response"), end)
    request = ChatRequest(messages=[Message(**m) for m in o.messages], max_output_tokens=32)
    await p.complete(request)
    assert p.socket.wire == [records[0].payload]
    sent = json.loads(records[0].payload)
    assert sent["input"][:len(opaque)] == opaque
    assert sent["input"][-1]["content"] == [{"type": "input_text", "text": "suffix"}]
    assert "previous_response_id" not in sent


@pytest.mark.asyncio
async def test_receive_trace_binds_each_early_tool_before_dispatch_not_final_aggregate():
    records, origins, dispatched, active = [], {}, [], [None]
    p, o = make(lambda *args, **kwargs: admission(records))
    o.config["background_delegate"] = True

    def trace(kind, event):
        if kind != "native.receive":
            return
        if event["type"] == "response.created":
            active[0] = event["response"]["id"]
        if event["type"] == "response.output_item.done":
            item = event["item"]
            origins[item["call_id"]] = (active[0], copy.deepcopy(item))

    async def start(call):
        assert call.id in origins  # before execution, not terminal aggregation
        dispatched.append((call.id, origins[call.id][0], call.arguments))

    p.trace = trace
    o.start_native_job = start
    socket = p.socket
    socket.feed(created("r1"))
    task = asyncio.create_task(p._native_response({
        "input": [], "tools": [{"type": "function", "name": "delegate", "parameters": {}}]
    }))
    try:
        await wait_created(p)
        for index in (1, 2):
            item = {"type": "function_call", "id": f"item-{index}",
                    "call_id": f"call-{index}", "name": "delegate",
                    "arguments": json.dumps({"task": f"synthetic-{index}"}), "async": True}
            socket.feed({"type": "response.output_item.done", "item": item})
            await asyncio.sleep(0)
            command = SimpleNamespace(id=f"local-{index}", text=f"direction-{index}")
            assert await p.steer_live(command)
            assert records[-1].input_id == command.id
            socket.feed({"type": "response.steer.accepted", "steer": {"id": f"s{index}"}},
                        finished(f"r{index}", incomplete=True), created(f"r{index+1}"))
            for _ in range(100):
                if p.response_id == f"r{index+1}":
                    break
                await asyncio.sleep(0)
            assert p.response_id == f"r{index+1}"
        socket.feed(finished("r3"))
        result = await asyncio.wait_for(task, 2)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert result.id == "r3"
    assert [(c, r) for c, r, _ in dispatched] == [("call-1", "r1"), ("call-2", "r2")]
    assert socket.wire == [e.payload for e in records]
    assert [e.kind for e in records] == ["response.create", "response.steer", "response.steer"]
    assert [json.loads(e.payload)["previous_response_id"] for e in records[1:]] == ["r1", "r2"]
    assert [(e["input_id"], e["response_id"]) for e in o.events
            if e["type"] == "steering.applied"] == [("local-1", "r2"), ("local-2", "r3")]


@pytest.mark.asyncio
async def test_unsent_create_cannot_silently_reuse_advanced_delta_state():
    records = []
    p, o = make(lambda *args, **kwargs: admission(records, decision=False))
    p.owner = None
    request = ChatRequest(messages=[Message(role="user", content="one")], max_output_tokens=32)
    o.messages = [m.model_dump() for m in request.messages]
    with pytest.raises(LLMError):
        await p.complete(request)
    p.prepare_send = lambda *args, **kwargs: admission(records)
    with pytest.raises(LLMError, match="not sent.*recovery"):
        await p.complete(request)
    assert not p.socket.wire and len(records) == 1
    assert not p.request_uncertain  # known unsent, not a fabricated unknown send


@pytest.mark.asyncio
async def test_awaiting_steer_preparation_does_not_hold_completed_parent_open():
    entered, release = asyncio.Event(), asyncio.Event()
    records = []

    async def prepare(kind, *, input_id):
        if kind == "response.steer":
            entered.set()
            await release.wait()
        return admission(records)

    p, _ = make(prepare)
    socket = p.socket
    socket.feed(created("parent"))
    parent = asyncio.create_task(p._native_response({"input": []}))
    await wait_created(p)
    steer = asyncio.create_task(p.steer_live(SimpleNamespace(id="local-1", text="later")))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not await p.steer_live(SimpleNamespace(id="local-2", text="duplicate"))
        socket.feed(finished("parent"))
        await asyncio.wait_for(parent, 2)  # no unsent steer is awaited by recv
        release.set()
        with pytest.raises(LLMError):
            await asyncio.wait_for(steer, 2)
    finally:
        for task in (parent, steer):
            if not task.done():
                task.cancel()
        await asyncio.gather(parent, steer, return_exceptions=True)
    assert len(socket.wire) == 1 and len(records) == 1 and p.inflight is None
    assert not p.request_uncertain


@pytest.mark.asyncio
async def test_steer_freezes_after_preparation_before_send():
    records, order = [], []
    command = SimpleNamespace(id="local-1", text="before")

    async def prepare(kind, *, input_id):
        assert (kind, input_id) == ("response.steer", "local-1")
        await asyncio.sleep(0)
        command.text = "after preparation"

        def admit(envelope):
            records.append(envelope)
            order.append("admit")
            command.text = "after admission"
            asyncio.get_running_loop().call_soon(order.append, "later")
            return True

        return native.NativeSendAdmission(attempt_id="steer-attempt", admit=admit)

    p, _ = make(prepare)
    p.response_id = "parent"
    original_send = p.socket.send

    async def send(value):
        assert order == ["admit"]
        assert value is records[0].payload
        order.append("send")
        await original_send(value)

    p.socket.send = send
    assert await p.steer_live(command)
    assert json.loads(p.socket.wire[0])["input"] == "after preparation"
    assert records[0].attempt_id == "steer-attempt"
