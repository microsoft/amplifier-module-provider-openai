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


class ReceiptHost:
    """Synthetic durable consumer: never infer not-sent from a local exception."""

    def __init__(self, path):
        self.path = path
        self.host_identity = {
            "attempt": "host-attempt", "owner": "host-owner", "budget": "original-budget",
        }
        self.events = []

    async def save(self, kind, data):
        if kind not in {"native.submitting", "native.not_sent"}:
            return
        self.events.append({"kind": kind, **copy.deepcopy(data), **self.host_identity})
        self.path.write_text(json.dumps(self.events))

    def restart(self):
        records = {}
        for event in json.loads(self.path.read_text()):
            identity = event["send_id"]
            if event["kind"] == "native.submitting":
                records[identity] = {**event, "status": "unknown"}
            elif event["kind"] == "native.not_sent":
                original = records[identity]
                for field in ("operation", "epoch", "response_id", "input_id",
                              "attempt", "owner", "budget"):
                    assert original[field] == event[field]
                records[identity] = {**event, "status": "not_sent"}
        return list(records.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refuse", "prepare-error", "prepare-cancel", "admit-cancel", "submitting-error"])
async def test_not_sent_survives_host_restart_with_original_identity(tmp_path, failure):
    host = ReceiptHost(tmp_path / "receipts.json")
    records = []

    async def prepare(*args, **kwargs):
        if failure == "prepare-error":
            raise ValueError("PRIVATE callback detail")
        if failure == "prepare-cancel":
            raise asyncio.CancelledError("PRIVATE cancellation")

        def decide(envelope):
            records.append(envelope)
            if failure == "admit-cancel":
                raise asyncio.CancelledError("PRIVATE cancellation")
            return False

        return native.NativeSendAdmission(attempt_id="attempt-1", admit=decide)

    p, _ = make(prepare)
    p.owner = None  # exercise complete(), including its terminal error handling
    request = ChatRequest(messages=[Message(role="user", content="PRIVATE input")], max_output_tokens=32)
    socket = p.socket
    epoch = p.epoch
    p.previous_response_id = "earlier-settled-response"

    async def lifecycle(kind, data):
        await host.save(kind, data)
        if kind == "native.submitting" and failure == "submitting-error":
            raise OSError("PRIVATE persistence detail")

    p.lifecycle = lifecycle
    with pytest.raises((LLMError, asyncio.CancelledError)):
        await p.complete(request)
    restored = host.restart()
    assert len(restored) == 1 and restored[0]["status"] == "not_sent"
    record = restored[0]
    assert record["epoch"] == epoch and record["response_id"] == "earlier-settled-response"
    assert record["operation"] == "response.create" and record["input_id"] is None
    assert record["attempt_id"] == ("attempt-1" if records else None)
    assert record["reason"] == ("cancelled" if "cancel" in failure else "rejected")
    assert record["execution_replayed"] is False
    assert "PRIVATE" not in host.path.read_text()
    assert not socket.wire and not socket.closed and p._native_send_refused
    assert not p.request_uncertain
    with pytest.raises(LLMError, match="not sent.*recovery"):
        await p.complete(request)
    await p.close_live()  # must settle original resources before reconstruction
    assert socket.closed and p.socket is None
    assert host.restart()[0]["status"] == "not_sent"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel", "timeout"])
async def test_not_sent_receipt_failure_is_visible_and_restart_stays_unknown(tmp_path, monkeypatch, failure):
    host = ReceiptHost(tmp_path / "receipts.json")
    p, o = make(lambda *args, **kwargs: admission([], decision=False))
    monkeypatch.setattr(native, "_NOT_SENT_RECEIPT_TIMEOUT", 0.01)

    async def lifecycle(kind, data):
        if kind == "native.not_sent":
            if failure == "timeout":
                await asyncio.Future()
            if failure == "cancel":
                raise asyncio.CancelledError("PRIVATE sink cancellation")
            raise OSError("PRIVATE sink error")
        await host.save(kind, data)

    p.lifecycle = lifecycle
    token = native.NATIVE_REQUEST.set(p)
    try:
        with pytest.raises(LLMError) as error:
            await asyncio.wait_for(p._create_response({"input": []}), 2)
    finally:
        native.NATIVE_REQUEST.reset(token)
    assert error.value.retryable is False and "PRIVATE" not in str(error.value)
    assert host.restart()[0]["status"] == "unknown"
    assert p.native_status()["observerFailed"] is True
    assert any(e["type"] == "native.not_sent" for e in o.events)  # other sink still attempted
    assert not p.socket.wire and not p.request_uncertain


@pytest.mark.asyncio
async def test_not_sent_receipt_uses_captured_sinks_after_owner_changes(tmp_path):
    host = ReceiptHost(tmp_path / "receipts.json")
    replacement = Owner()

    async def prepare(*args, **kwargs):
        p.owner = replacement
        p.lifecycle = AsyncMock()
        p.epoch = "replacement-epoch"
        return admission([])

    p, original = make(prepare)
    epoch = p.epoch
    p.lifecycle = host.save
    with pytest.raises(LLMError):
        await p._native_response({"input": []})
    assert host.restart()[0]["epoch"] == epoch
    assert host.restart()[0]["status"] == "not_sent"
    assert any(e["type"] == "native.not_sent" for e in original.events)
    assert not replacement.events and p.lifecycle.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["refuse", "cancel", "parent-completes"])
async def test_unsent_steer_receipt_cannot_terminate_or_downgrade_parent(tmp_path, ending):
    host = ReceiptHost(tmp_path / "receipts.json")
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare(kind, *, input_id):
        if kind == "response.steer":
            entered.set()
            await release.wait()
            if ending == "cancel":
                raise asyncio.CancelledError()
            return admission([], decision=False)
        return admission([])

    p, _ = make(prepare)
    p.lifecycle = host.save
    socket = p.socket
    socket.feed(created("parent"))
    parent = asyncio.create_task(p._native_response({"input": []}))
    await wait_created(p)
    steer = asyncio.create_task(p.steer_live(SimpleNamespace(id="local-1", text="PRIVATE steer")))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if ending == "parent-completes":
            socket.feed(finished("parent"))
            await asyncio.wait_for(parent, 2)
        release.set()
        with pytest.raises((LLMError, asyncio.CancelledError)):
            await asyncio.wait_for(steer, 2)
        persisted = host.restart()
        assert [(e["operation"], e["status"]) for e in persisted] == [
            ("response.create", "unknown"), ("response.steer", "not_sent"),
        ]  # synthetic consumer deliberately has no completion handler
        assert persisted[1]["input_id"] == "local-1"
        assert persisted[1]["response_id"] == "parent"
        assert p.request_uncertain is (ending != "parent-completes")
        assert not socket.closed and len(socket.wire) == 1
        assert p.inflight is None and not p._steer_preparing
        if not parent.done():
            socket.feed(finished("parent"))
            await asyncio.wait_for(parent, 2)
    finally:
        for task in (parent, steer):
            if not task.done():
                task.cancel()
        await asyncio.gather(parent, steer, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "steer"])
@pytest.mark.parametrize("failure", [ConnectionError("PRIVATE send"), asyncio.CancelledError()])
async def test_send_entry_failure_never_publishes_not_sent(tmp_path, operation, failure):
    host = ReceiptHost(tmp_path / "receipts.json")
    p, o = make(lambda *args, **kwargs: admission([]))
    p.lifecycle = host.save
    socket = p.socket
    socket.failure = failure
    token = native.NATIVE_REQUEST.set(p)
    try:
        if operation == "steer":
            p.response_id = "parent"
            p.request_uncertain = True
            call = p.steer_live(SimpleNamespace(id="local-1", text="PRIVATE steer"))
        else:
            call = p._create_response({"input": []})
        with pytest.raises((LLMError, ConnectionError, asyncio.CancelledError)):
            await call
    finally:
        native.NATIVE_REQUEST.reset(token)
    assert host.restart()[0]["status"] == "unknown"
    assert len(socket.wire) == 1 and socket.closed and p.request_uncertain
    assert not any(e["type"] == "native.not_sent" for e in o.events)


@pytest.mark.asyncio
async def test_not_sent_does_not_hide_failed_socket_cleanup(tmp_path):
    host = ReceiptHost(tmp_path / "receipts.json")
    p, _ = make(lambda *args, **kwargs: admission([], decision=False))
    p.lifecycle = host.save
    with pytest.raises(LLMError):
        await p._native_response({"input": []})
    socket = p.socket
    socket.close = AsyncMock(side_effect=OSError("synthetic close failure"))
    with pytest.raises(OSError, match="synthetic close failure"):
        await p.close_live()
    assert p.socket is socket and not socket.closed
    assert host.restart()[0]["status"] == "not_sent"  # not a cleanup success
    assert p._native_send_refused and not socket.wire
