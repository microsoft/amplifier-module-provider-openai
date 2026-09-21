"""Protocol fixtures use a real owned async transport, never live credentials."""

import asyncio
import copy
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import LLMError
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.native import (
    NATIVE_REQUEST,
    NativeResponsesProvider,
)


class Socket:
    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def send(self, value):
        self.sent.append(json.loads(value))

    async def recv(self):
        value = await self.events.get()
        if isinstance(value, BaseException):
            raise value
        return json.dumps(value)

    async def close(self):
        self.closed = True

    def feed(self, *events):
        for event in events:
            self.events.put_nowait(event)


class Owner:
    def __init__(self):
        self.messages = []
        self.events = []
        self.config = {}
        self.pending = deque()
        self.context = SimpleNamespace(
            get_messages=self.get_messages, add_message=self.add_message
        )
        self.runtime = SimpleNamespace(emit=self.emit)

    async def get_messages(self):
        return copy.deepcopy(self.messages)

    async def add_message(self, message):
        self.messages.append(message)

    async def emit(self, kind, **data):
        self.events.append({"type": kind, **data})

    def native_job(self, identity):
        return None

    def steer(self, text):
        self.wake = text

    def _text(self, command):
        return command.text


def make():
    owner = Owner()
    provider = NativeResponsesProvider.wrap(
        OpenAIProvider(
            api_key="fixture", config={"default_model": "gpt-6-astra", "max_retries": 0}
        ),
        owner_getter=lambda: owner,
    )
    provider.owner = owner
    provider.socket = Socket()
    return provider, owner


def created(identity):
    return {"type": "response.created", "response": {"id": identity}}


def finished(identity, *, incomplete=False):
    return {
        "type": "response.incomplete" if incomplete else "response.completed",
        "response": {
            "id": identity,
            "output": [],
            "usage": {"input_tokens": 2, "output_tokens": 3},
            **({"incomplete_details": {"reason": "steered"}} if incomplete else {}),
        },
    }


async def wait_created(provider):
    for _ in range(100):
        if provider.response_id:
            return
        await asyncio.sleep(0)
    raise AssertionError("no response created")


@pytest.mark.asyncio
async def test_acceptance_is_not_application_and_successor_does_not_recreate():
    p, o = make()
    p.socket.feed(created("parent"))
    task = asyncio.create_task(
        p._native_response({"model": "gpt-6-astra", "input": []})
    )
    await wait_created(p)
    command = SimpleNamespace(id="input-2", text="Change direction")
    assert await p.steer_live(command)
    p.socket.feed({"type": "response.steer.accepted", "steer": {"id": "steer-1"}})
    await asyncio.sleep(0)
    assert any(e["type"] == "steering.accepted" for e in o.events)
    assert not any(e["type"] == "steering.applied" for e in o.events)
    p.socket.feed(
        finished("parent", incomplete=True), created("successor"), finished("successor")
    )
    await task
    applied = next(e for e in o.events if e["type"] == "steering.applied")
    assert applied["input_id"] == "input-2" and applied["response_id"] == "successor"
    assert [i["type"] for i in p.socket.sent] == ["response.create", "response.steer"]
    assert len(o.messages) == 1 and p.request_uncertain is False


@pytest.mark.asyncio
async def test_pending_requires_matching_receipt_and_application_on_explicit_continuation():
    p, o = make()
    p.socket.feed(created("p"))
    task = asyncio.create_task(
        p._native_response({"model": "gpt-6-astra", "input": []})
    )
    await wait_created(p)
    await p.steer_live(SimpleNamespace(id="later", text="later"))
    p.socket.feed(
        {"type": "response.steer.accepted", "steer": {"id": "s"}},
        finished("p"),
        {"type": "response.steer.pending", "steer": {"id": "s"}, "required_input": []},
    )
    await task
    assert p.pending_parent and not any(
        e["type"] == "steering.applied" for e in o.events
    )
    p.socket.feed(created("next"), finished("next"))
    await p._native_response({"model": "gpt-6-astra", "input": []})
    assert p.socket.sent[-1]["previous_response_id"] == "p"
    assert (
        next(e for e in o.events if e["type"] == "steering.applied")["input_id"]
        == "later"
    )


@pytest.mark.asyncio
async def test_steer_failure_only_queues_explicit_matched_failed_input():
    p, o = make()
    p.socket.feed(created("p"))
    task = asyncio.create_task(p._native_response({"input": []}))
    await wait_created(p)
    c = SimpleNamespace(id="later", text="later")
    await p.steer_live(c)
    p.socket.feed(
        {"type": "response.steer.accepted", "steer": {"id": "s"}},
        {"type": "response.steer.failed", "steer": {"id": "other"}},
    )
    with pytest.raises(RuntimeError, match="Mismatched"):
        await task
    assert not o.pending and p.request_uncertain


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "end", [ConnectionError("fixture disconnect"), asyncio.CancelledError()]
)
async def test_transport_interruption_is_unknown_no_retry_or_automatic_reconnect(end):
    p, o = make()
    p._request_messages = []
    p._guard_assembled_params_with_provider_count = AsyncMock()
    socket = p.socket
    socket.feed(created("p"), end)
    token = NATIVE_REQUEST.set(p)
    try:
        with pytest.raises((LLMError, asyncio.CancelledError)):
            await p._create_response({"input": []})
        with pytest.raises(LLMError, match="uncertain"):
            await p._renew_idle_connection()
    finally:
        NATIVE_REQUEST.reset(token)
    assert socket.closed and len(socket.sent) == 1
    assert any(e["type"] == "native.outcome_unknown" for e in o.events)


@pytest.mark.asyncio
async def test_clean_idle_reconnect_renews_lineage_without_replaying_execution():
    from websockets.protocol import State

    p, o = make()
    socket = p.socket
    socket.state = State.CLOSED
    p.previous_response_id = "old"
    await p._renew_idle_connection()
    assert p.socket is None and p.previous_response_id is None and not socket.sent
    assert o.events[-1]["execution_replayed"] is False


@pytest.mark.asyncio
async def test_unsupported_auto_compaction_rejected_before_send():
    p, _ = make()
    with pytest.raises(ValueError, match="automatic compaction"):
        await p._native_response(
            {
                "input": [],
                "extra_body": {"context_management": [{"type": "compaction"}]},
            }
        )
    assert p.socket.sent == []


@pytest.mark.asyncio
async def test_current_provider_pure_planner_does_not_advance_lineage_before_actual_send():
    p, o = make()
    request = ChatRequest(
        messages=[Message(role="user", content="one")], max_output_tokens=32
    )
    o.messages = [m.model_dump() for m in request.messages]
    p._request_messages = o.messages
    p._guard_assembled_params_with_provider_count = AsyncMock()
    token = NATIVE_REQUEST.set(p)
    try:
        params, _, _ = p._assemble_initial_responses_params(request)
        assert not p.seen
        p.socket.feed(created("first"), finished("first"))
        await p._create_response(params)
        assert p.seen
        params, _, _ = p._assemble_initial_responses_params(request)
        assert params["input"]  # full preflight view
        p.socket.feed(created("second"), finished("second"))
        # A new top-level complete() resets this attempt boundary.
        p._native_request_attempted = False
        await p._create_response(params)
        assert p.socket.sent[-1]["input"] == []  # no duplicate user input
        assert p.socket.sent[-1]["previous_response_id"] == "first"
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
async def test_lifecycle_persistence_failure_blocks_first_external_send():
    p, _ = make()
    p.lifecycle = AsyncMock(side_effect=OSError("disk full"))
    with pytest.raises(OSError):
        await p._native_response({"input": []})
    assert p.socket.sent == []


@pytest.mark.asyncio
async def test_real_provider_completion_keeps_request_builder_response_parser_and_usage():
    p, o = make()
    p.owner = None
    request = ChatRequest(
        messages=[Message(role="user", content="fixture")], max_output_tokens=32
    )
    o.messages = [m.model_dump() for m in request.messages]
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=None)
    terminal = finished("r")
    terminal["response"].update(
        model="gpt-6-astra",
        status="completed",
        output=[
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "fixture response",
                        "annotations": [],
                    }
                ],
            }
        ],
    )
    p.socket.feed(created("r"), terminal)
    result = await p.complete(request)
    assert result.content[0].text == "fixture response"
    assert p.native_status()["compactAvailable"]
    assert p.socket.sent[0]["model"] == "gpt-6-astra"
    assert p.socket.sent[0]["input"][0]["content"]
    assert p.owner is None


@pytest.mark.asyncio
async def test_pending_native_result_is_not_synthesized_or_reexecuted():
    p, _ = make()
    p.pending_parent = True
    p.previous_response_id = "waiting"
    p.pending_steer = {
        "command": SimpleNamespace(id="i", text="later"),
        "required_input": [{"type": "function_call_output", "call_id": "actual-call"}],
    }
    with pytest.raises(ValueError, match="saved original"):
        await p._native_response(
            {
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "actual-call",
                        "output": "invented",
                    }
                ]
            }
        )
    assert not p.socket.sent


@pytest.mark.asyncio
async def test_broken_receipt_sink_cannot_escape_as_retryable_error():
    p, _ = make()
    socket = p.socket
    p._guard_assembled_params_with_provider_count = AsyncMock()
    p.lifecycle = AsyncMock(side_effect=OSError("fixture disk full"))
    token = NATIVE_REQUEST.set(p)
    try:
        with pytest.raises(LLMError) as result:
            await p._create_response({"input": []})
        assert result.value.retryable is False
        assert socket.closed and not socket.sent and p.request_uncertain
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
async def test_accepted_then_failed_input_is_retained_without_duplicate_boundary_queue():
    p, o = make()
    p.socket.feed(created("parent"))
    task = asyncio.create_task(p._native_response({"input": []}))
    await wait_created(p)
    await p.steer_live(SimpleNamespace(id="i", text="once"))
    assert not await p.steer_live(SimpleNamespace(id="second", text="later"))
    p.socket.feed(
        {"type": "response.steer.accepted", "steer": {"id": "s"}},
        {"type": "response.steer.failed", "steer": {"id": "s"}},
    )
    with pytest.raises(LLMError, match="explicit input recovery"):
        await task
    assert len(o.messages) == 1 and not o.pending


@pytest.mark.asyncio
async def test_definite_failure_before_acceptance_uses_boundary_once():
    p, o = make()
    p.socket.feed(created("parent"))
    task = asyncio.create_task(p._native_response({"input": []}))
    await wait_created(p)
    command = SimpleNamespace(id="i", text="once")
    await p.steer_live(command)
    p.socket.feed(
        {"type": "response.steer.failed", "steer": {"id": "s"}}, finished("parent")
    )
    await task
    assert list(o.pending) == [command] and not o.messages


@pytest.mark.asyncio
async def test_pending_steering_never_moves_to_rewritten_context_or_closed_socket():
    from collections import Counter

    from websockets.protocol import State

    p, _ = make()
    p.pending_parent = True
    p.last_context = Counter({"removed history": 1})
    token = NATIVE_REQUEST.set(p)
    try:
        with pytest.raises(LLMError, match="rewritten context"):
            p._convert_messages([])
    finally:
        NATIVE_REQUEST.reset(token)
    socket = p.socket
    socket.state = State.CLOSED
    with pytest.raises(LLMError, match="pending steering"):
        await p._renew_idle_connection()
    assert not socket.sent and p.request_uncertain


@pytest.mark.asyncio
async def test_driver_internal_continuation_cannot_issue_a_second_native_request():
    p, _ = make()
    p._native_request_attempted = True
    token = NATIVE_REQUEST.set(p)
    try:
        with pytest.raises(LLMError, match="no internal replay") as result:
            await p._create_response({"input": []})
        assert result.value.retryable is False and not p.socket.sent
    finally:
        NATIVE_REQUEST.reset(token)


def test_explicit_later_user_recovery_starts_new_lineage_once_without_native_call_replay():
    from amplifier_core.llm_errors import InvalidRequestError

    p, o = make()
    messages = [
        {"role": "user", "content": "Inspect"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "failed-computer",
                    "name": "computer",
                    "input": {"actions": [{"type": "screenshot"}]},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "failed-computer",
            "content": json.dumps({"success": False, "error": {"code": "safety_halt"}}),
        },
    ]
    original = copy.deepcopy(messages)
    p.previous_response_id = "prior"
    token = NATIVE_REQUEST.set(p)
    try:
        with pytest.raises(InvalidRequestError):
            p._convert_messages(messages)
        assert p.previous_response_id == "prior"
        later = [
            *messages,
            {"role": "user", "content": "Keep the stop. Explain without tools."},
        ]
        wire = p._convert_messages(later)
        assert p.previous_response_id is None
        assert not any(
            item.get("type") in {"computer_call", "computer_call_output"}
            for item in wire
        )
        assert "safety_halt" in json.dumps(wire) and messages == original
        epoch = p.epoch
        p.previous_response_id = "new-lineage"
        again = p._convert_messages(later)
        assert p.previous_response_id == "new-lineage" and p.epoch == epoch
        assert all(
            item.get("type") not in {"computer_call", "computer_call_output"}
            for item in again
        )
        assert p.socket.sent == [] and o.events == [] and o.messages == []
    finally:
        NATIVE_REQUEST.reset(token)
