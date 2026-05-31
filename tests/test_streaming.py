"""Tests for use_streaming config flag and streaming path in _do_complete().

The streaming path uses client.responses.stream() (higher-level SDK helper)
with get_final_response() to collect the complete response via chunked HTTP,
preventing timeouts on large context requests without progressive token emission.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import ModuleCoordinator, llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyResponse:
    """Minimal response stub for provider tests."""

    def __init__(self, output=None):
        self.output = output or []
        self.usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        self.stop_reason = "stop"


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


class MockStreamContext:
    """Async context manager that yields a mock stream object."""

    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *args):
        pass


class HangingStream:
    """A stream that hangs forever — used to test timeout."""

    async def get_final_response(self):
        await asyncio.sleep(9999)  # Simulate hanging


class NoCompletedEventStream:
    """A stream that raises RuntimeError (no completed event) — SDK behaviour."""

    async def get_final_response(self):
        raise RuntimeError("Didn't receive a `response.completed` event.")


class SuccessStream:
    """A stream that immediately returns the given response (yields no events)."""

    def __init__(self, response):
        self._response = response

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_response(self):
        return self._response


class SuccessStreamWithHeaders:
    """A stream that returns a response and exposes a mock HTTP _response with headers.

    Mimics the OpenAI SDK's AsyncResponseStream which stores the underlying
    httpx response as ``self._response`` (with a ``.headers`` mapping).
    Yields no events so it is safe to iterate with an emit_stream_events loop.
    """

    def __init__(self, api_response, headers):
        # _response simulates the underlying httpx response; headers is a dict or None.
        self._response = SimpleNamespace(headers=headers if headers is not None else {})
        self._api_response = api_response

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_response(self):
        return self._api_response


class FakeHooks:
    """Records every (event_name, payload) pair emitted via hooks."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


# ---------------------------------------------------------------------------
# 1. Default config
# ---------------------------------------------------------------------------


def test_streaming_default_on():
    """Provider created with default config should have use_streaming=True."""
    provider = OpenAIProvider(api_key="test-key")
    assert provider.use_streaming is True


# ---------------------------------------------------------------------------
# 2. Can be disabled
# ---------------------------------------------------------------------------


def test_streaming_can_be_disabled():
    """Provider created with use_streaming=False should have use_streaming=False."""
    provider = OpenAIProvider(api_key="test-key", config={"use_streaming": False})
    assert provider.use_streaming is False


# ---------------------------------------------------------------------------
# 3. Non-streaming path works (backward compat / test helper path)
# ---------------------------------------------------------------------------


def test_non_streaming_path_works():
    """With use_streaming=False, the blocking create() path still works."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": False, "max_retries": 0},
    )
    dummy = DummyResponse()
    provider.client.responses.create = AsyncMock(return_value=dummy)

    asyncio.run(provider.complete(_simple_request()))

    provider.client.responses.create.assert_awaited_once()


# ---------------------------------------------------------------------------
# 4. Streaming path collects response
# ---------------------------------------------------------------------------


def test_streaming_path_collects_response():
    """With use_streaming=True, stream is called and get_final_response() result returned."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    dummy = DummyResponse()
    stream = SuccessStream(dummy)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    provider.client.responses.stream.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Timeout fires on streaming path
# ---------------------------------------------------------------------------


def test_streaming_timeout_fires():
    """A hanging stream raises LLMTimeoutError (via asyncio.TimeoutError translation)."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={
            "use_streaming": True,
            "max_retries": 0,
            "timeout": 0.05,  # Very short so the test is fast
        },
    )
    hanging = HangingStream()
    provider.client.responses.stream = MagicMock(
        return_value=MockStreamContext(hanging)
    )

    with pytest.raises(kernel_errors.LLMTimeoutError):
        asyncio.run(provider.complete(_simple_request()))


# ---------------------------------------------------------------------------
# 6. Stream ends without completed event → error raised
# ---------------------------------------------------------------------------


def test_streaming_no_completed_event_raises():
    """A stream that finishes without completed event raises an LLMError."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    bad_stream = NoCompletedEventStream()
    provider.client.responses.stream = MagicMock(
        return_value=MockStreamContext(bad_stream)
    )

    with pytest.raises(kernel_errors.LLMError):
        asyncio.run(provider.complete(_simple_request()))


# ---------------------------------------------------------------------------
# 7. Rate limit header extraction from streaming response
# ---------------------------------------------------------------------------


def test_streaming_extracts_rate_limit_headers():
    """Streaming response with rate limit headers → rate_limits in llm:response event."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    dummy = DummyResponse()
    stream = SuccessStreamWithHeaders(
        dummy,
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-reset-requests": "1s",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "95000",
            "x-ratelimit-reset-tokens": "10ms",
        },
    )
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    response_payloads = [
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:response"
    ]
    assert len(response_payloads) == 1, "Expected exactly one llm:response event"
    rate_limits = response_payloads[0].get("rate_limits")
    assert rate_limits is not None, "rate_limits key missing from llm:response event"
    assert rate_limits["requests_limit"] == 1000
    assert rate_limits["requests_remaining"] == 999
    assert rate_limits["requests_reset"] == "1s"
    assert rate_limits["tokens_limit"] == 100000
    assert rate_limits["tokens_remaining"] == 95000
    assert rate_limits["tokens_reset"] == "10ms"


# ---------------------------------------------------------------------------
# 8. Missing headers → no error (graceful fallback)
# ---------------------------------------------------------------------------


def test_streaming_missing_headers_no_error():
    """Streaming response without rate limit headers completes without error."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # SuccessStreamWithHeaders with empty headers dict — no x-ratelimit-* values
    dummy = DummyResponse()
    stream = SuccessStreamWithHeaders(dummy, headers={})
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    # Must not raise
    asyncio.run(provider.complete(_simple_request()))

    response_payloads = [
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:response"
    ]
    assert len(response_payloads) == 1, "Expected exactly one llm:response event"
    # rate_limits key should be absent (no headers → empty dict → omitted)
    assert "rate_limits" not in response_payloads[0]


# ===========================================================================
# Contract conformance tests (llm:stream_* events — feat/proper-streaming)
# ===========================================================================


class FakeEventStream:
    """Async-iterable stream that yields a fixed list of synthetic SDK events,
    then returns a pre-built response from get_final_response().

    Attributes
    ----------
    _response : SimpleNamespace
        Simulates stream._response with an empty headers mapping so that
        _extract_rate_limit_headers() does not blow up.
    """

    def __init__(self, events, response=None):
        self._events = list(events)
        self._response = SimpleNamespace(headers={})
        self._final_response = response or DummyResponse()
        self._pos = 0

    # --- async iteration support ----------------------------------------

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._events):
            raise StopAsyncIteration
        ev = self._events[self._pos]
        self._pos += 1
        return ev

    # --- SDK stream API --------------------------------------------------

    async def get_final_response(self):
        return self._final_response


class ErrorMidStream(FakeEventStream):
    """Stream that raises RuntimeError after yielding all provided events.

    Used to test the llm:stream_aborted path: iterate normally (emitting
    block_start + delta so partial_emitted becomes True), then raise to
    trigger the abort handler.
    """

    def __init__(self, events, error=None, response=None):
        super().__init__(events, response)
        self._error = error or RuntimeError("mid-stream error")

    async def __anext__(self):
        if self._pos >= len(self._events):
            raise self._error
        ev = self._events[self._pos]
        self._pos += 1
        return ev


def _make_event(type_str, **kwargs):
    """Build a fake SDK event using SimpleNamespace."""
    return SimpleNamespace(type=type_str, **kwargs)


def _make_item(**kwargs):
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# 9. Text block: block_start -> delta (sequence 0) -> block_end
# ---------------------------------------------------------------------------


def test_stream_text_block_events_in_order():
    """A message output block emits block_start / block_delta / block_end in order."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    events = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="message"),
        ),
        _make_event(
            "response.output_text.delta",
            output_index=0,
            delta="Hello, world!",
        ),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="message"),
        ),
    ]
    stream = FakeEventStream(events)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    stream_events = [
        (name, payload)
        for name, payload in fake_coordinator.hooks.events
        if name.startswith("llm:stream_")
    ]

    # Exact order: block_start, block_delta, block_end
    assert len(stream_events) == 3, f"Expected 3 stream events, got {len(stream_events)}"

    name0, p0 = stream_events[0]
    assert name0 == "llm:stream_block_start"
    assert p0["block_index"] == 0
    assert p0["block_type"] == "text"
    request_id = p0["request_id"]
    assert isinstance(request_id, str) and len(request_id) == 36  # uuid4

    name1, p1 = stream_events[1]
    assert name1 == "llm:stream_block_delta"
    assert p1["request_id"] == request_id
    assert p1["block_index"] == 0
    assert p1["block_type"] == "text"
    assert p1["sequence"] == 0
    assert p1["text"] == "Hello, world!"

    name2, p2 = stream_events[2]
    assert name2 == "llm:stream_block_end"
    assert p2["request_id"] == request_id
    assert p2["block_index"] == 0
    assert p2["block_type"] == "text"


# ---------------------------------------------------------------------------
# 10. Multiple deltas: sequence is per-block and 0-based
# ---------------------------------------------------------------------------


def test_stream_delta_sequence_is_per_block():
    """sequence is 0-based and increments per block, not globally."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    events = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="message"),
        ),
        _make_event("response.output_text.delta", output_index=0, delta="A"),
        _make_event("response.output_text.delta", output_index=0, delta="B"),
        _make_event("response.output_text.delta", output_index=0, delta="C"),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="message"),
        ),
    ]
    stream = FakeEventStream(events)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    deltas = [
        p
        for name, p in fake_coordinator.hooks.events
        if name == "llm:stream_block_delta"
    ]
    assert len(deltas) == 3
    assert [d["sequence"] for d in deltas] == [0, 1, 2]
    assert [d["block_index"] for d in deltas] == [0, 0, 0]


# ---------------------------------------------------------------------------
# 11. Thinking (reasoning) block emits llm:stream_block_delta with block_type:"thinking"
# ---------------------------------------------------------------------------


def test_stream_thinking_block():
    """A reasoning output block emits block_delta with block_type='thinking' (not a separate event)."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    events = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="reasoning"),
        ),
        _make_event(
            "response.reasoning_summary_text.delta",
            output_index=0,
            delta="I am thinking...",
        ),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="reasoning"),
        ),
    ]
    stream = FakeEventStream(events)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    stream_events = [
        (name, payload)
        for name, payload in fake_coordinator.hooks.events
        if name.startswith("llm:stream_")
    ]

    names = [n for n, _ in stream_events]
    assert names == [
        "llm:stream_block_start",
        "llm:stream_block_delta",
        "llm:stream_block_end",
    ], f"Got: {names}"

    _, start_p = stream_events[0]
    assert start_p["block_type"] == "thinking"

    _, delta_p = stream_events[1]
    assert delta_p["text"] == "I am thinking..."
    assert delta_p["sequence"] == 0
    assert delta_p["request_id"] == start_p["request_id"]
    assert delta_p["block_type"] == "thinking"

    _, end_p = stream_events[2]
    assert end_p["block_type"] == "thinking"


# ---------------------------------------------------------------------------
# 12. tool_use block: block_start (with name) + block_end only — no deltas
# ---------------------------------------------------------------------------


def test_stream_tool_use_no_arg_deltas():
    """Function-call block emits block_start (with name) and block_end; no deltas."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    events = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="function_call", name="search"),
        ),
        # function_call_arguments.delta — must be silently ignored
        _make_event(
            "response.function_call_arguments.delta",
            output_index=0,
            delta='{"query": "hello"}',
        ),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="function_call", name="search"),
        ),
    ]
    stream = FakeEventStream(events)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    stream_events = [
        (name, payload)
        for name, payload in fake_coordinator.hooks.events
        if name.startswith("llm:stream_")
    ]

    names = [n for n, _ in stream_events]
    assert names == [
        "llm:stream_block_start",
        "llm:stream_block_end",
    ], f"Expected only block_start + block_end, got {names}"

    _, start_p = stream_events[0]
    assert start_p["block_type"] == "tool_use"
    assert start_p["name"] == "search"


# ---------------------------------------------------------------------------
# 13. Per-request override: metadata={"stream": False} suppresses all stream events
# ---------------------------------------------------------------------------


def test_stream_disabled_via_metadata_no_stream_events():
    """request.metadata={'stream': False} falls back to create(); no llm:stream_* events."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    dummy = DummyResponse()
    provider.client.responses.create = AsyncMock(return_value=dummy)

    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        metadata={"stream": False},
    )
    asyncio.run(provider.complete(request))

    provider.client.responses.create.assert_awaited_once()

    stream_events = [
        name for name, _ in fake_coordinator.hooks.events if name.startswith("llm:stream_")
    ]
    assert stream_events == [], f"Expected no stream events, got: {stream_events}"


# ---------------------------------------------------------------------------
# 14. Mid-stream exception after partial emit produces llm:stream_aborted
# ---------------------------------------------------------------------------


def test_stream_aborted_after_partial_emit():
    """After a delta is emitted, a mid-stream error triggers llm:stream_aborted."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Yield a block_start + delta (sets partial_emitted=True), then raise
    events_before_error = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="message"),
        ),
        _make_event(
            "response.output_text.delta",
            output_index=0,
            delta="partial text",
        ),
    ]
    stream = ErrorMidStream(events_before_error)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    with pytest.raises(kernel_errors.LLMError):
        asyncio.run(provider.complete(_simple_request()))

    aborted = [
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:stream_aborted"
    ]
    assert len(aborted) == 1, f"Expected 1 stream_aborted, got {len(aborted)}"
    assert "request_id" in aborted[0]
    assert aborted[0]["error"]["type"] == "RuntimeError"
    assert aborted[0]["error"]["msg"] == "mid-stream error"


# ---------------------------------------------------------------------------
# 15. No abort event when error occurs before any delta (not partial)
# ---------------------------------------------------------------------------


def test_stream_no_abort_when_no_delta_emitted():
    """If no delta was emitted before the error, llm:stream_aborted must NOT fire."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Yield only block_start (no deltas), then raise
    events_before_error = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="message"),
        ),
    ]
    stream = ErrorMidStream(events_before_error)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    with pytest.raises(kernel_errors.LLMError):
        asyncio.run(provider.complete(_simple_request()))

    aborted = [
        name for name, _ in fake_coordinator.hooks.events if name == "llm:stream_aborted"
    ]
    assert aborted == [], f"stream_aborted must not fire without a prior delta, got {aborted}"


# ---------------------------------------------------------------------------
# 16. All stream events for a single call share ONE request_id
# ---------------------------------------------------------------------------


def test_stream_single_request_id_for_all_events():
    """Every stream event in one complete() call shares the same request_id."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    events = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="message"),
        ),
        _make_event("response.output_text.delta", output_index=0, delta="Hi"),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="message"),
        ),
    ]
    stream = FakeEventStream(events)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    request_ids = {
        payload["request_id"]
        for name, payload in fake_coordinator.hooks.events
        if name.startswith("llm:stream_")
    }
    assert len(request_ids) == 1, f"Expected 1 request_id, got {request_ids}"


# ---------------------------------------------------------------------------
# 17. Empty delta is never emitted
# ---------------------------------------------------------------------------


def test_stream_empty_delta_not_emitted():
    """Empty-string deltas are silently skipped (contract: 'if text:')."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 0},
    )
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    events = [
        _make_event(
            "response.output_item.added",
            output_index=0,
            item=_make_item(type="message"),
        ),
        _make_event("response.output_text.delta", output_index=0, delta=""),  # empty
        _make_event("response.output_text.delta", output_index=0, delta="real"),
        _make_event(
            "response.output_item.done",
            output_index=0,
            item=_make_item(type="message"),
        ),
    ]
    stream = FakeEventStream(events)
    provider.client.responses.stream = MagicMock(return_value=MockStreamContext(stream))

    asyncio.run(provider.complete(_simple_request()))

    deltas = [
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:stream_block_delta"
    ]
    assert len(deltas) == 1, "Only the non-empty delta should be emitted"
    assert deltas[0]["text"] == "real"
    assert deltas[0]["block_type"] == "text"
