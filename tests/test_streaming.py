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
    """A stream that immediately returns the given response."""

    def __init__(self, response):
        self._response = response

    async def get_final_response(self):
        return self._response


class SuccessStreamWithHeaders:
    """A stream that returns a response and exposes a mock HTTP _response with headers.

    Mimics the OpenAI SDK's AsyncResponseStream which stores the underlying
    httpx response as ``self._response`` (with a ``.headers`` mapping).
    """

    def __init__(self, api_response, headers):
        # _response simulates the underlying httpx response; headers is a dict or None.
        self._response = SimpleNamespace(headers=headers if headers is not None else {})
        self._api_response = api_response

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
