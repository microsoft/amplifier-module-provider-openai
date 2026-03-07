"""Tests for use_streaming config flag and streaming path in _do_complete().

The streaming path uses client.responses.stream() (higher-level SDK helper)
with get_final_response() to collect the complete response via chunked HTTP,
preventing timeouts on large context requests without progressive token emission.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors as kernel_errors
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
