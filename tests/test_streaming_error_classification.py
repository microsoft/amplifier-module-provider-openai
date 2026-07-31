"""Streaming APIError classification.

OpenAI returns HTTP 200 for the Responses API, opens the SSE stream, then
emits an SSE "error" event when the request fails mid-stream. The openai SDK
re-raises this as a bare openai.APIError -- the PARENT of APIStatusError,
with no status_code attribute. Before the fix, such errors fell through
every typed except branch in _do_complete() and landed in the generic
`except Exception` catch-all, which hardcodes retryable=True. A deterministic
400-class error (e.g. context overflow) was therefore retried max_retries
times before failing.

This also covers the related stale-substring-gate fix: the non-streaming
BadRequestError branch's context-overflow detection now checks the error
code first (context_length_exceeded) and falls back to substring markers
that include OpenAI's current Responses API wording ("exceeds the context
window of this model"), which none of the legacy markers matched.

Harness mirrors test_error_translation.py / test_error_body_json.py.
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from amplifier_core import ModuleCoordinator, llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    """Create a provider with retries disabled so errors propagate immediately."""
    config = {"max_retries": 0, "use_streaming": False, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def _make_bare_api_error(message: str, body: object | None) -> openai.APIError:
    """Build a bare openai.APIError -- the shape the SDK raises when an SSE
    'error' event arrives mid-stream (HTTP 200 already sent, no status_code)."""
    return openai.APIError(message, _mock_httpx_request(), body=body)


CONTEXT_OVERFLOW_BODY_WITH_CODE = {
    "type": "invalid_request_error",
    "code": "context_length_exceeded",
    "message": "Your input exceeds the context window of this model",
}

INVALID_VALUE_BODY = {
    "type": "invalid_request_error",
    "code": "invalid_value",
    "message": "Invalid value for parameter",
}


# ---------------------------------------------------------------------------
# Case 1: bare APIError, code=context_length_exceeded -> ContextLengthError
# ---------------------------------------------------------------------------


def test_bare_api_error_with_context_overflow_code_raises_context_length_error():
    """bare openai.APIError with code=context_length_exceeded -> ContextLengthError,
    not a bare LLMError, and non-retryable."""
    provider = _make_provider()
    native = _make_bare_api_error(
        "Your input exceeds the context window of this model",
        body=CONTEXT_OVERFLOW_BODY_WITH_CODE,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "openai"
    assert err.status_code == 400
    assert err.retryable is False
    assert err.__cause__ is native


# ---------------------------------------------------------------------------
# Case 2: bare APIError, no code, message-only -> still ContextLengthError
# ---------------------------------------------------------------------------


def test_bare_api_error_message_only_overflow_raises_context_length_error():
    """Same overflow condition but with NO machine-readable code -- the
    substring fallback ("exceeds the context window") must still catch it.
    This is the stale-substring-gate fix: the old markers ("context length",
    "too many tokens", "maximum context") never matched this current wording.
    """
    provider = _make_provider()
    body = {
        "type": "invalid_request_error",
        "message": "Your input exceeds the context window of this model",
    }
    native = _make_bare_api_error(
        "Your input exceeds the context window of this model", body=body
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# Case 3: bare APIError, code=invalid_value -> InvalidRequestError
# ---------------------------------------------------------------------------


def test_bare_api_error_invalid_value_raises_invalid_request_error():
    """bare openai.APIError with a non-overflow code -> InvalidRequestError,
    non-retryable (deterministic client error, no point retrying)."""
    provider = _make_provider()
    native = _make_bare_api_error(
        "Invalid value for parameter", body=INVALID_VALUE_BODY
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "openai"
    assert err.retryable is False
    assert err.__cause__ is native


# ---------------------------------------------------------------------------
# Case 4: APIConnectionError -> LLMError retryable=True (blast-radius guard)
# ---------------------------------------------------------------------------


def test_api_connection_error_still_retryable():
    """openai.APIConnectionError is a subclass of openai.APIError but must
    remain retryable -- transport-level failures are genuinely transient.
    Regression guard for the blast-radius constraint: the new APIError
    branch must not swallow this into a non-retryable classification."""
    provider = _make_provider()
    native = openai.APIConnectionError(
        message="Connection failed", request=_mock_httpx_request()
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.provider == "openai"
    assert err.retryable is True
    assert err.__cause__ is native


# ---------------------------------------------------------------------------
# Case 5: bare APIError, body=None, opaque message -> LLMError retryable=True
# ---------------------------------------------------------------------------


def test_bare_api_error_unclassifiable_preserves_prior_default():
    """bare openai.APIError with body=None and no classifiable signal must
    preserve the pre-fix conservative default: LLMError, retryable=True."""
    provider = _make_provider()
    native = _make_bare_api_error("Something went wrong mid-stream", body=None)
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert not isinstance(err, kernel_errors.ContextLengthError)
    assert not isinstance(err, kernel_errors.InvalidRequestError)
    assert err.provider == "openai"
    assert err.retryable is True
    assert err.__cause__ is native


# ---------------------------------------------------------------------------
# Case 6: retry-count regression guard -- exactly ONE attempt, not 6
# ---------------------------------------------------------------------------


def test_context_overflow_api_error_is_not_retried():
    """Pre-fix behaviour: a bare APIError fell through to `except Exception`,
    which hardcodes retryable=True, so a deterministic context-overflow error
    was retried max_retries times (default 5 -> 6 total attempts). Post-fix,
    the error is classified as ContextLengthError (retryable=False) and must
    produce exactly ONE attempt. Modeled on test_retry.py's harness.
    """
    provider = _make_provider(max_retries=5)
    native = _make_bare_api_error(
        "Your input exceeds the context window of this model",
        body=CONTEXT_OVERFLOW_BODY_WITH_CODE,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(kernel_errors.ContextLengthError):
            asyncio.run(provider.complete(_simple_request()))

    assert provider.client.responses.create.await_count == 1, (
        "Context overflow is a deterministic 400 -- it must not be retried. "
        f"Got {provider.client.responses.create.await_count} attempts "
        "(pre-fix behaviour was 6: 1 initial + 5 retries)."
    )


# ---------------------------------------------------------------------------
# Case 7: non-streaming BadRequestError, current wording -> ContextLengthError
# ---------------------------------------------------------------------------


def test_non_streaming_bad_request_current_wording_raises_context_length_error():
    """Non-streaming path: openai.BadRequestError whose message is the current
    Responses API wording ("Your input exceeds the context window of this
    model") with code=context_length_exceeded must raise ContextLengthError,
    not fall through to InvalidRequestError. This is the stale-substring-gate
    fix on the 400 path (the pre-fix substring list never matched this
    wording, so it fell into the `else` branch -> InvalidRequestError).
    """
    provider = _make_provider()
    native = openai.BadRequestError(
        "Your input exceeds the context window of this model",
        response=httpx.Response(400, request=_mock_httpx_request()),
        body={
            "error": {
                "code": "context_length_exceeded",
                "message": "Your input exceeds the context window of this model",
            }
        },
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert not isinstance(exc_info.value, kernel_errors.InvalidRequestError)


# ---------------------------------------------------------------------------
# Streaming-path delivery
#
# Everything above hands a pre-built APIError to the classification ladder,
# which proves the ladder classifies correctly but says nothing about how the
# error gets there. These two tests drive the actual streaming code path -- the
# one this fix exists for -- so the load-bearing premise is guarded: a bare
# APIError raised out of client.responses.stream() must escape the streaming
# block intact and reach the classifier. If the streaming block ever starts
# wrapping exceptions, these fail while the classification tests still pass.
# ---------------------------------------------------------------------------


class _StreamContext:
    """Async context manager standing in for client.responses.stream()."""

    def __init__(self, stream):
        self._stream = stream

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *args):
        return False


class _StreamRaisingOnFinalResponse:
    """Stream that yields no events, then fails in get_final_response().

    Mirrors the SDK when the SSE 'error' event is consumed while collecting the
    terminal response (the no-coordinator path, where events are not emitted).
    """

    def __init__(self, error: openai.APIError):
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_response(self):
        raise self._error


class _StreamRaisingDuringIteration:
    """Stream that raises the bare APIError while events are being iterated.

    Mirrors openai._streaming.__stream__, which raises APIError(body=data["error"])
    the moment an SSE 'error' event is decoded.
    """

    def __init__(self, error: openai.APIError):
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._error

    async def get_final_response(self):  # pragma: no cover - never reached
        raise AssertionError("iteration should have raised first")


class _FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class _FakeCoordinator:
    def __init__(self):
        self.hooks = _FakeHooks()


def test_streaming_path_sse_error_reaches_classifier_and_is_not_retried():
    """A bare APIError from the stream must classify as ContextLengthError.

    Drives the real streaming branch (use_streaming=True). max_retries=5 would
    have produced 5 stream() calls before the fix; a correctly classified
    deterministic error produces exactly 1.
    """
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 5},
    )
    error = _make_bare_api_error(
        "Your input exceeds the context window of this model.",
        CONTEXT_OVERFLOW_BODY_WITH_CODE,
    )
    stream_mock = MagicMock(
        return_value=_StreamContext(_StreamRaisingOnFinalResponse(error))
    )
    provider.client.responses.stream = stream_mock

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_simple_request()))

    assert stream_mock.call_count == 1, (
        f"deterministic streaming error was retried "
        f"{stream_mock.call_count} times; expected a single attempt"
    )


def test_streaming_path_sse_error_during_event_iteration_is_classified():
    """The same classification must hold on the event-emitting stream path."""
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 5},
    )
    provider.coordinator = cast(ModuleCoordinator, _FakeCoordinator())
    error = _make_bare_api_error(
        "Your input exceeds the context window of this model.",
        CONTEXT_OVERFLOW_BODY_WITH_CODE,
    )
    stream_mock = MagicMock(
        return_value=_StreamContext(_StreamRaisingDuringIteration(error))
    )
    provider.client.responses.stream = stream_mock

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_simple_request()))

    assert stream_mock.call_count == 1


def test_streaming_path_transient_sse_error_stays_retryable():
    """A mid-stream server error must remain retryable.

    Guards the classification boundary from the opposite side: the fix must not
    make transient streaming failures permanent. server_error and
    rate_limit_exceeded are the first two codes documented for this API in
    openai.types.responses.ResponseError.
    """
    provider = OpenAIProvider(
        api_key="test-key",
        config={"use_streaming": True, "max_retries": 2},
    )
    error = _make_bare_api_error(
        "The server had an error while processing your request.",
        {"type": "server_error", "code": "server_error", "message": "server error"},
    )
    stream_mock = MagicMock(
        return_value=_StreamContext(_StreamRaisingOnFinalResponse(error))
    )
    provider.client.responses.stream = stream_mock

    with patch("asyncio.sleep", new=AsyncMock()):
        with pytest.raises(kernel_errors.LLMError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

    assert not isinstance(exc_info.value, kernel_errors.InvalidRequestError)
    assert exc_info.value.retryable is True
    assert stream_mock.call_count == 3, (
        f"transient streaming error should exhaust retries (3 attempts), "
        f"got {stream_mock.call_count}"
    )
