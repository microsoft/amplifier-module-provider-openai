"""Adversarial regression coverage for `misalignment_policy_violation`.

OpenAI's behavioral-misalignment monitor can stop a request or response for
this reason, documented to appear in (at least) four shapes:

  1. A pre-stream HTTP 403 (JSON body `error.code`).
  2. A streamed `response.failed` terminal (`response.error.code`).
  3. A flat SSE `error` event (`event.code`).
  4. An ordinary HTTP 200 non-streaming (or continuation/truncation-retry)
     response body with `status: "failed"` (`response.error.code`).

Every one of these must translate to a non-retryable
`kernel_errors.ContentFilterError` -- never retried, never returned as a
(partial) success, never left to escape as a raw SDK exception. This file
drives the REAL openai-python SDK over `httpx.MockTransport` so the SSE
parsing exercised is real; only the network transport is faked. This is
therefore genuinely SSE / streaming-bytes testing, not a mocked event
object standing in for one -- see the note in test_streaming_error_classification.py
for the older, non-streaming-only style this complements.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(events: list[tuple[str, dict]]) -> bytes:
    out = ""
    for name, data in events:
        out += f"event: {name}\ndata: {json.dumps(data)}\n\n"
    return out.encode()


_BASE_RESPONSE: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-6-sol",
    "status": "in_progress",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": {},
    "temperature": None,
    "top_p": None,
}

_MISALIGNMENT_ERROR: dict[str, Any] = {
    "code": "misalignment_policy_violation",
    "message": "blocked by monitor",
    "type": "invalid_request_error",
    "param": None,
}


def _created_event() -> tuple[str, dict]:
    return (
        "response.created",
        {"type": "response.created", "sequence_number": 0, "response": _BASE_RESPONSE},
    )


def _function_call_added_event() -> tuple[str, dict]:
    return (
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "bash",
                "arguments": "",
                "status": "in_progress",
            },
        },
    )


def _failed_event() -> tuple[str, dict]:
    failed_response = dict(
        _BASE_RESPONSE,
        status="failed",
        error={"code": "misalignment_policy_violation", "message": "blocked by monitor"},
    )
    return (
        "response.failed",
        {"type": "response.failed", "sequence_number": 2, "response": failed_response},
    )


def _flat_error_event() -> tuple[str, dict]:
    return (
        "error",
        {
            "type": "error",
            "sequence_number": 2,
            "code": "misalignment_policy_violation",
            "message": "blocked by monitor",
            "param": None,
        },
    )


def _nested_error_event() -> tuple[str, dict]:
    return (
        "error",
        {"type": "error", "sequence_number": 2, "error": _MISALIGNMENT_ERROR},
    )


def _unrelated_flat_error_event() -> tuple[str, dict]:
    return (
        "error",
        {
            "type": "error",
            "sequence_number": 2,
            "code": "server_error",
            "message": "temporary hiccup",
            "param": None,
        },
    )


# ---------------------------------------------------------------------------
# Provider / transport helpers
# ---------------------------------------------------------------------------


class _Hooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class _Coordinator:
    def __init__(self) -> None:
        self.hooks = _Hooks()


def _mount_transport(provider: OpenAIProvider, handler) -> None:
    provider._client = openai.AsyncOpenAI(
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _make_provider(*, streaming: bool, max_retries: int = 2) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "use_streaming": streaming,
            "max_retries": max_retries,
            "min_retry_delay": 0.01,
            "max_retry_delay": 0.02,
            "default_model": "gpt-6-sol",
        },
    )


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="hi")])


def _input_tokens_response() -> httpx.Response:
    return httpx.Response(
        200, json={"object": "response.input_tokens", "input_tokens": 5}
    )


def _stream_events_handler(events: list[tuple[str, dict]], calls: list[int]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        calls.append(1)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(events),
        )

    return handler


def _status_handler(status: int, body: dict, calls: list[int]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        calls.append(1)
        return httpx.Response(status, json=body)

    return handler


# ---------------------------------------------------------------------------
# 1. Pre-stream HTTP 403 -> ContentFilterError, real status_code, no retry
# ---------------------------------------------------------------------------


def test_prestream_403_misalignment_is_content_filter_error_streaming():
    calls: list[int] = []
    provider = _make_provider(streaming=True)
    _mount_transport(
        provider, _status_handler(403, {"error": _MISALIGNMENT_ERROR}, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    assert err.retryable is False
    assert err.status_code == 403
    assert err.provider == "openai"
    assert len(calls) == 1, "must not retry a misalignment policy stop"
    assert "sk-" not in str(err)


def test_prestream_403_misalignment_is_content_filter_error_non_streaming():
    """Same pre-stream 403, but with the blocking (non-streaming) create() path."""
    calls: list[int] = []
    provider = _make_provider(streaming=False)
    _mount_transport(
        provider, _status_handler(403, {"error": _MISALIGNMENT_ERROR}, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 403
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2. response.failed AFTER a tool-call block start (real SSE bytes)
# ---------------------------------------------------------------------------


def test_sse_response_failed_after_tool_call_start_aborts_and_does_not_retry():
    calls: list[int] = []
    provider = _make_provider(streaming=True)
    events = [_created_event(), _function_call_added_event(), _failed_event()]
    _mount_transport(provider, _stream_events_handler(events, calls))
    coordinator = _Coordinator()
    provider.coordinator = coordinator

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    # status_code must NOT be fabricated for a mid-stream detection.
    assert exc_info.value.status_code is None
    assert len(calls) == 1, "must not retry a misalignment policy stop"

    names = [name for name, _ in coordinator.hooks.events]
    assert "llm:stream_block_start" in names
    assert "llm:stream_aborted" in names, (
        "a tool-call block opened and never closed must abort even though "
        "no text/reasoning delta was ever emitted"
    )
    # Balanced: exactly one start, no synthesized end, exactly one abort.
    assert names.count("llm:stream_block_start") == 1
    assert names.count("llm:stream_block_end") == 0
    assert names.count("llm:stream_aborted") == 1


def test_sse_response_failed_no_coordinator_path_still_classifies():
    """The no-hooks streaming branch (no coordinator mounted) must still
    classify and raise -- it doesn't get a free pass just because it emits
    no UI events."""
    calls: list[int] = []
    provider = _make_provider(streaming=True)
    events = [_created_event(), _function_call_added_event(), _failed_event()]
    _mount_transport(provider, _stream_events_handler(events, calls))
    provider.coordinator = None

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 3. Flat SSE `error` event -> ContentFilterError, no retry
# ---------------------------------------------------------------------------


def test_sse_flat_error_event_is_content_filter_error_not_retried():
    calls: list[int] = []
    provider = _make_provider(streaming=True)
    events = [_created_event(), _function_call_added_event(), _flat_error_event()]
    _mount_transport(provider, _stream_events_handler(events, calls))
    coordinator = _Coordinator()
    provider.coordinator = coordinator

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code is None
    assert len(calls) == 1

    names = [name for name, _ in coordinator.hooks.events]
    assert "llm:stream_aborted" in names


# ---------------------------------------------------------------------------
# 4. Bare-APIError-body path via a malformed nested SSE "error" event
# ---------------------------------------------------------------------------


def test_sse_error_event_with_nested_error_key_is_rejected_by_sdk_and_still_classified():
    """The real openai-python SDK does NOT accept `{"type": "error", "error":
    {...}}` as a well-formed `error` SSE event -- its typed event model wants
    the error fields flat on the event itself (see `_flat_error_event()`
    above), and this nested shape fails that parsing. Verified directly
    against the real SDK: iterating the stream never yields an
    `event.type == "error"` for it at all; the SDK instead unwraps the inner
    `error` dict and raises a bare `openai.APIError` from stream
    iteration/exit, with that inner dict as `e.body` and its `code` also set
    as `e.code`.

    So this test does NOT exercise this module's own `elif et == "error":`
    event-handling branch (`_flat_error_event()`'s test already covers that).
    It instead exercises the SAME `exc=e` fallback used for a pre-stream
    APIStatusError -- reading `e.body`/`e.code` off a bare SDK exception
    after normal iteration ends -- and is a real-world-shaped regression
    guard for that path, not a second exercise of the flat-event branch.
    """
    calls: list[int] = []
    provider = _make_provider(streaming=True)
    events = [_created_event(), _function_call_added_event(), _nested_error_event()]
    _mount_transport(provider, _stream_events_handler(events, calls))
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 5. Non-stream response body with status="failed" (HTTP 200)
# ---------------------------------------------------------------------------


def test_non_streaming_status_failed_body_is_content_filter_error():
    """A genuine HTTP 200 whose JSON body says status='failed' with the
    misalignment code must not be handed back as a completed response."""
    calls: list[int] = []
    provider = _make_provider(streaming=False)
    failed_body = dict(
        _BASE_RESPONSE,
        status="failed",
        error={"code": "misalignment_policy_violation", "message": "blocked"},
    )
    _mount_transport(provider, _status_handler(200, failed_body, calls))
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code is None
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 6 & 7. Continuation and truncation-retry call sites: 403 and status=failed
# ---------------------------------------------------------------------------


def _incomplete_response(output_item: dict) -> dict:
    return dict(
        _BASE_RESPONSE,
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        output=[output_item],
        usage={
            "input_tokens": 5,
            "output_tokens": 5,
            "total_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        service_tier="default",
    )


_INCOMPLETE_MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "incomplete",
    "content": [{"type": "output_text", "text": "partial", "annotations": []}],
}

_INCOMPLETE_FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "bash",
    "arguments": '{"cmd":',
    "status": "incomplete",
}


def _second_call_handler(first_output: dict, second_response_factory, calls: list[bool]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        body = json.loads(request.content)
        calls.append(bool(body.get("stream")))
        if len(calls) == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(
                    [_created_event(), (
                        "response.incomplete",
                        {
                            "type": "response.incomplete",
                            "sequence_number": 1,
                            "response": _incomplete_response(first_output),
                        },
                    )]
                ),
            )
        return second_response_factory()

    return handler


def test_continuation_call_403_misalignment_is_not_returned_as_partial_success():
    calls: list[bool] = []
    provider = _make_provider(streaming=True, max_retries=2)
    provider.max_output_tokens = 1000

    def second_call():
        return httpx.Response(403, json={"error": _MISALIGNMENT_ERROR})

    _mount_transport(
        provider, _second_call_handler(_INCOMPLETE_MESSAGE, second_call, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert calls == [True, False]


def test_continuation_call_status_failed_is_not_returned_as_partial_success():
    calls: list[bool] = []
    provider = _make_provider(streaming=True, max_retries=2)
    provider.max_output_tokens = 1000

    def second_call():
        return httpx.Response(
            200,
            json=dict(
                _BASE_RESPONSE,
                status="failed",
                error={
                    "code": "misalignment_policy_violation",
                    "message": "blocked",
                },
            ),
        )

    _mount_transport(
        provider, _second_call_handler(_INCOMPLETE_MESSAGE, second_call, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert calls == [True, False]


def test_truncation_retry_403_misalignment_does_not_leak_raw_sdk_exception():
    calls: list[bool] = []
    provider = _make_provider(streaming=True, max_retries=2)
    provider.max_output_tokens = 1000

    def second_call():
        return httpx.Response(403, json={"error": _MISALIGNMENT_ERROR})

    _mount_transport(
        provider, _second_call_handler(_INCOMPLETE_FUNCTION_CALL, second_call, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert calls == [True, False]


def test_truncation_retry_status_failed_is_not_returned_as_partial_success():
    calls: list[bool] = []
    provider = _make_provider(streaming=True, max_retries=2)
    provider.max_output_tokens = 1000

    def second_call():
        return httpx.Response(
            200,
            json=dict(
                _BASE_RESPONSE,
                status="failed",
                error={
                    "code": "misalignment_policy_violation",
                    "message": "blocked",
                },
            ),
        )

    _mount_transport(
        provider, _second_call_handler(_INCOMPLETE_FUNCTION_CALL, second_call, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert calls == [True, False]


# ---------------------------------------------------------------------------
# 8. Negative controls: unrelated errors keep their prior classification
# ---------------------------------------------------------------------------


def test_negative_control_server_error_still_retryable():
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=2)
    _mount_transport(
        provider,
        _status_handler(500, {"error": {"message": "boom", "type": "server_error"}}, calls),
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is True
    assert len(calls) == 3, "transient 5xx must still retry up to max_retries+1"


def test_negative_control_unrelated_403_stays_access_denied():
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=2)
    _mount_transport(
        provider,
        _status_handler(
            403,
            {"error": {"code": "insufficient_quota", "message": "no access", "type": "invalid_request_error"}},
            calls,
        ),
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.AccessDeniedError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert not isinstance(exc_info.value, kernel_errors.ContentFilterError)
    assert len(calls) == 1


def test_negative_control_cloudflare_403_stays_transient():
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        calls.append(1)
        return httpx.Response(
            403,
            headers={"content-type": "text/html"},
            content=b"<html><title>Just a moment...</title>Cloudflare</html>",
        )

    _mount_transport(provider, handler)
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is True
    assert len(calls) == 2


def test_negative_control_flat_sse_error_unrelated_code_not_classified_as_misalignment():
    """A flat SSE `error` event with a different code must keep its prior
    (unclassified/generic) fallthrough behavior -- only the misalignment
    code is intercepted here."""
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=0)
    events = [_created_event(), _unrelated_flat_error_event()]
    _mount_transport(provider, _stream_events_handler(events, calls))
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert not isinstance(exc_info.value, kernel_errors.ContentFilterError)


# ---------------------------------------------------------------------------
# 9. Stream lifecycle balance: message block with no delta still no-aborts
# ---------------------------------------------------------------------------


def test_message_block_started_no_delta_then_error_does_not_abort():
    """Regression guard: a text/reasoning block that only ever got a
    block_start (no delta) before an unrelated mid-stream failure must NOT
    emit llm:stream_aborted -- only a dangling TOOL-CALL block (or a prior
    delta) does. This must hold even with the new open-block tracking."""
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        calls.append(1)
        # A message block_start, then a truncated/invalid stream (no proper
        # SSE terminal) -- the SDK will fail with a generic error once the
        # connection ends without a completed/incomplete/failed terminal.
        events = [
            _created_event(),
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
            ),
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(events),
        )

    _mount_transport(provider, handler)
    coordinator = _Coordinator()
    provider.coordinator = coordinator

    with pytest.raises(kernel_errors.LLMError):
        asyncio.run(provider.complete(_simple_request()))

    names = [name for name, _ in coordinator.hooks.events]
    assert "llm:stream_block_start" in names
    assert "llm:stream_aborted" not in names, (
        "a non-tool-call block with no delta must not trigger an abort"
    )


# ---------------------------------------------------------------------------
# 10. Unexpected HTTP status carrying the exact code: 429 and 401
# ---------------------------------------------------------------------------


def test_429_with_exact_misalignment_code_is_content_filter_error_not_retried():
    """An unexpected HTTP 429 carrying the exact misalignment code must be
    classified as the policy stop, not retried as an ordinary rate limit."""
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=2)
    _mount_transport(
        provider, _status_handler(429, {"error": _MISALIGNMENT_ERROR}, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert len(calls) == 1, "must not retry a misalignment policy stop"


def test_429_without_misalignment_code_stays_rate_limit_error_and_retries():
    """Negative control: an ordinary 429 (no misalignment code) must keep its
    prior RateLimitError/retryable=True classification unchanged."""
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=2)
    _mount_transport(
        provider,
        _status_handler(
            429,
            {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "too many requests",
                    "type": "rate_limit_error",
                }
            },
            calls,
        ),
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert not isinstance(exc_info.value, kernel_errors.ContentFilterError)
    assert exc_info.value.retryable is True
    assert len(calls) == 3, "an ordinary rate limit must still retry up to max_retries+1"


def test_401_with_exact_misalignment_code_is_content_filter_error():
    """An unexpected HTTP 401 carrying the exact misalignment code must be
    classified as the policy stop, not mislabeled as an authentication
    failure."""
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=2)
    _mount_transport(
        provider, _status_handler(401, {"error": _MISALIGNMENT_ERROR}, calls)
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is False
    assert len(calls) == 1


def test_401_without_misalignment_code_stays_authentication_error():
    """Negative control: an ordinary 401 (no misalignment code) must keep its
    prior AuthenticationError classification unchanged."""
    calls: list[int] = []
    provider = _make_provider(streaming=True, max_retries=2)
    _mount_transport(
        provider,
        _status_handler(
            401,
            {
                "error": {
                    "code": "invalid_api_key",
                    "message": "bad key",
                    "type": "invalid_request_error",
                }
            },
            calls,
        ),
    )
    provider.coordinator = _Coordinator()

    with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert not isinstance(exc_info.value, kernel_errors.ContentFilterError)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 11. Background (deep-research) polling: queued -> failed with the code
# ---------------------------------------------------------------------------


def test_background_polled_failed_with_misalignment_code_is_content_filter_error():
    """A background response that starts queued, is polled once, and comes
    back status='failed' with the exact misalignment code must raise
    ContentFilterError -- not a generic RuntimeError -- with exactly one
    generation call and one poll, and no replacement/retry generation."""
    calls = {"generate": 0, "poll": 0}
    response_id = "resp_bg_1"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        if request.url.path.endswith(f"/responses/{response_id}"):
            calls["poll"] += 1
            failed_body = dict(
                _BASE_RESPONSE,
                id=response_id,
                status="failed",
                error={
                    "code": "misalignment_policy_violation",
                    "message": "blocked by monitor",
                },
            )
            return httpx.Response(200, json=failed_body)
        calls["generate"] += 1
        queued_body = dict(_BASE_RESPONSE, id=response_id, status="queued")
        return httpx.Response(200, json=queued_body)

    provider = _make_provider(streaming=False, max_retries=0)
    provider.config["poll_interval"] = 0.01
    provider.poll_interval = 0.01
    _mount_transport(provider, handler)
    provider.coordinator = _Coordinator()

    request = _simple_request()
    request.metadata = {"stream": False}

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(request, background=True))

    assert exc_info.value.retryable is False
    assert calls == {"generate": 1, "poll": 1}


def test_background_initial_status_failed_with_misalignment_code_is_classified_before_generic_block():
    """An initial background response ALREADY status='failed' (poll loop
    never entered, poll_count stays 0) must still classify -- proving the
    check runs before the generic 'Background request failed after N polls'
    RuntimeError block, not only inside the polling loop."""
    calls = {"generate": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/input_tokens"):
            return _input_tokens_response()
        calls["generate"] += 1
        failed_body = dict(
            _BASE_RESPONSE,
            id="resp_bg_2",
            status="failed",
            error={
                "code": "misalignment_policy_violation",
                "message": "blocked by monitor",
            },
        )
        return httpx.Response(200, json=failed_body)

    provider = _make_provider(streaming=False, max_retries=0)
    _mount_transport(provider, handler)
    provider.coordinator = _Coordinator()

    request = _simple_request()
    request.metadata = {"stream": False}

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(request, background=True))

    assert exc_info.value.retryable is False
    assert calls == {"generate": 1}
