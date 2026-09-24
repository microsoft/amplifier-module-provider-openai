"""Regression tests: a misalignment-monitoring policy stop must classify
consistently as a non-retryable `ContentFilterError` (HTTP 403), across all
three documented wire shapes:

1. Pre-stream / non-streaming: HTTP 403, `openai.APIStatusError`.
2. Mid-stream `response.failed` terminal (no `response.completed` event).
3. Mid-stream bare/flat SSE error: `openai.APIError` with no `status_code`.

Per https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring
(verified 2026-09-24): misalignment monitoring can stop a covered model's
conversation -- before streaming begins, or mid-stream even after output was
already emitted -- with error type "invalid_request_error" and code
"misalignment_policy_violation". This is NOT GPT-6-specific: the fix lives in
the generic error-classification/streaming-response handling shared by every
model, so these tests intentionally exercise it against a plain gpt-5.6-sol
provider (no GPT-6-specific setup) to prove the classification is generic.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider

sys.path.insert(0, str(Path(__file__).parent))
from _stream_fakes import CompletedStream as _CompletedStream
from _stream_fakes import StreamContext as _StreamContext
from _stream_fakes import TerminalFailedStream as _MisalignmentFailedStream


def _provider(**config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="[REDACTED:SECRET]",
        config={
            "default_model": "gpt-5.6-sol",
            "max_retries": 1,
            "min_retry_delay": 0,
            "max_retry_delay": 0,
            "retry_jitter": False,
            "use_streaming": True,
            **config,
        },
    )


def _request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _misalignment_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_misaligned",
        model="gpt-5.6-sol",
        status="failed",
        error=SimpleNamespace(
            code="misalignment_policy_violation",
            message="Misalignment monitoring stopped this conversation.",
        ),
    )


def test_misalignment_policy_stop_mid_stream_is_not_retried() -> None:
    provider = _provider()
    stream = _MisalignmentFailedStream(_misalignment_response())
    provider.client.responses.stream = MagicMock(
        return_value=_StreamContext(stream)
    )

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_request()))

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 403
    assert "misalignment" in str(exc_info.value).lower()
    # Only ONE attempt -- proves this did not exhaust the retry loop.
    assert provider.client.responses.stream.call_count == 1


def test_unrelated_response_failed_still_falls_through_to_generic_retryable() -> None:
    """A `response.failed` WITHOUT the misalignment code is unaffected --
    it keeps the pre-existing (retryable) classification. Uses max_retries=1
    so the second attempt succeeds and the call completes."""
    other_failure = SimpleNamespace(
        id="resp_other_fail",
        model="gpt-5.6-sol",
        status="failed",
        error=SimpleNamespace(code="server_error", message="transient"),
    )
    succeeded = SimpleNamespace(
        id="resp_ok",
        model="gpt-5.6-sol",
        status="completed",
        service_tier="default",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ],
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        ),
    )

    provider = _provider()
    provider.client.responses.stream = MagicMock(
        side_effect=[
            _StreamContext(_MisalignmentFailedStream(other_failure)),
            _StreamContext(_CompletedStream(succeeded)),
        ]
    )

    result = asyncio.run(provider.complete(_request()))

    assert result.usage.output_tokens == 5
    assert provider.client.responses.stream.call_count == 2


MISALIGNMENT_BODY = {
    "error": {
        "type": "invalid_request_error",
        "code": "misalignment_policy_violation",
        "message": "Misalignment monitoring stopped this conversation.",
    }
}


def test_misalignment_policy_stop_pre_stream_http_403_is_content_filter_error() -> None:
    """A documented pre-stream/non-streaming misalignment stop arrives as a
    plain HTTP 403 `openai.APIStatusError` -- classify it the SAME way as the
    mid-stream shapes: non-retryable `ContentFilterError`, status_code=403.
    Uses use_streaming=False so this exercises the non-streaming create path.
    """
    provider = _provider(use_streaming=False)
    response = httpx.Response(
        403,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )
    native = openai.APIStatusError(
        "Forbidden", response=response, body=MISALIGNMENT_BODY
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_request()))

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 403
    assert exc_info.value.provider == "openai"
    assert exc_info.value.__cause__ is native
    provider.client.responses.create.assert_awaited_once()


def test_misalignment_policy_stop_bare_mid_stream_api_error_is_content_filter_error() -> None:
    """A misalignment stop delivered as a bare/flat SSE `openai.APIError`
    (HTTP 200 already sent, no `status_code` attribute at all -- the shape
    the SDK raises for an SSE 'error' event) must classify identically:
    non-retryable `ContentFilterError` with status_code=403 preserved even
    though the native exception carries none."""
    provider = _provider(use_streaming=False)
    native = openai.APIError(
        "Misalignment monitoring stopped this conversation.",
        httpx.Request("POST", "https://api.openai.com/v1/responses"),
        body=MISALIGNMENT_BODY["error"],
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_request()))

    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 403
    assert exc_info.value.provider == "openai"
    assert exc_info.value.__cause__ is native
    provider.client.responses.create.assert_awaited_once()


def test_unrelated_bare_api_error_still_falls_through_to_invalid_request_error() -> None:
    """A bare APIError with an unrelated invalid_request_error code keeps its
    existing classification (InvalidRequestError, not ContentFilterError)."""
    provider = _provider(use_streaming=False)
    native = openai.APIError(
        "Invalid value for parameter",
        httpx.Request("POST", "https://api.openai.com/v1/responses"),
        body={
            "type": "invalid_request_error",
            "code": "invalid_value",
            "message": "Invalid value for parameter",
        },
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.InvalidRequestError):
        asyncio.run(provider.complete(_request()))
