"""Tests for json.dumps(e.body) in KernelLLMError constructors.

Verifies that when OpenAI SDK errors carry a structured `body` dict,
the kernel error message contains the JSON-serialized body rather than
the SDK's str(e) representation.

Keyword matching in BadRequestError must still work (using str(e).lower()).
"""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    """Create a provider with retries disabled so errors propagate immediately."""
    config = {"max_retries": 0, **config_overrides}
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


def _mock_httpx_response(
    status_code: int = 429, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
    )


RATE_LIMIT_BODY = {
    "error": {
        "message": "Rate limit exceeded for model",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
    }
}

AUTH_BODY = {
    "error": {
        "message": "Incorrect API key provided",
        "type": "authentication_error",
        "code": "invalid_api_key",
    }
}

CONTEXT_LENGTH_BODY = {
    "error": {
        "message": "This model's maximum context length is 128000 tokens",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
    }
}

CONTENT_FILTER_BODY = {
    "error": {
        "message": "Content filter triggered by safety system",
        "type": "invalid_request_error",
        "code": "content_filter",
    }
}

INVALID_REQUEST_BODY = {
    "error": {
        "message": "Invalid parameter: temperature must be between 0 and 2",
        "type": "invalid_request_error",
        "code": "invalid_parameter",
    }
}

SERVER_ERROR_BODY = {
    "error": {
        "message": "Internal server error",
        "type": "server_error",
    }
}


# ---------------------------------------------------------------------------
# Block 1a: RateLimitError uses json.dumps(body)
# ---------------------------------------------------------------------------


def test_rate_limit_error_message_uses_json_body():
    """RateLimitError message should be json.dumps(body) when body is present."""
    provider = _make_provider()
    native = openai.RateLimitError(
        "Rate limit exceeded",
        response=_mock_httpx_response(429),
        body=RATE_LIMIT_BODY,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    err = exc_info.value
    expected = json.dumps(RATE_LIMIT_BODY)
    assert str(err) == expected, f"Expected {expected!r}, got {str(err)!r}"


def test_rate_limit_error_message_falls_back_to_str_when_no_body():
    """RateLimitError message should fall back to str(e) when body is None."""
    provider = _make_provider()
    native = openai.RateLimitError(
        "Rate limit exceeded",
        response=_mock_httpx_response(429),
        body=None,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    # Should not be empty; should contain the original message
    assert str(exc_info.value) != ""


# ---------------------------------------------------------------------------
# Block 1b: AuthenticationError uses json.dumps(body)
# ---------------------------------------------------------------------------


def test_authentication_error_message_uses_json_body():
    """AuthenticationError message should be json.dumps(body) when body present."""
    provider = _make_provider()
    native = openai.AuthenticationError(
        "Invalid API key",
        response=_mock_httpx_response(401),
        body=AUTH_BODY,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(AUTH_BODY)
    assert str(exc_info.value) == expected


# ---------------------------------------------------------------------------
# Block 1c: BadRequestError uses json.dumps(body) but keyword matching intact
# ---------------------------------------------------------------------------


def test_bad_request_context_length_uses_json_body():
    """ContextLengthError message is json.dumps(body); keyword match still works."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "This model's maximum context length is 128000 tokens",
        response=_mock_httpx_response(400),
        body=CONTEXT_LENGTH_BODY,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(CONTEXT_LENGTH_BODY)
    assert str(exc_info.value) == expected


def test_bad_request_content_filter_uses_json_body():
    """ContentFilterError message is json.dumps(body); keyword match still works."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "Your request was rejected as a result of our safety system. Content filter triggered.",
        response=_mock_httpx_response(400),
        body=CONTENT_FILTER_BODY,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContentFilterError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(CONTENT_FILTER_BODY)
    assert str(exc_info.value) == expected


def test_bad_request_invalid_request_uses_json_body():
    """InvalidRequestError message is json.dumps(body)."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "Invalid parameter: temperature must be between 0 and 2",
        response=_mock_httpx_response(400),
        body=INVALID_REQUEST_BODY,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.InvalidRequestError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(INVALID_REQUEST_BODY)
    assert str(exc_info.value) == expected


def test_bad_request_keyword_matching_still_works_with_maximum_context():
    """'maximum context' keyword in str(e) still triggers ContextLengthError."""
    provider = _make_provider()
    native = openai.BadRequestError(
        "maximum context length exceeded for this model",
        response=_mock_httpx_response(400),
        body={"error": {"message": "maximum context length exceeded"}},
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ContextLengthError):
        asyncio.run(provider.complete(_simple_request()))


# ---------------------------------------------------------------------------
# Block 1d: APIStatusError uses json.dumps(body)
# ---------------------------------------------------------------------------


def test_api_status_403_uses_json_body():
    """AccessDeniedError message is json.dumps(body)."""
    provider = _make_provider()
    body = {"error": {"message": "Permission denied", "type": "forbidden"}}
    native = openai.APIStatusError(
        "permission denied",
        response=_mock_httpx_response(403),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.AccessDeniedError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(body)
    assert str(exc_info.value) == expected


def test_api_status_404_uses_json_body():
    """NotFoundError message is json.dumps(body)."""
    provider = _make_provider()
    body = {"error": {"message": "Model not found", "type": "not_found"}}
    native = openai.APIStatusError(
        "model not found",
        response=_mock_httpx_response(404),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.NotFoundError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(body)
    assert str(exc_info.value) == expected


def test_api_status_500_uses_json_body():
    """ProviderUnavailableError message is json.dumps(body)."""
    provider = _make_provider()
    native = openai.APIStatusError(
        "Internal server error",
        response=_mock_httpx_response(500),
        body=SERVER_ERROR_BODY,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(SERVER_ERROR_BODY)
    assert str(exc_info.value) == expected


def test_api_status_other_uses_json_body():
    """Generic LLMError for unhandled status codes uses json.dumps(body)."""
    provider = _make_provider()
    body = {"error": {"message": "Payment required", "type": "billing_error"}}
    native = openai.APIStatusError(
        "payment required",
        response=_mock_httpx_response(402),
        body=body,
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(body)
    assert str(exc_info.value) == expected


# ---------------------------------------------------------------------------
# Block 1g: Generic Exception catch-all uses body pattern with fallback
# ---------------------------------------------------------------------------


def test_generic_exception_with_body_uses_json_body():
    """Generic Exception with body attr uses json.dumps(body)."""
    provider = _make_provider()

    class CustomError(Exception):
        def __init__(self, msg, body):
            super().__init__(msg)
            self.body = body

    body = {"detail": "unexpected issue"}
    native = CustomError("Something broke", body)
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    expected = json.dumps(body)
    assert str(exc_info.value) == expected


def test_generic_exception_without_body_falls_back_to_str():
    """Generic Exception without body falls back to str(e)."""
    provider = _make_provider()
    native = RuntimeError("Something unexpected")
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert "Something unexpected" in str(exc_info.value)


def test_generic_exception_with_none_body_falls_back_to_str():
    """Generic Exception with body=None falls back to str(e)."""
    provider = _make_provider()

    class CustomError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            self.body = None

    native = CustomError("Some error")
    provider.client.responses.create = AsyncMock(side_effect=native)

    with pytest.raises(kernel_errors.LLMError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert "Some error" in str(exc_info.value)
