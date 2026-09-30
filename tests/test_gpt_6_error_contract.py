"""Deterministic GPT-6 (Astra/Sol/Luna) error-contract coverage.

Exercises the exact HTTP-error -> kernel-error translation for every
verified GPT-6 family model across complete() and list_models(): 401, an
ordinary (non-Cloudflare, non-misalignment) 403, a 404 "model not found"
(the shape a caller sees for an invented/unavailable model ID), 429 with
`Retry-After`, 503 with `Retry-After`, the oversized-`Retry-After`
fail-fast path, and no-header compatibility (missing header still
succeeds/retries with `retry_after=None`).

This module does not add cross-provider fallback logic -- this repo only
translates provider errors into kernel error types/attributes
(`retryable`, `retry_after`); an external orchestrator decides what to do
with those attributes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors

from amplifier_module_provider_openai import OpenAIProvider, _parse_retry_after_seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GPT6_MODELS = ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]


def _make_provider(model: str, **config_overrides) -> OpenAIProvider:
    config = {
        "default_model": model,
        "max_retries": 3,
        "min_retry_delay": 0.01,
        "max_retry_delay": 1.0,
        "use_streaming": False,
        **config_overrides,
    }
    return OpenAIProvider(api_key="[REDACTED:SECRET]", config=config)


def _httpx_response(
    method: str, url: str, status_code: int, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request(method, url),
    )


def _complete_response(status_code: int, headers: dict | None = None) -> httpx.Response:
    return _httpx_response(
        "POST", "https://api.openai.com/v1/responses", status_code, headers
    )


def _list_models_response(
    status_code: int, headers: dict | None = None
) -> httpx.Response:
    return _httpx_response(
        "GET", "https://api.openai.com/v1/models", status_code, headers
    )


def _body(code: str, message: str = "an error occurred") -> dict:
    return {"error": {"code": code, "message": message}}


async def _run_complete(provider: OpenAIProvider):
    from amplifier_core.message_models import ChatRequest, Message

    return await provider.complete(
        ChatRequest(messages=[Message(role="user", content="Hello")])
    )


# ===========================================================================
# complete() -- non-retryable classification errors
# ===========================================================================


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_401_raises_authentication_error(model: str) -> None:
    provider = _make_provider(model, max_retries=0)
    native = openai.AuthenticationError(
        "Invalid API key",
        response=_complete_response(401),
        body=_body("invalid_api_key"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.AuthenticationError) as exc_info:
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 1
    e = exc_info.value
    assert e.provider == "openai"
    assert e.status_code == 401
    assert e.retryable is False


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_ordinary_403_raises_access_denied_error(model: str) -> None:
    """A structured (JSON-body) 403 that is NOT the misalignment code and
    NOT a Cloudflare HTML challenge is an ordinary access-denied error."""
    provider = _make_provider(model, max_retries=0)
    native = openai.APIStatusError(
        "insufficient permissions",
        response=_complete_response(403, headers={"content-type": "application/json"}),
        body=_body("insufficient_permissions"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.AccessDeniedError) as exc_info:
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 1
    e = exc_info.value
    assert e.provider == "openai"
    assert e.status_code == 403
    assert e.retryable is False


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_404_unavailable_model_raises_not_found_error(model: str) -> None:
    """The shape a caller sees when the requested (e.g. invented) GPT-6
    model ID does not exist / is not available to the account."""
    provider = _make_provider(model, max_retries=0)
    native = openai.APIStatusError(
        "model not found",
        response=_complete_response(404),
        body=_body("model_not_found", message=f"The model `{model}` does not exist"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.NotFoundError) as exc_info:
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 1
    e = exc_info.value
    assert e.provider == "openai"
    assert e.status_code == 404
    assert e.retryable is False


# ===========================================================================
# complete() -- 429 slow_down with Retry-After
# ===========================================================================


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_429_slow_down_honors_retry_after_and_retries(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_complete_response(429, headers={"retry-after": "2"}),
        body=_body("slow_down", message="Slow down"),
    )

    import asyncio
    from unittest.mock import patch

    from tests._gpt6_streaming_fixtures import make_completed_response

    success = make_completed_response(model=model)
    provider.client.responses.create = AsyncMock(side_effect=[native, success])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 2


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_429_oversized_retry_after_fails_fast(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=10.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_complete_response(429, headers={"retry-after": "120"}),
        body=_body("slow_down"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 1
    e = exc_info.value
    assert e.status_code == 429
    assert e.retryable is False
    assert e.retry_after == 120.0


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_429_no_header_still_retries_with_none_retry_after(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_complete_response(429),
        body=_body("slow_down"),
    )

    import asyncio
    from unittest.mock import patch

    from tests._gpt6_streaming_fixtures import make_completed_response

    success = make_completed_response(model=model)
    provider.client.responses.create = AsyncMock(side_effect=[native, success])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 2


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_429_azure_ms_header_is_parsed(model: str) -> None:
    """Azure's `x-ms-retry-after-ms` header is honored the same way."""
    provider = _make_provider(model, max_retries=3, max_retry_delay=1.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_complete_response(429, headers={"x-ms-retry-after-ms": "5000"}),
        body=_body("slow_down"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(_run_complete(provider))

    # 5000ms == 5.0s > max_retry_delay(1.0) -> fail fast, single attempt.
    assert provider.client.responses.create.await_count == 1
    assert exc_info.value.retry_after == 5.0
    assert exc_info.value.retryable is False


# ===========================================================================
# complete() -- 503 server_is_overloaded with Retry-After
# ===========================================================================


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_503_server_is_overloaded_honors_retry_after_and_retries(
    model: str,
) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.APIStatusError(
        "Server overloaded",
        response=_complete_response(503, headers={"retry-after": "1"}),
        body=_body("server_is_overloaded", message="Server overloaded"),
    )

    import asyncio
    from unittest.mock import patch

    from tests._gpt6_streaming_fixtures import make_completed_response

    success = make_completed_response(model=model)
    provider.client.responses.create = AsyncMock(side_effect=[native, success])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 2


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_503_oversized_retry_after_fails_fast(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=10.0)
    native = openai.APIStatusError(
        "Server overloaded",
        response=_complete_response(503, headers={"retry-after": "300"}),
        body=_body("server_is_overloaded"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 1
    e = exc_info.value
    assert e.status_code == 503
    assert e.retryable is False
    assert e.retry_after == 300.0


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_503_no_header_still_retries_with_none_retry_after(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.APIStatusError(
        "Server overloaded",
        response=_complete_response(503),
        body=_body("server_is_overloaded"),
    )

    import asyncio
    from unittest.mock import patch

    from tests._gpt6_streaming_fixtures import make_completed_response

    success = make_completed_response(model=model)
    provider.client.responses.create = AsyncMock(side_effect=[native, success])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 2


# ===========================================================================
# list_models() -- Retry-After coverage
# ===========================================================================


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_list_models_429_honors_retry_after_and_retries(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_list_models_response(429, headers={"retry-after": "2"}),
        body=_body("slow_down"),
    )
    from types import SimpleNamespace

    success = SimpleNamespace(data=[SimpleNamespace(id=model)])
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=[native, success])

    import asyncio
    from unittest.mock import patch

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 2


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_list_models_429_oversized_retry_after_fails_fast(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=10.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_list_models_response(429, headers={"retry-after": "999"}),
        body=_body("slow_down"),
    )
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 1
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after == 999.0


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_list_models_503_honors_retry_after_and_retries(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.APIStatusError(
        "Server overloaded",
        response=_list_models_response(503, headers={"retry-after": "1"}),
        body=_body("server_is_overloaded"),
    )
    from types import SimpleNamespace

    success = SimpleNamespace(data=[SimpleNamespace(id=model)])
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=[native, success])

    import asyncio
    from unittest.mock import patch

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 2


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_list_models_503_oversized_retry_after_fails_fast(model: str) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=5.0)
    native = openai.APIStatusError(
        "Server overloaded",
        response=_list_models_response(503, headers={"retry-after": "30"}),
        body=_body("server_is_overloaded"),
    )
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.ProviderUnavailableError) as exc_info:
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 1
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after == 30.0


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_list_models_503_no_header_still_retries_with_none_retry_after(
    model: str,
) -> None:
    provider = _make_provider(model, max_retries=3, max_retry_delay=60.0)
    native = openai.APIStatusError(
        "Server overloaded",
        response=_list_models_response(503),
        body=_body("server_is_overloaded"),
    )
    from types import SimpleNamespace

    success = SimpleNamespace(data=[SimpleNamespace(id=model)])
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=[native, success])

    import asyncio
    from unittest.mock import patch

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 2


# ===========================================================================
# _parse_retry_after_seconds -- direct unit coverage for hardening
# ===========================================================================


class _Headers(dict):
    """Minimal header-like mapping exposing `.get`, matching real usage."""


def test_parse_retry_after_negative_standard_is_rejected() -> None:
    assert _parse_retry_after_seconds(_Headers({"retry-after": "-5"})) is None


def test_parse_retry_after_nan_standard_is_rejected() -> None:
    assert _parse_retry_after_seconds(_Headers({"retry-after": "nan"})) is None


def test_parse_retry_after_inf_standard_is_rejected() -> None:
    assert _parse_retry_after_seconds(_Headers({"retry-after": "inf"})) is None
    assert _parse_retry_after_seconds(_Headers({"retry-after": "-inf"})) is None


def test_parse_retry_after_negative_azure_ms_is_rejected() -> None:
    assert (
        _parse_retry_after_seconds(_Headers({"x-ms-retry-after-ms": "-1000"})) is None
    )


def test_parse_retry_after_nan_azure_ms_is_rejected() -> None:
    assert _parse_retry_after_seconds(_Headers({"x-ms-retry-after-ms": "nan"})) is None


def test_parse_retry_after_inf_azure_ms_is_rejected() -> None:
    assert _parse_retry_after_seconds(_Headers({"x-ms-retry-after-ms": "inf"})) is None


def test_parse_retry_after_invalid_standard_falls_back_to_valid_azure() -> None:
    """An HTTP-date (non-numeric) or otherwise invalid standard header must
    not shadow a usable Azure header -- fall through to it instead."""
    assert (
        _parse_retry_after_seconds(
            _Headers(
                {
                    "retry-after": "Wed, 21 Oct 2026 07:28:00 GMT",
                    "x-ms-retry-after-ms": "2500",
                }
            )
        )
        == 2.5
    )


def test_parse_retry_after_negative_standard_falls_back_to_valid_azure() -> None:
    assert (
        _parse_retry_after_seconds(
            _Headers({"retry-after": "-1", "x-ms-retry-after-ms": "3000"})
        )
        == 3.0
    )


def test_parse_retry_after_nan_standard_falls_back_to_valid_azure() -> None:
    assert (
        _parse_retry_after_seconds(
            _Headers({"retry-after": "nan", "x-ms-retry-after-ms": "1500"})
        )
        == 1.5
    )


def test_parse_retry_after_both_invalid_returns_none() -> None:
    assert (
        _parse_retry_after_seconds(
            _Headers({"retry-after": "nan", "x-ms-retry-after-ms": "-1"})
        )
        is None
    )


def test_parse_retry_after_no_headers_present_returns_none() -> None:
    assert _parse_retry_after_seconds(_Headers({})) is None


def test_parse_retry_after_none_headers_returns_none() -> None:
    assert _parse_retry_after_seconds(None) is None


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_429_invalid_standard_header_falls_back_to_valid_azure_header(
    model: str,
) -> None:
    """End-to-end: an HTTP-date `Retry-After` alongside a valid Azure
    `x-ms-retry-after-ms` header still yields the Azure-derived retry_after,
    not None -- the invalid standard header does not shadow it."""
    provider = _make_provider(model, max_retries=3, max_retry_delay=1.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_complete_response(
            429,
            headers={
                "retry-after": "Wed, 21 Oct 2026 07:28:00 GMT",
                "x-ms-retry-after-ms": "5000",
            },
        ),
        body=_body("slow_down"),
    )
    provider.client.responses.create = AsyncMock(side_effect=native)

    import asyncio

    with pytest.raises(kernel_errors.RateLimitError) as exc_info:
        asyncio.run(_run_complete(provider))

    # 5000ms == 5.0s > max_retry_delay(1.0) -> fail fast, single attempt.
    assert provider.client.responses.create.await_count == 1
    assert exc_info.value.retry_after == 5.0
    assert exc_info.value.retryable is False


@pytest.mark.parametrize("model", GPT6_MODELS)
def test_complete_429_invalid_only_headers_expose_retry_after_none(model: str) -> None:
    """A negative/non-finite standard header with no usable Azure fallback
    must expose `retry_after=None` on the raised kernel error, not a
    poisoned negative/nan/inf value, and must not fail fast (no usable
    wait hint to compare against max_retry_delay)."""
    provider = _make_provider(model, max_retries=3, max_retry_delay=1.0)
    native = openai.RateLimitError(
        "Slow down",
        response=_complete_response(429, headers={"retry-after": "nan"}),
        body=_body("slow_down"),
    )

    import asyncio
    from unittest.mock import patch

    from tests._gpt6_streaming_fixtures import make_completed_response

    success = make_completed_response(model=model)
    provider.client.responses.create = AsyncMock(side_effect=[native, success])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        asyncio.run(_run_complete(provider))

    assert provider.client.responses.create.await_count == 2
