"""Retry behavior tests for list_models().

Verifies that list_models() uses the same shared retry_with_backoff()/
_retry_config machinery as complete(): transient failures (5xx) are
retried with backoff, non-retryable failures (401) raise immediately,
and persistent transient failures raise the translated kernel error
once retries are exhausted.

See test_retry.py for the equivalent tests on the complete() path --
this file mirrors that call shape for list_models().
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from amplifier_core import llm_errors as kernel_errors
from amplifier_module_provider_openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {
        "max_retries": 3,
        "min_retry_delay": 0.01,
        "max_retry_delay": 1.0,
        **config_overrides,
    }
    return OpenAIProvider(api_key="test-key", config=config)


def _fake_models_response(model_ids: list[str]):
    """Create a fake OpenAI models.list() response."""
    data = [SimpleNamespace(id=mid) for mid in model_ids]
    return SimpleNamespace(data=data)


def _mock_httpx_response(
    status_code: int = 500, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.openai.com/v1/models"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_models_succeeds_first_try():
    """No transient failure: exactly one API call, result unchanged."""
    provider = _make_provider()
    response = _fake_models_response(["gpt-5.1"])
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(return_value=response)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        models = asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 1
    mock_sleep.assert_not_awaited()
    assert len(models) == 1
    assert models[0].id == "gpt-5.1"


def test_list_models_recovers_from_transient_500():
    """A single transient 500 is retried, then the call succeeds."""
    provider = _make_provider()
    native_500 = openai.APIStatusError(
        "Server error",
        response=_mock_httpx_response(500),
        body=None,
    )
    response = _fake_models_response(["gpt-5.1"])
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=[native_500, response])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        models = asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 2
    assert len(models) == 1
    assert models[0].id == "gpt-5.1"


def test_list_models_raises_after_retries_exhausted():
    """Persistent transient failure raises the kernel error after retries."""
    provider = _make_provider(max_retries=2)
    native_500 = openai.APIStatusError(
        "Server error",
        response=_mock_httpx_response(500),
        body=None,
    )
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=native_500)

    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(kernel_errors.ProviderUnavailableError),
    ):
        asyncio.run(provider.list_models())

    # 1 initial + 2 retries = 3 total attempts
    assert provider.client.models.list.await_count == 3


def test_list_models_non_retryable_error_raised_immediately():
    """A non-retryable error (401) raises immediately without retrying."""
    provider = _make_provider()
    native_401 = openai.AuthenticationError(
        "Invalid key",
        response=_mock_httpx_response(401),
        body=None,
    )
    provider._client = AsyncMock()
    provider._client.models.list = AsyncMock(side_effect=native_401)

    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        pytest.raises(kernel_errors.AuthenticationError),
    ):
        asyncio.run(provider.list_models())

    assert provider.client.models.list.await_count == 1
    mock_sleep.assert_not_awaited()
