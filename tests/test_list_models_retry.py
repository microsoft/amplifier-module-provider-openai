"""Provider retry policy through the real SDK; no nested SDK retry loop."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from amplifier_core import llm_errors as kernel_errors
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses, retries, expected_error", [
    ([200], 3, None),
    ([500, 200], 3, None),
    ([500, 500, 500], 2, kernel_errors.ProviderUnavailableError),
    ([429, 200], 3, None),
    ([401], 3, kernel_errors.AuthenticationError),
])
async def test_listing_retry_policy(statuses, retries, expected_error):
    requests = []
    coordinator = AsyncMock()

    def receive(request):
        assert request.method == "GET" and not request.url.query
        assert request.headers["authorization"] == "Bearer fixture"
        assert provider.client is client and client.max_retries == 2
        requests.append(request)
        status = statuses[len(requests) - 1]
        body = ({"object": "list", "data": [{"id": "gpt-6-sol"}]} if status == 200
                else {"error": {"message": "fixture-secret"}})
        return httpx.Response(status, json=body)

    async with AsyncOpenAI(api_key="fixture", max_retries=2, http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(receive), trust_env=False,
    )) as client:
        provider = OpenAIProvider(client=client, coordinator=coordinator, config={
            "max_retries": retries, "min_retry_delay": 0.01, "max_retry_delay": 1.0,
        })
        with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            if expected_error:
                with pytest.raises(expected_error):
                    await provider.list_models()
            else:
                assert [row.id for row in await provider.list_models()] == ["gpt-6-sol"]
        assert len(requests) == len(statuses)
        assert sleep.await_count == len(statuses) - 1
        assert coordinator.hooks.emit.await_count == len(statuses) - 1
        for call in coordinator.hooks.emit.call_args_list:
            assert "fixture-secret" not in str(call)
