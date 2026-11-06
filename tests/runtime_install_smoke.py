"""Exercise the installed wheel without dev dependencies or provider requests."""

import asyncio

import httpx
from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai import _single_attempt as bounded


async def main():
    provider = OpenAIProvider(api_key="synthetic-runtime-smoke")
    assert bounded.CAPABILITY in provider.get_info().capabilities
    assert provider.supports_native_compaction()
    transport = bounded._http_client(5)
    try:
        assert transport.follow_redirects is False
        client = AsyncOpenAI(
            api_key="synthetic-runtime-smoke",
            http_client=transport,
            max_retries=0,
        )
        try:
            assert client.max_retries == 0
            assert client._client is transport
        finally:
            await client.close()
    finally:
        await transport.aclose()
    assert transport.is_closed

    requests = []

    def catalog(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == "https://installed.example.invalid/v1/models"
        return httpx.Response(200, json={
            "object": "list",
            "data": [{"id": "installed-api-id", "object": "model",
                      "created": 1, "owned_by": "fixture"}],
        })

    async with AsyncOpenAI(
        api_key="synthetic-runtime-smoke",
        base_url="https://installed.example.invalid/v1/",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(catalog)),
        max_retries=0,
    ) as client:
        provider = OpenAIProvider(client=client)
        assert await provider.discover_models() == {
            "version": 1,
            "source": "provider-api",
            "models": [{"id": "installed-api-id", "display_name": "installed-api-id"}],
        }
    assert len(requests) == 1


if __name__ == "__main__":
    asyncio.run(main())
