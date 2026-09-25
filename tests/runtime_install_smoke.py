"""Exercise the installed wheel without dev dependencies or provider requests."""

import asyncio

from openai import AsyncOpenAI

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai import _single_attempt as bounded


async def main():
    provider = OpenAIProvider(api_key="synthetic-runtime-smoke")
    assert bounded.CAPABILITY in provider.get_info().capabilities
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


if __name__ == "__main__":
    asyncio.run(main())
