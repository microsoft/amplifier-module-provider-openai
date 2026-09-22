import base64
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError

from amplifier_module_provider_openai.images import OpenAIImageBackend


@pytest.mark.asyncio
async def test_real_sdk_serializes_generate_and_multipart_edit_without_retry():
    requests = []

    def receive(request):
        requests.append(request)
        if len(requests) == 3:
            return httpx.Response(500, json={"error": {"message": "fixture failure"}})
        return httpx.Response(
            200,
            headers={"x-request-id": "fixture-image-request"},
            json={
                "created": 1,
                "data": [{"b64_json": base64.b64encode(b"exact png bytes").decode()}],
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(receive))
    async with AsyncOpenAI(
        api_key="fixture-key", http_client=http, max_retries=2
    ) as client:
        backend = OpenAIImageBackend(
            SimpleNamespace(client=client), {"model": "fixture-image"}
        )
        args = {
            "prompt": "An original image",
            "size": "1024x1024",
            "quality": "low",
            "background": "opaque",
        }
        generated = await backend.generate(action="generate", images=[], **args)
        assert (
            generated["data"] == b"exact png bytes"
            and generated["request_id"] == "fixture-image-request"
        )
        edited = await backend.generate(
            action="edit",
            images=[
                {
                    "name": "original.png",
                    "mimeType": "image/png",
                    "data": b"original-bytes",
                }
            ],
            **args,
        )
        assert edited["data"] == generated["data"]
        with pytest.raises(InternalServerError):
            await backend.generate(action="generate", images=[], **args)
    assert len(requests) == 3
    assert requests[0].url.path == "/v1/images/generations"
    assert requests[1].url.path == "/v1/images/edits"
    assert b"original-bytes" in requests[1].content
    assert "multipart/form-data" in requests[1].headers["content-type"]
