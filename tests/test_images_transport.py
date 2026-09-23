import asyncio
import base64
from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError, AsyncOpenAI, InternalServerError

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


def image_args(action):
    return {
        "action": action,
        "images": ([{"name": "source.png", "data": b"source", "mimeType": "image/png"}]
                   if action == "edit" else []),
        "prompt": "One original image",
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["generate", "edit"])
@pytest.mark.parametrize(
    "configured, expected",
    [({}, None), ({"timeout": None}, None), ({"timeout": 0.125}, 0.125),
     ({"timeout": 180}, 180.0), ({"timeout": 1200}, 1200.0)],
)
async def test_image_deadline_reaches_real_sdk_transport_without_inheriting_parent(
    action, configured, expected,
):
    requests = []

    def receive(request):
        requests.append(request)
        assert request.extensions["timeout"] == {
            "connect": expected, "read": expected, "write": expected, "pool": expected,
        }
        return httpx.Response(200, json={"created": 1, "data": [
            {"b64_json": base64.b64encode(b"image").decode()}
        ]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(receive), timeout=7)
    async with AsyncOpenAI(
        api_key="fixture-key", http_client=http, timeout=7, max_retries=2,
    ) as client:
        backend = OpenAIImageBackend(
            SimpleNamespace(client=client), {"model": "fixture-image", **configured},
        )
        assert (await backend.generate(**image_args(action)))["data"] == b"image"
        assert client.timeout == 7 and client.max_retries == 2
    assert len(requests) == 1
    assert requests[0].url.path == (
        "/v1/images/edits" if action == "edit" else "/v1/images/generations"
    )
    assert http.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["generate", "edit"])
async def test_image_timeout_stays_unknown_and_is_not_retried_or_replaced(action):
    requests = []

    def receive(request):
        requests.append(request)
        raise httpx.ReadTimeout("fixture response not received", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(receive))
    async with AsyncOpenAI(api_key="fixture-key", http_client=http, max_retries=2) as client:
        backend = OpenAIImageBackend(
            SimpleNamespace(client=client), {"model": "fixture-image", "timeout": 0.125},
        )
        with pytest.raises(APITimeoutError):
            await backend.generate(**image_args(action))
    assert len(requests) == 1
    assert requests[0].url.path == (
        "/v1/images/edits" if action == "edit" else "/v1/images/generations"
    )
    assert http.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["generate", "edit"])
async def test_cancelled_image_request_propagates_without_retry_or_fallback(action):
    requests = []
    started = asyncio.Event()

    async def receive(request):
        requests.append(request)
        started.set()
        await asyncio.Future()

    http = httpx.AsyncClient(transport=httpx.MockTransport(receive))
    async with AsyncOpenAI(api_key="fixture-key", http_client=http, max_retries=2) as client:
        backend = OpenAIImageBackend(
            SimpleNamespace(client=client), {"model": "fixture-image"},
        )
        task = asyncio.create_task(backend.generate(**image_args(action)))
        try:
            await asyncio.wait_for(started.wait(), 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert len(requests) == 1
    assert http.is_closed
