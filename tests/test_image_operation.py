"""Real SDK serialization and one-send image effects over an offline transport."""
import asyncio
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json

import httpx
from openai import AsyncOpenAI
from PIL import Image
import pytest

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.image_operation import PROTOCOL


def png():
    output = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(output, format="PNG")
    return output.getvalue()


def request(*, edit=False, transport="responses"):
    data = png()
    return {"protocol": PROTOCOL, "request_id": "one-effect", "operation": "edit" if edit else "generate",
            "prompt": "Make one image", "inputs": [{"purpose": "target", "mime_type": "image/png", "data": data,
              "sha256": hashlib.sha256(data).hexdigest()}] if edit else [],
            "selection": {"model": "gpt-6.1-sol", "config": {"image_generation": {
              "model": "gpt-image-2.5-flare", "transport": transport}}},
            "output": {"count": 1, "mime_type": "image/png", "size": "auto", "quality": "auto", "background": "auto"}}


def response():
    return {"id": "resp-fixture", "object": "response", "created_at": 0, "status": "completed",
            "model": "gpt-6.1-sol", "output": [
                {"type": "image_generation_call", "id": "ig-fixture", "status": "completed",
                 "result": base64.b64encode(png()).decode(), "revised_prompt": "Revised"},
                {"type": "message", "id": "msg-fixture", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "Accompanying text", "annotations": []}]},
            ], "usage": None, "parallel_tool_calls": False, "tool_choice": {"type": "image_generation"},
            "tools": [], "temperature": 1, "top_p": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("edit", [False, True])
async def test_responses_uses_distinct_models_bound_account_and_final_image(edit):
    sent = []
    def respond(req):
        sent.append(req)
        assert req.headers["authorization"] == "Bearer fixture-blue"
        assert str(req.url) == "https://api.openai.com/v1/responses"
        payload = json.loads(req.content)
        assert payload["model"] == "gpt-6.1-sol"
        tool = payload["tools"][0]
        assert tool["type"] == "image_generation" and tool["model"] == "gpt-image-2.5-flare"
        assert tool["action"] == ("edit" if edit else "generate")
        assert payload["tool_choice"] == {"type": "image_generation"}
        assert payload["max_tool_calls"] == 1 and payload["parallel_tool_calls"] is False
        assert payload["store"] is False and payload["stream"] is False
        assert "previous_response_id" not in payload
        if edit:
            assert payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json=response(), headers={"x-request-id": "req-fixture"})
    async with AsyncOpenAI(api_key="fixture-blue", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        assert provider.get_image_generation_capabilities()["supported"] is True
        original = deepcopy(provider.config)
        item = request(edit=edit)
        prepared = await provider.prepare_image_generation(item)
        assert prepared["supported"] is True and sent == []
        result = await provider.generate_image(item, plan=prepared["plan"])
        assert result["status"] == "completed" and result["outputs"][0]["data"] == png()
        assert result["outputs"][1]["text"] == "Accompanying text"
        assert result["models"][1]["resolved"] is None
        assert result["usage"]["normalized"] is None
        assert result["upstream"]["request_id"] == "req-fixture"
        assert provider.config == original
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=prepared["plan"])
        assert len(sent) == 1


@pytest.mark.asyncio
async def test_sdk_500_is_one_request_and_cannot_reuse_plan():
    sent = []
    def respond(req):
        sent.append(req)
        return httpx.Response(500, json={"error": {"message": "fixture", "type": "server_error"}})
    async with AsyncOpenAI(api_key="fixture-blue", max_retries=5,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "unknown"
        with pytest.raises(ValueError):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["prompt", "model", "plan", "account", "endpoint", "project", "organization", "headers"])
async def test_changed_request_plan_or_binding_never_sends(change):
    sent = []
    def respond(req):
        sent.append(req)
        raise AssertionError("must not dispatch")
    async with AsyncOpenAI(api_key="fixture-blue", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        if change == "prompt": item["prompt"] = "different"
        if change == "model": item["selection"]["config"]["image_generation"]["model"] = "gpt-image-1"
        if change == "plan": plan["model"] = "gpt-image-1"
        if change == "account": client.api_key = "fixture-green"
        if change == "endpoint": client.base_url = "https://other.example.invalid/v1"
        if change == "project": client.project = "project-other"
        if change == "organization": client.organization = "organization-other"
        if change == "headers": client._custom_headers["Authorization"] = "Bearer fixture-green"
        with pytest.raises(ValueError):
            await provider.generate_image(item, plan=plan)
        assert not sent


@pytest.mark.asyncio
async def test_missing_native_capability_controls_and_endpoint_refuse_without_call():
    provider = OpenAIProvider(api_key="fixture-blue", config={"base_url": "https://proxy.example.invalid/v1"})
    assert provider.get_image_generation_capabilities()["supported"] is False
    assert (await provider.prepare_image_generation(request()))["supported"] is False
    assert provider._client is None
    provider = OpenAIProvider(api_key="fixture-blue")
    item = request()
    item["output"]["size"] = "1792x1024"
    assert (await provider.prepare_image_generation(item))["supported"] is False
    assert provider._client is None


@pytest.mark.asyncio
async def test_cancelled_send_stays_consumed():
    started = asyncio.Event()
    async def respond(req):
        started.set()
        await asyncio.Event().wait()
    async with AsyncOpenAI(api_key="fixture-blue", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        task = asyncio.create_task(provider.generate_image(item, plan=plan))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        with pytest.raises(ValueError): await provider.generate_image(item, plan=plan)
