"""Real SDK serialization and one-send image effects over an offline transport."""
import asyncio
import base64
import hashlib
import json
from copy import deepcopy
from io import BytesIO

import httpx
import pytest
from openai import APIError, AsyncOpenAI
from PIL import Image

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
@pytest.mark.parametrize("failure", ["server", "disconnect", "redirect"])
async def test_legacy_images_also_sends_once(failure):
    from amplifier_module_provider_openai.images import OpenAIImageBackend
    sent = []
    def respond(req):
        sent.append(req)
        if failure == "disconnect":
            raise httpx.RemoteProtocolError("fixture disconnect after possible send")
        if failure == "redirect":
            return httpx.Response(307, headers={"location": "https://other.example.invalid/generate"})
        return httpx.Response(500, json={"error": {"message": "fixture", "type": "server_error"}})
    async with AsyncOpenAI(api_key="fixture-blue", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=True)) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        backend = OpenAIImageBackend(provider, {"model": "gpt-image-2.5-flare"})
        with pytest.raises(APIError):
            await asyncio.wait_for(backend.generate(action="generate", prompt="fixture",
                images=[], size="auto", quality="auto", background="auto"), 3)
        assert len(sent) == 1 and sent[0].url.path == "/v1/images/generations"
        assert client._client.follow_redirects is True and not client._client.is_closed


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


@pytest.mark.asyncio
async def test_redirect_does_not_repeat_generation_or_change_bound_endpoint():
    sent = []
    def respond(req):
        sent.append(req)
        return httpx.Response(307, headers={"location": "https://other.example.invalid/v1/responses"})
    async with AsyncOpenAI(api_key="fixture-blue", http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True)) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "unknown" and len(sent) == 1
        assert client._client.follow_redirects is True and not client._client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", ["oversized_text", "text", "output_count", "usage", "model"])
async def test_parse_failure_retains_paid_image_and_safe_provenance_once(malformed, recwarn):
    payload = response()
    payload["output"][0]["model"] = "gpt-image-2.5-flare"
    if malformed == "oversized_text":
        payload["output"][1]["content"][0]["text"] = "x" * 32001
    elif malformed == "text":
        payload["output"][1]["content"][0]["text"] = {"secret": "fixture-secret"}
    elif malformed == "output_count":
        payload["output"][1]["content"] *= 16
    elif malformed == "usage":
        payload["usage"] = {"input_tokens": "fixture-secret"}
    else:
        payload["model"] = {"secret": "fixture-secret"}
    sent = []
    def respond(req):
        sent.append(req)
        return httpx.Response(200, json=payload, headers={"x-request-id": "req-fixture"})
    async with AsyncOpenAI(api_key="fixture-blue", max_retries=5,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "failed"
        assert result["reason_code"] == "invalid_response"
        assert result["dispatch_state"] == "response_received"
        assert result["outputs"][0] == {"kind": "image", "mime_type": "image/png", "data": png()}
        assert sum(o["kind"] == "image" for o in result["outputs"]) == 1
        assert len(result["outputs"]) <= 16
        assert sum(len(o.get("text", "")) for o in result["outputs"]) <= 32000
        assert result["upstream"] == {
            "request_id": "req-fixture", "response_id": "resp-fixture", "image_call_id": "ig-fixture"}
        assert result["models"] == [
            {"purpose": "controller", "requested": "gpt-6.1-sol",
             "resolved": None if malformed == "model" else "gpt-6.1-sol"},
            {"purpose": "image", "requested": "gpt-image-2.5-flare", "resolved": "gpt-image-2.5-flare"}]
        assert "fixture-secret" not in repr(result)
        assert all("fixture-secret" not in str(warning.message) for warning in recwarn)
        assert result["continuation"] is None
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["responses", "images"])
@pytest.mark.parametrize("matching", [False, True])
async def test_requested_geometry_failure_keeps_safe_image_evidence_once(transport, matching):
    output = BytesIO()
    Image.new("RGB", (1024, 1024) if matching else (8, 8), "blue").save(output, format="PNG")
    data = output.getvalue()
    encoded = base64.b64encode(data).decode()
    payload = response() if transport == "responses" else {"created": 0, "data": [{"b64_json": encoded}]}
    if transport == "responses":
        payload["output"][0]["result"] = encoded
    sent = []

    def respond(req):
        sent.append(req)
        return httpx.Response(200, json=payload)

    async with AsyncOpenAI(api_key="fixture-blue",
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request(transport=transport)
        item["output"]["size"] = "1024x1024"
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == ("completed" if matching else "failed")
        assert result["dispatch_state"] == "response_received"
        images = [part for part in result["outputs"] if part["kind"] == "image"]
        assert images == [{"kind": "image", "mime_type": "image/png", "data": data}]
        if not matching:
            assert result["reason_code"] == "output_control_mismatch"
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1


@pytest.mark.asyncio
async def test_incomplete_response_retains_completed_image_without_continuation():
    payload = response()
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": "max_output_tokens"}
    payload["output"][0]["model"] = "gpt-image-2.5-flare"
    payload["usage"] = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    sent = []
    def respond(req):
        sent.append(req)
        return httpx.Response(200, json=payload, headers={"x-request-id": "req-fixture"})
    async with AsyncOpenAI(api_key="fixture-blue", max_retries=5,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "failed" and result["reason_code"] == "incomplete_response"
        assert result["dispatch_state"] == "response_received"
        assert result["outputs"] == [
            {"kind": "image", "mime_type": "image/png", "data": png()},
            {"kind": "text", "text": "Accompanying text"}]
        assert result["upstream"] == {
            "request_id": "req-fixture", "response_id": "resp-fixture", "image_call_id": "ig-fixture"}
        assert [model["resolved"] for model in result["models"]] == ["gpt-6.1-sol", "gpt-image-2.5-flare"]
        assert result["usage"]["normalized"] == payload["usage"]
        assert result["continuation"] is None
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["responses", "images"])
async def test_image_disconnect_retains_uncertain_dispatch_without_replay(transport):
    sent = []

    def respond(req):
        sent.append(req)
        raise httpx.RemoteProtocolError("private-disconnect-detail")

    async with AsyncOpenAI(api_key="fixture-blue", max_retries=5,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request(transport=transport)
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "unknown"
        assert result["dispatch_state"] == "unknown"
        assert result["reason_code"] == "generation_outcome_unknown"
        assert result["outputs"] == []
        assert "private-disconnect-detail" not in repr(result)
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("response_status", ["queued", "in_progress", "unrecognized-state"])
async def test_nonterminal_image_response_remains_unknown_but_received(response_status):
    payload = response()
    payload["status"] = response_status
    payload["output"] = []
    sent = []

    def respond(req):
        sent.append(req)
        return httpx.Response(200, json=payload)

    async with AsyncOpenAI(api_key="fixture-blue",
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
        provider = OpenAIProvider(api_key="fixture-blue", client=client)
        item = request()
        plan = (await provider.prepare_image_generation(item))["plan"]
        result = await provider.generate_image(item, plan=plan)
        assert result["status"] == "unknown"
        assert result["dispatch_state"] == "response_received"
        assert result["outputs"] == []
        with pytest.raises(ValueError, match="consumed"):
            await provider.generate_image(item, plan=plan)
        assert len(sent) == 1
