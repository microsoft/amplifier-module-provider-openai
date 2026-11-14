"""Private media admission through real SDK HTTP transport, never inference."""

import base64
import copy
import hashlib
import io
import json
from types import SimpleNamespace

import httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
)
from openai import AsyncOpenAI
from PIL import Image

from amplifier_module_provider_openai import OpenAIProvider, _media


def pixels(image_format="PNG"):
    out = io.BytesIO()
    Image.new("RGB", (3, 2), (29, 101, 173)).save(out, format=image_format)
    return out.getvalue()


def request(role="user", image_format="PNG"):
    data = base64.b64encode(pixels(image_format)).decode()
    parts = [
        TextBlock(text="before"),
        ImageBlock(
            source={
                "type": "base64",
                "media_type": "image/png" if image_format == "PNG" else "image/jpeg",
                "data": data,
            }
        ),
        TextBlock(text="after"),
    ]
    if role == "tool":
        messages = [
            Message(role="user", content="read"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="c1", name="read_file", input={})],
            ),
            Message(role="tool", tool_call_id="c1", content=parts),
        ]
    else:
        messages = [Message(role="user", content=parts)]
    return ChatRequest(messages=messages, max_output_tokens=64)


class Cap:
    version = 1

    def __init__(self, request):
        self.request, self.valid, self.calls = request, True, []

    def validate(self, request, *, purpose, request_options):
        assert request is self.request
        self.calls.append((purpose, copy.deepcopy(request_options)))
        if not self.valid:
            raise ValueError("expired")
        return {"valid": True}


def response(model):
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_fixture",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "OK", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 1, "total_tokens": 31},
    }


def provider(req, model="gpt-6.1-sol", *, stream=False, handler=None):
    cap = Cap(req)
    calls, events = [], []

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))

    def receive(r):
        body = json.loads(r.content)
        calls.append((r.url.path, body))
        if handler:
            custom = handler(r, cap, calls)
            if custom is not None:
                return custom
        if r.url.path.endswith("/input_tokens"):
            return httpx.Response(
                200, json={"object": "response.input_tokens", "input_tokens": 30}
            )
        if r.url.path.endswith("/compact"):
            return httpx.Response(
                200,
                json={
                    "id": "cmp_fixture",
                    "object": "response.compaction",
                    "created_at": 0,
                    "output": [
                        {
                            "type": "compaction",
                            "id": "cmp_item",
                            "encrypted_content": "opaque",
                        }
                    ],
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 2,
                        "total_tokens": 32,
                    },
                },
            )
        reply = response(model)
        if stream:
            events_wire = [
                {"type": "response.created", "response": {**reply, "output": []}},
                {"type": "response.completed", "response": reply},
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join("data: " + json.dumps(v) + "\n\n" for v in events_wire),
            )
        return httpx.Response(200, json=reply)

    client = AsyncOpenAI(
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive)),
    )
    p = OpenAIProvider(
        client=client,
        coordinator=SimpleNamespace(
            get_capability=lambda _: cap, hooks=SimpleNamespace(emit=emit)
        ),
        config={
            "default_model": model,
            "raw": True,
            "use_streaming": stream,
            "max_retries": 1,
            "min_retry_delay": 0,
            "max_retry_delay": 0,
        },
    )
    return p, cap, calls, events


@pytest.mark.asyncio
@pytest.mark.parametrize("model", sorted(_media.VISION_MODELS))
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
async def test_model_wire_count_send_original_pixels(model, role, stream, image_format):
    req = request(role, image_format)
    canonical = req.model_dump_json()
    p, cap, calls, events = provider(req, model, stream=stream)
    try:
        budget = await p.request_budget(req, context_estimate=100)
        assert budget["measurement"]["kind"] == "provider_count"
        await p.complete(req)
        assert len(calls) == 3
        assert calls[0][1]["input"] == calls[1][1]["input"] == calls[2][1]["input"]
        assert all(body["model"] == model for _, body in calls)
        final = calls[-1][1]["input"][-1]
        if role == "tool":
            assert final["call_id"] == "c1"
            parts = final["output"]
        else:
            parts = final["content"]
        assert [v["type"] for v in parts] == ["input_text", "input_image", "input_text"]
        data = parts[1]["image_url"].split(",", 1)[1]
        assert (
            hashlib.sha256(base64.b64decode(data, validate=True)).digest()
            == hashlib.sha256(pixels(image_format)).digest()
        )
        assert data not in json.dumps(events)
        assert "not retained" in json.dumps(events)
        assert req.model_dump_json() == canonical
        assert {purpose for purpose, _ in cap.calls} == {"count", "transport"}
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_revoke_after_count_no_generation():
    req = request()
    p, cap, calls, _ = provider(req)
    try:
        await p.request_budget(req, context_estimate=100)
        cap.valid = False
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_changed_options_after_fit_refuses_before_second_count():
    req = request()
    p, _, calls, _ = provider(req)
    try:
        await p.request_budget(req, context_estimate=100)
        with pytest.raises(InvalidRequestError):
            await p.complete(req, reasoning_effort="low")
        assert len(calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_unknown_transport_not_replayed():
    req = request()

    def fail(r, cap, calls):
        if not r.url.path.endswith("/input_tokens"):
            raise httpx.ReadTimeout("lost response")

    p, _, calls, _ = provider(req, handler=fail)
    try:
        with pytest.raises(LLMError) as caught:
            await p.complete(req)
        assert not caught.value.retryable
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_compaction_same_private_image_input():
    req = request("tool")
    p, _, calls, _ = provider(req)
    try:
        result = await p.compact_context(req)
        assert result["kind"] == "native"
        assert len(calls) == 2
        assert calls[0][1]["input"] == calls[1][1]["input"]
        assert calls[1][0].endswith("/compact")
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_known_rejection_rechecks_revocation_before_retry():
    req = request()

    def reject(r, cap, calls):
        if not r.url.path.endswith("/input_tokens"):
            cap.valid = False
            return httpx.Response(503, json={"error": {"message": "known rejection"}})

    p, _, calls, _ = provider(req, handler=reject)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_request_hook_cannot_mutate_canonical_after_fit():
    req = request()
    p, _, calls, _ = provider(req)

    async def mutate(name, data):
        if name == "llm:request":
            req.messages[0].content[0].text = "changed after planning"

    p.coordinator.hooks.emit = mutate
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        # Exact read-only counting precedes llm:request; no generation may
        # observe the subsequently changed canonical object.
        assert all(path.endswith("/input_tokens") for path, _ in calls)
    finally:
        await p.close()


def test_known_wire_diagnostic_copy_does_not_change_opaque_provenance():
    encoded = base64.b64encode(pixels()).decode()
    source = {
        "input": [
            {
                "type": "computer_call_output",
                "call_id": "c1",
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "data:image/png;base64," + encoded,
                },
            }
        ],
        "metadata": {"opaque": "must survive"},
        "arguments": {"image_url": "opaque"},
    }
    original = copy.deepcopy(source)
    safe = _media.diagnostic_copy(source)
    assert source == original and encoded not in json.dumps(safe)
    assert (
        safe["metadata"] == source["metadata"]
        and safe["arguments"] == source["arguments"]
    )


@pytest.mark.asyncio
async def test_account_changed_after_count_refuses_before_generation():
    req = request()
    p, _, calls, _ = provider(req)
    try:
        await p.request_budget(req, context_estimate=100)
        p.client.api_key = "changed-account"
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(calls) == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_nested_request_cannot_borrow_outer_private_witness():
    req = request()
    other = req.model_copy(deep=True)
    p, _, calls, _ = provider(req)
    refused = []

    async def observe(name, data):
        if name == "llm:request":
            with pytest.raises(InvalidRequestError):
                await p.complete(other)
            refused.append(True)

    p.coordinator.hooks.emit = observe
    try:
        await p.complete(req)
        assert refused == [True]
        assert sum(not path.endswith("/input_tokens") for path, _ in calls) == 1
    finally:
        await p.close()
