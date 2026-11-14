"""Installed public Foundation wrapper to actual SDK HTTP, no inference."""

import base64
import io
import json
from types import SimpleNamespace

import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock
from PIL import Image

media = pytest.importorskip("amplifier_foundation.session.media")
from amplifier_foundation.session.media_provider import MediaPreparedProvider

from amplifier_module_provider_openai import _media
from tests.test_media_projection import provider


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize(
    "model,stream,revoke",
    [
        (model, stream, False)
        for model in sorted(_media.VISION_MODELS)
        for stream in [False, True]
        if not stream or model not in _media.NONSTREAM_MODELS
    ]
    + [("gpt-6.1-sol", stream, True) for stream in [False, True, "native"]]
    + [("gpt-6.1-sol", "native", False)],
)
async def test_store_wrapper_count_send(tmp_path, role, image_format, model, stream, revoke):
    output = io.BytesIO()
    Image.new("RGB", (3, 2), (17, 89, 167)).save(output, format=image_format)
    pixels = output.getvalue()
    encoded = base64.b64encode(pixels).decode()
    store = media.SessionMediaStore(
        tmp_path / "media",
        "fixture",
        policy=media.MediaPolicy(100000, 10000, 8),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    cap = media.SessionMediaCapability(store, enabled=True)
    origin = {"kind": "tool", "session_id": "fixture", "id": "c1"}
    ref = (
        await cap.ingest(
            pixels, source_format="image-bytes", origin=origin, policy=None
        )
    )["value"]
    await cap.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    parts = [
        TextBlock(text="before"),
        TextBlock(text=f"[Session image {ref['sha256']}]"),
        TextBlock(text="after"),
    ]
    metadata = {"experimental.media.v1": {"refs": [ref]}, "opaque": {"keep": True}}
    messages = (
        [Message(role="user", content=parts, metadata=metadata)]
        if role == "user"
        else [
            Message(role="user", content="read"),
            Message(
                role="assistant",
                content=[ToolCallBlock(id="c1", name="read_file", input={})],
            ),
            Message(role="tool", tool_call_id="c1", content=parts, metadata=metadata),
        ]
    )
    req = ChatRequest(messages=messages, max_output_tokens=64)
    canonical = req.model_dump_json()
    if stream == "native":
        from tests.test_native_transport import created, finished, make

        p, owner = make()
        p.owner = None
        owner.messages = [message.model_dump() for message in messages]
        socket = p.socket
        socket.feed(created("fixture"), finished("fixture"))
        calls, events = socket.sent, []
        p.trace = lambda name, data: events.append((name, data))
        p.trace_context = lambda data: events.append(("context", data))
        p.coordinator = SimpleNamespace(get_capability=lambda _: cap)
    else:
        p, _, calls, events = provider(req, model, stream=stream)
    p.coordinator.get_capability = lambda _: cap
    wrapped = MediaPreparedProvider(
        p,
        cap,
        provider_identity={"provider": "openai", "account": "offline", "generation": 1},
        model=model,
        execution_scope={},
    )
    try:
        budget = await wrapped.request_budget(
            req, context_estimate=100, request_options={}
        )
        if stream == "native":
            assert budget is None  # Existing native exact-type count gate.
            assert not calls
        else:
            assert budget["measurement"]["kind"] == "provider_count"
            assert len(calls) == 1
        if revoke:
            store.epoch += 1
            with pytest.raises((InvalidRequestError, media.MediaRefusal)):
                await wrapped.complete(req)
            assert len(calls) == (0 if stream == "native" else 1)
        else:
            await wrapped.complete(req)
            if stream == "native":
                assert len(calls) == 1
                item = calls[0]["input"][-1]
                assert p._request_messages == [] and p._native_full_params is None
                assert not hasattr(p, "_last_native_request")
                assert encoded not in str(p.seen) and encoded not in str(p.last_context)
            else:
                assert len(calls) == 3
                assert all(body["model"] == model for _, body in calls)
                assert (
                    calls[0][1]["input"] == calls[1][1]["input"] == calls[2][1]["input"]
                )
                item = calls[2][1]["input"][-1]
            actual = item["content"] if role == "user" else item["output"]
            assert [x["type"] for x in actual] == [
                "input_text",
                "input_image",
                "input_text",
            ]
            assert (
                base64.b64decode(actual[1]["image_url"].split(",", 1)[1], validate=True)
                == pixels
            )
        assert req.model_dump_json() == canonical and encoded not in canonical
        assert encoded not in json.dumps(events)
    finally:
        wrapped.close()
        store.close()
        await p.close()
