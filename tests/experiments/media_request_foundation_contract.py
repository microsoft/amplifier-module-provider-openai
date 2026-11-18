"""Finite real-Foundation wheel contract; invoke explicitly after exact install.

No import fallback, source injection or fake prepared-object registration.
The normal provider dependency set intentionally does not require Foundation.
"""

import base64
import io
import json

import pytest
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock
from amplifier_foundation.session.media import (
    MediaPolicy,
    SessionMediaCapability,
    SessionMediaStore,
)
from PIL import Image

from tests.test_media_request_seams import KEY, MODEL, facts, fixture


@pytest.mark.asyncio
@pytest.mark.parametrize("http_mode", ["legacy", "sdk-injected", "sdk-default"])
async def test_real_store_prepared_issuer_exact_private_count_send_and_refusals(tmp_path, http_mode):
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (20, 80, 130)).save(stream, format="PNG")
    pixels = stream.getvalue()
    origin = {"kind": "tool", "session_id": "session1", "id": "read-1"}
    store = SessionMediaStore(
        tmp_path / "media", "session1", policy=MediaPolicy(10000, 1000, 4),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    ref = (await store.ingest(
        pixels, source_format="image-bytes", origin=origin, policy=None,
    ))["value"]
    await store.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    cap = SessionMediaCapability(store, enabled=True)
    canonical = ChatRequest(model=MODEL, messages=[
        Message(role="user", content="Inspect"),
        Message(role="assistant", content=[ToolCallBlock(id="read-1", name="read_file", input={})]),
        Message(role="tool", tool_call_id="read-1", name="read_file", content=[
            TextBlock(text="before"), TextBlock(text=f"[Session image {ref['sha256']}]"), TextBlock(text="after"),
        ], metadata={KEY: {"refs": [ref]}}),
    ])
    p, _, wires, events = fixture(canonical, http_mode=http_mode)
    p.coordinator.cap = cap
    before = canonical.model_dump()
    identity = {"provider": "openai", "account": "fixture", "generation": 1,
                "media_support": facts(p, canonical)}
    prepared = await cap.prepare_request(
        canonical, provider_identity=identity, model=MODEL,
        request_options={}, purpose="count", execution_scope={},
    )
    private = prepared.request
    try:
        assert private is not canonical
        assert cap.validate(private, purpose="provider-count", request_options={"model": MODEL}) == {"valid": True}
        await p.request_budget(private, context_estimate=5, request_options={"model": MODEL})
        # Matching redundant model options normalize away in the real F API.
        await p.complete(private)
        assert [path for path, _, _ in wires] == [
            "/v1/responses/input_tokens", "/v1/responses/input_tokens", "/v1/responses",
        ]
        assert all(body["input"] == wires[0][1]["input"] for _, body, _ in wires)
        encoded = wires[-1][1]["input"][-1]["output"][1]["image_url"]
        assert base64.b64decode(encoded.split(",", 1)[1]) == pixels
        assert canonical.model_dump() == before
        assert base64.b64encode(pixels).decode() not in json.dumps(events)
        from amplifier_core.llm_errors import InvalidRequestError

        # The real issuer rejects copied, wrapped/canonical and forged-marker
        # objects, even when all visible bytes/options appear equivalent.
        for unknown in [private.model_copy(deep=True), canonical, ChatRequest.model_validate(private.model_dump())]:
            with pytest.raises(InvalidRequestError) as caught:
                await p.complete(unknown)
            assert ref["sha256"] not in str(caught.value)
        count = len(wires)
        with pytest.raises(InvalidRequestError):
            await p.complete(private, request_options={"model": MODEL, "temperature": 0.1})
        cap.enabled = False
        with pytest.raises(InvalidRequestError):
            await p.complete(private, request_options={"model": MODEL})
        cap.enabled = True
        p.coordinator.cap = SessionMediaCapability(store, enabled=True)
        with pytest.raises(InvalidRequestError):
            await p.complete(private, request_options={"model": MODEL})
        p.coordinator.cap = cap
        private.messages[0].content = "mutation"
        with pytest.raises(InvalidRequestError):
            await p.complete(private, request_options={"model": MODEL})
        assert len(wires) == count
    finally:
        prepared.release()
        assert private.messages == []
        assert canonical.model_dump() == before
        from amplifier_core.llm_errors import InvalidRequestError

        with pytest.raises(InvalidRequestError):
            await p.complete(private)
        store.close()
        await p.client.close()


@pytest.mark.asyncio
async def test_real_capability_policy_revocation_after_count_blocks_generation(tmp_path):
    stream = io.BytesIO()
    Image.new("RGB", (1, 1)).save(stream, format="PNG")
    allowed = [True]
    store = SessionMediaStore(
        tmp_path / "media", "session1", policy=MediaPolicy(10000, 1000, 4),
        checkpoint_verifier=lambda receipt: receipt["refs"],
        policy_verifier=lambda ref, purpose: allowed[0],
    )
    origin = {"kind": "tool", "session_id": "session1", "id": "read-1"}
    ref = (await store.ingest(stream.getvalue(), source_format="image-bytes", origin=origin, policy=None))["value"]
    await store.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    cap = SessionMediaCapability(store, enabled=True)
    canonical = ChatRequest(model=MODEL, messages=[
        Message(role="assistant", content=[ToolCallBlock(id="read-1", name="read_file", input={})]),
        Message(role="tool", tool_call_id="read-1", content=[
            TextBlock(text=f"[Session image {ref['sha256']}]"),
        ], metadata={KEY: {"refs": [ref]}}),
    ])
    p, _, wires, _ = fixture(canonical)
    p.coordinator.cap = cap
    prepared = await cap.prepare_request(
        canonical, provider_identity={"provider": "openai", "media_support": facts(p, canonical)},
        model=MODEL, request_options={}, purpose="count", execution_scope={},
    )
    original = p.coordinator.hooks.emit

    async def emit(name, data):
        await original(name, data)
        if name == "llm:request":
            allowed[0] = False

    p.coordinator.hooks.emit = emit
    try:
        from amplifier_core.llm_errors import InvalidRequestError

        with pytest.raises(InvalidRequestError):
            await p.complete(prepared.request)
        assert len(wires) == 1 and wires[0][0].endswith("input_tokens")
    finally:
        prepared.release()
        store.close()
        await p.client.close()