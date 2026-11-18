"""Offline real-SDK physical count/send and read-only indexed media contracts."""

import asyncio
import copy
import gc
import json
import weakref
from types import SimpleNamespace

import httpx
import openai
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolSpec,
)
from amplifier_core.utils.retry import RetryConfig

from amplifier_module_provider_openai import OpenAIProvider, _token_count

KEY = "experimental.media.v1"
MODEL = "gpt-6.1-sol"
PIXELS = "cHJpdmF0ZS1pbWFnZS1ieXRlcw=="


def request(*, marked=True):
    return ChatRequest(model=MODEL, messages=[
        Message(role="user", content="Inspect"),
        Message(role="assistant", content=[ToolCallBlock(id="read-1", name="read_file", input={})]),
        Message(role="tool", tool_call_id="read-1", name="read_file", content=[
            TextBlock(text="before"), ImageBlock(source={
                "type": "base64", "media_type": "image/png", "data": PIXELS,
            }), TextBlock(text="after"),
        ], metadata={KEY: {"refs": [{"mime": "image/png"}]}} if marked else None),
    ])


def completed():
    return {
        "id": "resp-fixture", "object": "response", "created_at": 1,
        "model": MODEL, "status": "completed", "output": [{
            "type": "message", "id": "msg-fixture", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
        }], "usage": {"input_tokens": 12, "output_tokens": 1, "total_tokens": 13},
    }


class Capability:
    enabled = True

    def __init__(self, private):
        self.private = weakref.ref(private)
        self.allowed = True
        self.calls = []
        self.result = {"valid": True}

    def validate(self, value, *, purpose, request_options=None):
        self.calls.append((value is self.private(), purpose, copy.deepcopy(request_options)))
        if value is not self.private() or not self.allowed:
            raise ValueError("private pixels must not escape")
        return self.result


def fixture(private, *, streaming=False, hook=None, replies=None, sdk_default_http=False):
    cap = Capability(private)
    wires, events = [], []

    async def transport(req):
        body = json.loads(req.content)
        wires.append((req.url.path, body, dict(req.headers)))
        assert cap.allowed, "revoked media reached actual HTTP transport"
        if replies:
            response = replies.pop(0)
            if callable(response):
                return response(req)
            return response
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 12})
        if body.get("stream"):
            a = {"type": "response.created", "response": {**completed(), "output": []}, "sequence_number": 0}
            b = {"type": "response.completed", "response": completed(), "sequence_number": 1}
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
                f"event: response.created\ndata: {json.dumps(a)}\n\n"
                f"event: response.completed\ndata: {json.dumps(b)}\n\n"
            ))
        return httpx.Response(200, json=completed())

    # SDK3.29's default is HTTPX2; legacy HTTPX injection is independently
    # supported by the SDK and must not conceal default request/stream types.
    if sdk_default_http:
        import httpx2

        legacy_transport = transport

        async def default_transport(req):
            response = await legacy_transport(req)
            return httpx2.Response(
                response.status_code, headers=dict(response.headers), content=await response.aread(),
            )

        http = openai.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(default_transport), trust_env=False,
            event_hooks={"request": [hook] if hook else []},
        )
    else:
        http = httpx.AsyncClient(transport=httpx.MockTransport(transport), trust_env=False,
                                 event_hooks={"request": [hook] if hook else []})
    sdk = openai.AsyncOpenAI(api_key="offline-fixture", max_retries=0, http_client=http)
    p = OpenAIProvider(client=sdk, config={
        "default_model": MODEL, "use_streaming": streaming, "raw": True, "max_retries": 0,
    })

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))

    coordinator = SimpleNamespace(cap=cap, hooks=SimpleNamespace(emit=emit))
    coordinator.get_capability = lambda key: coordinator.cap if key == KEY else None
    p.coordinator = coordinator
    return p, cap, wires, events


def facts(p, req, **kwargs):
    return p.get_media_request_support_for_request(request=req, model=MODEL, **kwargs)


def test_indexed_support_is_readonly_and_needs_no_client_capability_or_pixels():
    p = OpenAIProvider(api_key=None)
    req = request()
    req.messages[2].content = [TextBlock(text="[Session image digest]")]
    before = req.model_dump()
    assert facts(p, req) == {"v": 2, "positions": [
        {"message_index": 0, "role": "user"}, {"message_index": 2, "role": "tool"},
    ], "mime_types": ["image/png", "image/jpeg"]}
    assert req.model_dump() == before and p._client is None
    first = facts(p, req)
    first["positions"].clear()
    assert len(facts(p, req)["positions"]) == 2


@pytest.mark.parametrize("bad", ["orphan", "duplicate", "wrong", "following", "dual-conflict", "native", "model", "proxy"])
def test_indexed_support_refuses_ambiguous_opaque_or_unqualified_request(bad):
    p, req = OpenAIProvider(api_key=None), request()
    if bad == "orphan":
        req.messages.pop(1)
    elif bad == "duplicate":
        req.messages.append(req.messages[2].model_copy(deep=True))
    elif bad == "wrong":
        req.messages[2].tool_call_id = "other"
    elif bad == "following":
        req.messages[1:3] = reversed(req.messages[1:3])
    elif bad == "dual-conflict":
        req.messages[1].tool_calls = [{"id": "read-1", "name": "other", "arguments": {}}]
    elif bad == "native":
        req.messages[1].metadata = {"openai:output_items": [{"type": "computer_call", "call_id": "read-1"}]}
    elif bad == "model":
        req.model = "unknown-deployment"
    elif bad == "proxy":
        p.base_url = "https://proxy.invalid/v1"
    assert facts(p, req)["positions"] == []
    assert p._client is None


def test_indexed_overlay_indices_and_equivalent_dual_core_calls():
    req, p = request(), OpenAIProvider(api_key=None)
    req.messages.insert(0, Message(role="system", content="overlay"))
    req.messages[2].tool_calls = [{"id": "read-1", "name": "read_file", "arguments": "{}"}]
    assert facts(p, req)["positions"] == [
        {"message_index": 1, "role": "user"}, {"message_index": 3, "role": "tool"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("sdk_default_http", [False, True])
async def test_exact_private_request_budget_count_and_send_same_pixels(streaming, sdk_default_http):
    req = request()
    p, cap, wires, events = fixture(req, streaming=streaming, sdk_default_http=sdk_default_http)
    before = req.model_dump()
    try:
        decision = await p.request_budget(req, context_estimate=4)
        assert decision["measurement"]["kind"] == "provider_count"
        result = await p.complete(req)
        assert result.content[0].text == "OK"
        assert [w[0] for w in wires] == [
            "/v1/responses/input_tokens", "/v1/responses/input_tokens", "/v1/responses",
        ]
        assert wires[0][1]["input"] == wires[1][1]["input"] == wires[2][1]["input"]
        output = wires[-1][1]["input"][-1]
        assert output["type"] == "function_call_output" and output["call_id"] == "read-1"
        assert output["output"][1]["image_url"] == "data:image/png;base64," + PIXELS
        assert all(same for same, _, _ in cap.calls)
        assert {purpose for _, purpose, _ in cap.calls} == {"provider-count", "provider-send"}
        assert req.model_dump() == before and PIXELS not in json.dumps(events)
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["count", "send"])
@pytest.mark.parametrize("sdk_default_http", [False, True])
async def test_revocation_during_awaited_http_hook_precedes_physical_transport(where, sdk_default_http):
    req = request()

    async def hook(wire):
        if wire.url.path.endswith("input_tokens") == (where == "count"):
            await asyncio.sleep(0)
            cap.allowed = False

    p, cap, wires, events = fixture(req, hook=hook, sdk_default_http=sdk_default_http)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(wires) == (0 if where == "count" else 1)
        assert PIXELS not in json.dumps(events)
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["capability", "enabled", "coordinator", "key", "project", "endpoint", "client", "model", "options", "wire"])
async def test_changes_after_count_before_send_refuse(mutation):
    req = request()
    options = {"reasoning": {"effort": "low"}}
    p, cap, wires, _ = fixture(req)
    original_emit = p.coordinator.hooks.emit

    async def emit(name, data):
        await original_emit(name, data)
        if name != "llm:request":
            return
        if mutation == "capability":
            p.coordinator.cap = Capability(req)
        elif mutation == "enabled":
            cap.enabled = False
        elif mutation == "coordinator":
            p.coordinator = SimpleNamespace(get_capability=lambda _: cap)
        elif mutation == "key":
            p.client.api_key = "changed"
        elif mutation == "project":
            p.client.project = "changed"
        elif mutation == "endpoint":
            p.client.base_url = "https://elsewhere.invalid/v1"
        elif mutation == "client":
            p._client = p.client.with_options(max_retries=0)
        elif mutation == "model":
            p.default_model = "unknown"
        elif mutation == "options":
            options["reasoning"]["effort"] = "high"
        else:
            req.messages[0].content = "changed"

    p.coordinator.hooks.emit = emit
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req, request_options=options)
        assert len(wires) == 1
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [True, {"valid": 1}, {"valid": False}, None])
async def test_invalid_capability_success_is_never_truthy_admission(result):
    req = request()
    p, cap, wires, _ = fixture(req)
    cap.result = result
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_copied_marker_missing_capability_and_unmarked_traffic():
    req = request()
    p, _, wires, _ = fixture(req)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req.model_copy(deep=True))
        p.coordinator.cap = None
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
        await p.complete(request(marked=False))
        assert len(wires) == 2
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_count_retry_backoff_revocation_is_not_count_failure_or_retry(monkeypatch):
    req = request()
    replies = [httpx.Response(503, json={"error": {"message": "temporary"}})]
    p, cap, wires, _ = fixture(req, replies=replies)

    async def sleep(_):
        cap.allowed = False

    monkeypatch.setattr(_token_count.asyncio, "sleep", sleep)
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await p.request_budget(req, context_estimate=1)
        assert caught.value.retryable is False and len(wires) == 1
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_generation_retry_observer_revokes_before_next_count():
    req = request()
    p, cap, wires, _ = fixture(req, replies=[
        httpx.Response(200, json={"input_tokens": 12}),
        httpx.Response(429, json={"error": {"message": "limited"}}),
    ])
    p._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0)
    original = p.coordinator.hooks.emit

    async def emit(name, data):
        await original(name, data)
        if name == "provider:retry":
            cap.allowed = False

    p.coordinator.hooks.emit = emit
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(wires) == 2
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_guarded_binding_does_not_retain_private_pixels_after_collection():
    req = request()
    p, _cap, _, _ = fixture(req)
    try:
        await p.request_budget(req, context_estimate=1)
        ref = weakref.ref(req)
        del req
        gc.collect()
        assert ref() is None and p._media_request_bindings == {}
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sdk_default_http", [False, True])
async def test_cold_count_and_generation_bind_one_legitimate_client(monkeypatch, sdk_default_http):
    req = request()
    p, cap, wires, _ = fixture(req, sdk_default_http=sdk_default_http)
    sdk = p._client
    p._client = None
    p._api_key = "offline-fixture"
    made = []

    def construct(**kwargs):
        made.append(kwargs["api_key"])
        return sdk

    monkeypatch.setattr("amplifier_module_provider_openai.AsyncOpenAI", construct)
    try:
        await p.request_budget(req, context_estimate=1)
        await p.complete(req)
        assert p._client is sdk and made == ["offline-fixture"]
        assert len(wires) == 3 and all(same for same, _, _ in cap.calls)
    finally:
        await sdk.close()


@pytest.mark.asyncio
async def test_default_sdk_family_transport_identity_change_refuses():
    import httpx2

    req = request()
    p, _, wires, _ = fixture(req, sdk_default_http=True)
    original_emit = p.coordinator.hooks.emit
    original_transport = p.client._client._transport

    async def emit(name, data):
        await original_emit(name, data)
        if name == "llm:request":
            p.client._client._transport = httpx2.MockTransport(lambda _: pytest.fail("changed transport reached"))

    p.coordinator.hooks.emit = emit
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(wires) == 1 and wires[0][0].endswith("input_tokens")
    finally:
        p.client._client._transport = original_transport
        await p.client.close()


@pytest.mark.asyncio
async def test_count_unavailable_failure_and_async_capability_are_distinct(monkeypatch):
    req = request()
    p, cap, wires, _ = fixture(req)
    try:
        monkeypatch.setattr(p, "_provider_count_available", lambda: False)
        with pytest.raises(InvalidRequestError):
            p.request_budget(req, context_estimate=1)
        assert not wires
    finally:
        await p.client.close()
    p, cap, wires, _ = fixture(req, replies=[
        httpx.Response(400, json={"error": {"message": "image count unavailable"}}),
    ])
    try:
        with pytest.raises(_token_count.TokenCountError):
            await p.request_budget(req, context_estimate=1)
        assert len(wires) == 1
    finally:
        await p.client.close()
    p, cap, wires, _ = fixture(req)

    async def validate(*args, **kwargs):
        return {"valid": True}

    cap.validate = validate
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_wire_stream_changed_by_awaited_hook_is_checked_not_cached_body():
    req = request()

    async def hook(wire):
        await asyncio.sleep(0)
        wire.stream = httpx.ByteStream(b'{"model":"other"}')

    p, _, wires, _ = fixture(req, hook=hook)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_marked_continuation_requires_new_projection_without_second_send():
    req = request()
    incomplete = {**completed(), "status": "incomplete", "output": [],
                  "incomplete_details": {"reason": "max_output_tokens"}}
    p, _, wires, _ = fixture(req, replies=[
        httpx.Response(200, json={"input_tokens": 12}), httpx.Response(200, json=incomplete),
    ])
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(wires) == 2
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_concurrent_private_requests_keep_their_own_capability_identity():
    one, two = request(), request()
    p, cap, wires, _ = fixture(one)
    second_digest = "c2Vjb25k"
    two.messages[2].content[1].source["data"] = second_digest
    original_validate = cap.validate

    def validate(value, *, purpose, request_options=None):
        if value is two:
            return {"valid": True}
        return original_validate(value, purpose=purpose, request_options=request_options)

    cap.validate = validate
    try:
        await asyncio.gather(p.complete(one), p.complete(two))
        sends = [body for path, body, _ in wires if path.endswith("/responses")]
        assert len(sends) == 2
        assert {body["input"][-1]["output"][1]["image_url"] for body in sends} == {
            "data:image/png;base64," + PIXELS, "data:image/png;base64," + second_digest,
        }
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["url", "base64"])
@pytest.mark.parametrize("names", [["computer", "inspect"], ["inspect", "computer"]])
async def test_existing_model_generator_component_keeps_later_ordinary_image_history(source, names):
    from amplifier_core.message_models import ToolResultBlock

    from tests.test_chat_image_loss import DATA, URL, image, provider

    p, wires, events = provider()
    req = ChatRequest(model=MODEL, tools=[
        ToolSpec(name=name, parameters={"type": "object", "properties": {}}) for name in names
    ], messages=[
        Message(role="assistant", content=[ToolCallBlock(id="prior", name="inspect", input={})]),
        Message(role="tool", tool_call_id="prior", content="plain"),
        Message(role="assistant", content=[ToolCallBlock(id="target", name="computer", input={"actions": []})]),
        Message(role="tool", tool_call_id="target", content=[ToolResultBlock(
            tool_call_id="target", output=[
                TextBlock(text="before").model_dump(), image(source).model_dump(), TextBlock(text="after").model_dump(),
            ],
        )]),
    ])
    before = req.model_dump()
    try:
        await p.complete(req)
        items = wires[-1][1]["input"]
        assert [item["type"] for item in items] == ["function_call", "function_call_output"] * 2
        assert items[-1]["call_id"] == "target"
        assert items[-1]["output"][1]["image_url"] == (
            URL if source == "url" else f"data:image/png;base64,{DATA}"
        )
        assert wires[0][1]["input"] == wires[1][1]["input"]
        assert req.model_dump() == before and DATA not in json.dumps(events)
    finally:
        await p.client.close()


def test_generator_component_escaped_names_are_not_hashed(monkeypatch):
    p = OpenAIProvider(api_key=None)
    p.extra_request_params = {"extra_body": {"tools": [
        {"type": "function", "name": ["escaped"]}, {"type": "function", "name": "computer"},
    ]}}

    class Reached(Exception):
        pass

    def capture(messages, **kwargs):
        assert kwargs["ordinary_names"] == (["escaped"], "computer")
        raise Reached()

    monkeypatch.setattr("amplifier_module_provider_openai.project_chat_images", capture)
    with pytest.raises(Reached):
        p._budget_params(ChatRequest(messages=[Message(role="user", content="plain")]))


@pytest.mark.asyncio
async def test_removing_marker_cannot_downgrade_already_bound_private_object():
    req = request()
    p, _, wires, _ = fixture(req)
    try:
        await p.request_budget(req, context_estimate=1)
        req.messages[2].metadata = None
        p.coordinator.cap = None
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert len(wires) == 1
    finally:
        await p.client.close()


@pytest.mark.parametrize("name,args", [
    ("apply_patch", {"type": "create_file", "path": "x", "diff": "+x"}),
    ("computer", {"actions": []}),
])
def test_indexed_support_never_qualifies_inferred_native_image_results(name, args):
    req, p = request(), OpenAIProvider(api_key=None)
    req.messages[1].content[0].name = name
    req.messages[1].content[0].input = args
    req.messages[2].name = name
    assert facts(p, req)["positions"] == []
    req.tools = [ToolSpec(name=name, parameters={"type": "object", "properties": {}})]
    if name == "apply_patch":
        # The existing patch converter still infers native shape despite an
        # ordinary roster; the new support API must not advertise pixels there.
        assert facts(p, req)["positions"] == []


@pytest.mark.asyncio
async def test_typed_tool_options_are_bound_consistently_with_indexed_facts():
    req = request()
    p, _, wires, _ = fixture(req)
    options = {"tools": [ToolSpec(name="read_file", parameters={"type": "object", "properties": {}})]}
    try:
        assert facts(p, req, request_options=options)["positions"]
        await p.request_budget(req, context_estimate=1, request_options=options)
        await p.complete(req, request_options=options)
        assert len(wires) == 3
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_sdk_after_auth_preparation_await_revocation_precedes_http(monkeypatch):
    req = request()
    p, cap, wires, _ = fixture(req)
    original = openai.AsyncOpenAI._prepare_request

    async def prepare(client, wire):
        await original(client, wire)
        await asyncio.sleep(0)
        cap.allowed = False

    monkeypatch.setattr(openai.AsyncOpenAI, "_prepare_request", prepare)
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_count_retry_and_generation_retry_each_keep_private_wire(monkeypatch):
    req = request()
    p, cap, wires, _ = fixture(req, replies=[
        httpx.Response(503, json={"error": {"message": "temporary count"}}),
        httpx.Response(200, json={"input_tokens": 12}),
        httpx.Response(429, json={"error": {"message": "temporary generation"}}),
        httpx.Response(200, json={"input_tokens": 12}),
        httpx.Response(200, json=completed()),
    ])
    p._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0)

    async def sleep(_):
        return None

    monkeypatch.setattr(_token_count.asyncio, "sleep", sleep)
    try:
        await p.complete(req)
        assert len(wires) == 5
        assert all(body["input"] == wires[0][1]["input"] for _, body, _ in wires)
        assert all(same for same, _, _ in cap.calls)
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_guarded_redirect_never_follows_or_replays():
    req = request()
    p, _, wires, _ = fixture(req, replies=[
        httpx.Response(307, headers={"location": "https://elsewhere.invalid/steal"}),
    ])
    try:
        with pytest.raises(_token_count.TokenCountError):
            await p.request_budget(req, context_estimate=1)
        assert len(wires) == 1
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_marked_native_compaction_refuses_before_any_client_dispatch():
    req = request()
    p, _, wires, _ = fixture(req)
    try:
        with pytest.raises(InvalidRequestError):
            await p.compact_context(req)
        assert not wires
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_existing_injected_account_cannot_change_in_first_issuer_validation():
    req = request()
    p, cap, wires, _ = fixture(req)
    original = cap.validate

    def validate(*args, **kwargs):
        p.client.api_key = "changed-in-first-validator"
        p.client.project = "changed-project"
        return original(*args, **kwargs)

    cap.validate = validate
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["OpenAI-Organization", "OpenAI-Project", "Authorization"])
async def test_http_default_account_override_is_refused_not_silently_dropped(header):
    req = request()
    p, _, wires, _ = fixture(req)
    p.client._client.headers[header] = "another-account"
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(req)
        assert not wires
    finally:
        await p.client.close()