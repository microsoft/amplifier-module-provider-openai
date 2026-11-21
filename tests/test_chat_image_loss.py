"""Offline entry-to-real-SDK regressions for the audited image-loss paths."""

import copy
import json
import sys
import traceback
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
    ToolResultBlock,
    ToolSpec,
)
from amplifier_core.utils.retry import RetryConfig

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._chat_images import UNAVAILABLE

URL = "https://images.example.test/a.png?private-token=fixture"
DATA = "cHJpdmF0ZS1waXhlbHM="


def image(source="url"):
    return ImageBlock(source=(
        {"type": "url", "url": URL} if source == "url" else
        {"type": "base64", "media_type": "image/png", "data": DATA}
    ))


def body():
    return {
        "id": "resp-fixture", "object": "response", "created_at": 1,
        "model": "gpt-6.1-sol", "status": "completed",
        "output": [{"type": "message", "id": "msg-fixture", "role": "assistant",
                    "status": "completed", "content": [
                        {"type": "output_text", "text": "OK", "annotations": []},
                    ]}],
        "usage": {"input_tokens": 12, "output_tokens": 1, "total_tokens": 13},
    }


def provider(*, stream=False, failure=False, replies=None):
    wire, events = [], []

    def handler(request):
        data = json.loads(request.content) if request.content else {}
        wire.append((request.url.path, data))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 12})
        if replies:
            return replies.pop(0)
        if failure:
            return httpx.Response(400, json={"error": {
                "type": "invalid_request_error", "message": f"bad image {URL} {DATA}",
            }})
        if stream:
            event = {"type": "response.completed", "response": body(), "sequence_number": 1}
            created = {"type": "response.created", "response": {**body(), "output": []}, "sequence_number": 0}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 text=f"event: response.created\ndata: {json.dumps(created)}\n\n"
                                      f"event: response.completed\ndata: {json.dumps(event)}\n\n")
        return httpx.Response(200, json=body())

    sdk = openai.AsyncOpenAI(api_key="fixture", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    p = OpenAIProvider(client=sdk, config={
        "default_model": "gpt-6.1-sol", "use_streaming": stream,
        "max_retries": 0, "raw": True,
    })

    async def emit(name, data):
        # Capture at FIRST subscriber, not a downstream redaction hook.
        events.append((name, copy.deepcopy(data)))

    p.coordinator = SimpleNamespace(hooks=SimpleNamespace(emit=emit))
    return p, wire, events


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("source", ["url", "base64"])
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-5", "gpt-5-2025-08-07"])
async def test_user_images_count_and_send_same_order_without_local_fetch(stream, source, model):
    p, wire, events = provider(stream=stream)
    request = ChatRequest(model=model, messages=[Message(role="user", content=[
        TextBlock(text="before"), image(source), TextBlock(text="after"),
    ])])
    before = request.model_dump()
    try:
        await p.complete(request)
        assert [path for path, _ in wire] == ["/v1/responses/input_tokens", "/v1/responses"]
        assert all(payload["model"] == model for _, payload in wire)
        assert wire[0][1]["input"] == wire[1][1]["input"]
        content = wire[1][1]["input"][0]["content"]
        assert content == [
            {"type": "input_text", "text": "before"},
            {"type": "input_image", "image_url": URL if source == "url" else f"data:image/png;base64,{DATA}"},
            {"type": "input_text", "text": "after"},
        ]
        assert request.model_dump() == before
        assert URL not in json.dumps(events) and DATA not in json.dumps(events)
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("role", ["tool", "user"])
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-5", "gpt-5-2025-08-07"])
async def test_nested_tool_result_uses_existing_rich_transport_and_keeps_status(envelope, role, model):
    p, wire, events = provider()
    blocks = [TextBlock(text="before").model_dump(), image("base64").model_dump(),
              TextBlock(text="after").model_dump()]
    output = {"success": False, "status": "failed", "content": blocks} if envelope else blocks
    request = ChatRequest(model=model, messages=[
        Message(role="assistant", content=[ToolCallBlock(id="call-1", name="inspect", input={})]),
        Message(role=role, tool_call_id="call-1" if role == "tool" else None,
                name="inspect", content=[ToolResultBlock(tool_call_id="call-1", output=output)]),
    ])
    before = request.model_dump()
    try:
        await p.complete(request)
        assert wire[0][1]["input"] == wire[1][1]["input"]
        items = wire[1][1]["input"]
        assert [i["type"] for i in items] == ["function_call", "function_call_output"]
        assert items[1]["call_id"] == "call-1"
        rich = items[1]["output"]
        if envelope:
            assert json.loads(rich.pop(0)["text"]) == {"success": False, "status": "failed"}
        assert rich[0]["text"] == "before" and rich[2]["text"] == "after"
        assert rich[1] == {"type": "input_image", "image_url": f"data:image/png;base64,{DATA}"}
        assert DATA not in json.dumps(events) and request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_runtime_direct_tool_url_and_base64_keep_tool_call_order():
    p, wire, _ = provider()
    request = ChatRequest(messages=[
        Message(role="assistant", content=[
            ToolCallBlock(id="one", name="inspect", input={}),
            ToolCallBlock(id="two", name="inspect", input={}),
        ]),
        *[Message(role="tool", tool_call_id=identity, name="inspect", content=[
            TextBlock(text=identity), image(source),
        ]) for identity, source in [("one", "url"), ("two", "base64")]],
    ])
    try:
        await p.complete(request)
        outputs = [i for i in wire[-1][1]["input"] if i["type"] == "function_call_output"]
        assert [i["call_id"] for i in outputs] == ["one", "two"]
        assert [i["output"][1]["image_url"] for i in outputs] == [URL, f"data:image/png;base64,{DATA}"]
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outer", [None, "wrong"])
async def test_nested_correlation_error_is_payload_free_before_count_and_send(outer):
    p, wire, _ = provider()
    request = ChatRequest(messages=[Message(role="tool", tool_call_id=outer, content=[
        ToolResultBlock(tool_call_id="missing", output=[image().model_dump()]),
    ])])
    try:
        with pytest.raises(InvalidRequestError) as error:
            await p.complete(request)
        assert not wire and URL not in str(error.value)
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["assistant", "system", "developer"])
async def test_unqualified_historical_positions_keep_original_slot_and_plain_followup(role):
    p, wire, _ = provider()
    request = ChatRequest(messages=[
        Message(role=role, content=[TextBlock(text="before"), image(), TextBlock(text="after")]),
        Message(role="user", content="plain followup"),
    ])
    before = request.model_dump()
    try:
        await p.complete(request)
        payload = wire[-1][1]
        assert URL not in json.dumps(payload) and UNAVAILABLE in json.dumps(payload)
        assert payload["input"][-1]["content"] == [{"type": "input_text", "text": "plain followup"}]
        if role == "assistant":
            assert payload["input"][0]["role"] == "assistant"
            assert [b["text"] for b in payload["input"][0]["content"]] == ["before", UNAVAILABLE, "after"]
        assert request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_error_echoes_are_removed_before_first_subscriber_and_public_traceback(stream, caplog):
    p, wire, events = provider(stream=stream, failure=True)
    request = ChatRequest(messages=[Message(role="user", content=[image(), image("base64")])])
    try:
        with pytest.raises(InvalidRequestError) as error:
            await p.complete(request)
        diagnostic = "".join(traceback.format_exception(error.value)) + json.dumps(events) + caplog.text
        assert URL not in diagnostic and DATA not in diagnostic
        assert error.value.retryable is False and error.value.status_code == 400
        assert len(wire) == 2
    finally:
        await p.client.close()


@pytest.mark.parametrize("model", [
    "unknown-deployment", "gpt-6.2-future", "o3-mini",
    "gpt-5-2099-01-01", "gpt-5-unknown", "gpt-5.99-future",
])
def test_unknown_or_text_only_model_is_not_forced_vision(model):
    p = OpenAIProvider(api_key="fixture")
    converted = p._convert_messages([Message(role="user", content=[image()]).model_dump()], model=model)
    assert converted[0]["content"] == [{"type": "input_text", "text": UNAVAILABLE}]


def test_dynamic_azure_subclass_does_not_infer_deployment_vision():
    class AzureProvider(OpenAIProvider):
        @property
        def name(self):
            return "azure-openai"

    p = AzureProvider(api_key="fixture")
    converted = p._convert_messages([Message(role="user", content=[image()]).model_dump()])
    assert converted[0]["content"] == [{"type": "input_text", "text": UNAVAILABLE}]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["orphan", "duplicate", "mismatch", "duplicate-call", "later-text", "later-call"])
async def test_direct_runtime_tool_images_require_unique_preceding_call(kind):
    p, wire, _ = provider()
    call = Message(role="assistant", content=[ToolCallBlock(id="one", name="inspect", input={})])
    result = Message(role="tool", tool_call_id="one", name="inspect", content=[
        TextBlock(text="before"), image(), TextBlock(text="after"),
    ])
    messages = {
        "orphan": [result],
        "duplicate": [call, result, result],
        "mismatch": [call, result.model_copy(update={"tool_call_id": "other"})],
        "duplicate-call": [call, call, result],
        "later-text": [call, result, Message(role="tool", tool_call_id="one", content="other")],
        "later-call": [call, result, call],
    }[kind]
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(ChatRequest(messages=messages))
        assert not wire
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_provider_produced_canonical_calls_roundtrip_with_real_image_result(nested):
    p, wire, _ = provider()
    payload = body()
    payload["output"] = [{
        "type": "function_call", "id": "fc-one", "call_id": "one",
        "name": "inspect", "arguments": "{}", "status": "completed",
    }]
    reply = p._convert_to_chat_response(payload)
    content = [image()]
    if nested:
        content = [ToolResultBlock(tool_call_id="one", output=[image().model_dump()])]
    request = ChatRequest(messages=[
        Message(role="assistant", content=reply.content, metadata=reply.metadata),
        Message(role="tool", tool_call_id="one", content=content),
    ])
    try:
        await p.complete(request)
        assert [i["type"] for i in wire[-1][1]["input"]] == ["function_call", "function_call_output"]
        assert wire[-1][1]["input"][1]["output"][0]["image_url"] == URL
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "system", "developer"])
@pytest.mark.parametrize("nested", [False, True])
async def test_nonassistant_call_shapes_cannot_qualify_image_result(role, nested):
    p, wire, _ = provider()
    content = [image()]
    if nested:
        content = [ToolResultBlock(tool_call_id="one", output=[image().model_dump()])]
    request = ChatRequest(messages=[
        Message(role=role, content=[ToolCallBlock(id="one", name="inspect", input={})],
                metadata={"openai:output_items": [{
                    "type": "function_call", "call_id": "one", "name": "inspect", "arguments": "{}",
                }]}),
        Message(role="tool", tool_call_id="one", content=content),
    ])
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(request)
        assert not wire
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("equivalent", [False, True])
@pytest.mark.parametrize("nested", [False, True])
async def test_dual_core_call_views_must_actually_be_equivalent(equivalent, nested):
    p, wire, _ = provider()
    content = [image()]
    if nested:
        content = [ToolResultBlock(tool_call_id="one", output=[image().model_dump()])]
    request = ChatRequest(messages=[
        Message(role="assistant", content=[ToolCallBlock(id="one", name="inspect", input={"a": 1})],
                tool_calls=[{"id": "one", "name": "inspect", "arguments": {"a": 1 if equivalent else 2}}]),
        Message(role="tool", tool_call_id="one", content=content),
    ])
    try:
        if equivalent:
            await p.complete(request)
            assert [i["type"] for i in wire[-1][1]["input"]] == ["function_call", "function_call_output"]
        else:
            with pytest.raises(InvalidRequestError):
                await p.complete(request)
            assert not wire
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["catalog", ["business-id"]])
async def test_image_shaped_business_arguments_are_not_a_media_protocol(source):
    p, wire, _ = provider()
    arguments = {"type": "image", "source": source}
    request = ChatRequest(messages=[
        Message(role="assistant", content=[ToolCallBlock(id="one", name="inspect", input=arguments)]),
        Message(role="tool", tool_call_id="one", content="plain result"),
    ])
    try:
        await p.complete(request)
        assert json.loads(wire[-1][1]["input"][0]["arguments"]) == arguments
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["continuation", "background"])
async def test_later_failure_media_echo_is_safe_without_replaying_unknown_effects(phase, caplog):
    first = body()
    if phase == "continuation":
        first.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    else:
        first.update(status="queued", background=True)
    replies = [
        httpx.Response(200, json=first),
        httpx.Response(400, json={"error": {"message": f"bad {URL} {DATA}", "type": "invalid_request_error"}}),
    ]
    p, wire, events = provider(replies=replies)
    request = ChatRequest(messages=[Message(role="user", content=[image(), image("base64")])])
    try:
        with pytest.raises(Exception) as caught:
            await p.complete(request, **({"background": True, "poll_interval": 0} if phase == "background" else {}))
        diagnostic = "".join(traceback.format_exception(caught.value)) + json.dumps(events) + caplog.text
        assert URL not in diagnostic and DATA not in diagnostic
        assert len([path for path, _ in wire if not path.endswith("input_tokens")]) == 2
        assert caught.value.retryable is False
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("private_url", [
    'https://images.example.test/a.png?token="fixture"',
    "https://images.example.test/a.png?token=back\\slash",
    "https://images.example.test/a.png?token=秘密",
])
async def test_json_escaped_media_echo_is_safe_after_decoding(private_url, monkeypatch, caplog):
    monkeypatch.setattr(sys.modules[__name__], "URL", private_url)
    p, _, events = provider(failure=True)
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await p.complete(ChatRequest(messages=[Message(role="user", content=[image()])]))
        decoded = json.loads(str(caught.value))
        assert private_url not in decoded["message"]
        diagnostic = "".join(traceback.format_exception(caught.value)) + json.dumps(events) + caplog.text
        assert private_url not in diagnostic and json.dumps(private_url)[1:-1] not in diagnostic
        assert caught.value.status_code == 400 and caught.value.retryable is False
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [
    {"customer": "business", "content": {"type": "image", "source": "catalog"}},
    {"customer": "business", "content": [{"content": {"type": "image", "source": "catalog"}}]},
])
async def test_business_content_mapping_is_not_a_known_media_envelope(output):
    p, wire, _ = provider()
    request = ChatRequest(messages=[
        Message(role="assistant", content=[ToolCallBlock(id="one", name="inspect", input={})]),
        Message(role="tool", tool_call_id="one", content=[ToolResultBlock(tool_call_id="one", output=output)]),
    ])
    before = request.model_dump()
    try:
        await p.complete(request)
        result = wire[-1][1]["input"][1]["output"]
        # Legacy opaque ToolResult JSON remains unchanged, not media-classified.
        if isinstance(result, list):
            result = result[0]["text"]
        assert json.loads(result)[0]["output"] == output
        assert request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["url", "base64"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "o3-mini"])
async def test_ordinary_computer_roster_wins_over_native_name_shape(source, nested, model):
    p, wire, _ = provider()
    content = [TextBlock(text="before"), image(source), TextBlock(text="after")]
    if nested:
        content = [ToolResultBlock(tool_call_id="one", output=[b.model_dump() for b in content])]
    request = ChatRequest(model=model, tools=[
        ToolSpec(name="computer", parameters={"type": "object", "properties": {}}),
    ], messages=[
        Message(role="assistant", content=[ToolCallBlock(
            id="one", name="computer", input={"actions": []},
        )]),
        Message(role="tool", tool_call_id="one", content=content),
    ])
    try:
        await p.complete(request)
        items = wire[-1][1]["input"]
        assert items[0]["type"] == "function_call"
        assert items[1]["type"] == "function_call_output"
        if model == "o3-mini":
            assert items[1]["output"][1]["text"].startswith("[Image omitted:")
        else:
            assert items[1]["output"][1]["image_url"] == (
                URL if source == "url" else f"data:image/png;base64,{DATA}"
            )
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6.2-sol"])
@pytest.mark.parametrize("source", ["url", "base64"])
@pytest.mark.parametrize("names,prior,nested", [
    (["computer", "inspect"], True, True),
    (["inspect", "computer"], True, True),
    (["computer"], False, True),
    (["computer"], True, True),
    (["computer", "inspect"], True, False),
])
async def test_ordinary_image_history_is_independent_of_roster_consumption(
    model, source, names, prior, nested,
):
    p, wire, events = provider()
    history = []
    if prior:
        history = [
            Message(role="assistant", content=[ToolCallBlock(id="prior", name="inspect", input={})]),
            Message(role="tool", tool_call_id="prior", name="inspect", content="plain prior result"),
        ]
    content = [TextBlock(text="before"), image(source), TextBlock(text="after")]
    if nested:
        content = [ToolResultBlock(tool_call_id="target", output=[b.model_dump() for b in content])]
    history.extend([
        Message(role="assistant", content=[ToolCallBlock(
            id="target", name="computer", input={"actions": []},
        )]),
        Message(role="tool", tool_call_id="target", name="computer", content=content),
    ])
    request = ChatRequest(model=model, tools=[
        ToolSpec(name=name, parameters={"type": "object", "properties": {}})
        for name in names
    ], messages=history)
    before = request.model_dump()
    try:
        response = await p.complete(request)
        assert response.content[0].text == "OK"
        assert [path for path, _ in wire] == ["/v1/responses/input_tokens", "/v1/responses"]
        assert wire[0][1]["input"] == wire[1][1]["input"]
        assert wire[0][1]["tools"] == wire[1][1]["tools"]
        assert [t["name"] for t in wire[1][1]["tools"]] == names
        assert all(t["type"] == "function" for t in wire[1][1]["tools"])
        items = wire[1][1]["input"]
        assert [i["type"] for i in items] == (
            ["function_call", "function_call_output"] * (2 if prior else 1)
        )
        outputs = [i for i in items if i["type"] == "function_call_output"]
        assert [i["call_id"] for i in outputs] == (["prior", "target"] if prior else ["target"])
        rich = outputs[-1]["output"]
        assert [b["type"] for b in rich] == ["input_text", "input_image", "input_text"]
        assert rich[0]["text"] == "before" and rich[2]["text"] == "after"
        assert rich[1]["image_url"] == (
            URL if source == "url" else f"data:image/png;base64,{DATA}"
        )
        assert URL not in json.dumps([b for b in rich if b["type"] == "input_text"])
        assert DATA not in json.dumps([b for b in rich if b["type"] == "input_text"])
        assert URL not in json.dumps(events) and DATA not in json.dumps(events)
        assert request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_image_projection_snapshots_escaped_roster_without_hashing_names(monkeypatch):
    import amplifier_module_provider_openai as module

    p, _, _ = provider()
    # Untyped extra-body rosters are validated later. Snapshotting them must
    # not introduce a set/hash requirement before that existing boundary.
    p.extra_request_params = {"extra_body": {"tools": [
        {"type": "computer"},
        {"type": "apply_patch"},
        {"type": "function", "name": ["escaped"]},
        {"type": "function", "name": "computer"},
    ]}}
    captured = []

    class ProjectionReached(Exception):
        pass

    def capture(messages, **kwargs):
        inventory = kwargs["ordinary_names"]
        captured.append(inventory)
        assert isinstance(inventory, tuple)
        assert list(inventory) == list(inventory) == [["escaped"], "computer"]
        assert ["escaped"] in inventory and "computer" in inventory
        raise ProjectionReached()

    monkeypatch.setattr(module, "project_chat_images", capture)
    try:
        with pytest.raises(ProjectionReached):
            p._budget_params(ChatRequest(messages=[Message(role="user", content="plain")]))
        assert len(captured) == 1
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("historical", [False, True])
async def test_image_only_user_turn_is_preserved_even_before_plain_followup(historical):
    p, wire, _ = provider()
    messages = [Message(role="user", content=[image()])]
    if historical:
        messages += [Message(role="assistant", content="earlier answer"),
                     Message(role="user", content="plain followup")]
    try:
        await p.complete(ChatRequest(messages=messages))
        assert wire[-1][1]["input"][0]["content"] == [{"type": "input_image", "image_url": URL}]
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_retry_observer_has_no_json_escaped_media_echo_and_retry_policy_is_unchanged(monkeypatch):
    private_url = 'https://images.example.test/a.png?token="秘密\\fixture"'
    monkeypatch.setattr(sys.modules[__name__], "URL", private_url)
    replies = [httpx.Response(429, json={"error": {
        "message": f"limited {private_url}", "type": "rate_limit_error",
    }}), httpx.Response(200, json=body())]
    p, wire, events = provider(replies=replies)
    p._retry_config = RetryConfig(max_retries=1, initial_delay=0, max_delay=0, jitter=None)
    try:
        await p.complete(ChatRequest(messages=[Message(role="user", content=[image()])]))
        retries = [data for name, data in events if name == "provider:retry"]
        assert len(retries) == 1 and retries[0]["error_type"] == "RateLimitError"
        assert private_url not in json.loads(retries[0]["error_message"])["message"]
        assert len([path for path, _ in wire if path.endswith("/responses")]) == 2
        assert wire[1][1]["input"] == wire[-1][1]["input"]
    finally:
        await p.client.close()


@pytest.mark.parametrize("discriminator", [
    {"type": "string"}, ["string", "null"], {}, [], None, 7,
])
def test_media_walkers_preserve_schema_type_values_and_still_find_nested_media(discriminator):
    from amplifier_module_provider_openai._chat_images import (
        OMITTED,
        diagnostic_copy,
        media_values,
    )

    schema = {"properties": {"type": discriminator}}
    before = copy.deepcopy(schema)
    assert media_values(schema) == set()
    assert diagnostic_copy(schema) == before
    assert schema == before
    # A non-media discriminator must not stop recursion or disable redaction.
    nested = {"type": discriminator, "content": [
        image().model_dump(), image("base64").model_dump(),
    ]}
    echoes = media_values(nested)
    assert echoes == {URL, DATA}
    sanitized = diagnostic_copy(nested, echoes)
    assert sanitized["type"] == discriminator
    assert URL not in json.dumps(sanitized) and DATA not in json.dumps(sanitized)
    assert all(item["media"] == OMITTED for item in sanitized["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6.2-sol"])
@pytest.mark.parametrize("raw", [False, True])
async def test_ordinary_tool_schema_property_named_type_survives_count_and_completion(model, raw):
    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "description": "Ordinary business type"},
            "value": {"type": ["string", "null"]},
        },
        "required": ["type", "value"],
        "additionalProperties": False,
    }
    p, wire, events = provider()
    p.raw = raw
    tool = ToolSpec(name="inspect", parameters=schema)
    req = ChatRequest(model=model, reasoning_effort="medium", tools=[tool],
                      messages=[Message(role="user", content="Inspect the business type")])
    before = req.model_dump()
    try:
        result = await p.complete(req)
        assert [path for path, _ in wire] == ["/v1/responses/input_tokens", "/v1/responses"]
        assert wire[0][1]["tools"] == wire[1][1]["tools"]
        assert wire[1][1]["tools"][0]["parameters"] == schema
        assert wire[1][1]["reasoning"]["effort"] == "medium"
        assert result.content[0].text == "OK"
        assert req.model_dump() == before
        assert any(name == "llm:response" for name, _ in events)
    finally:
        await p.client.close()

@pytest.mark.asyncio
async def test_request_reuses_encoded_image_values_across_diagnostic_events(monkeypatch):
    from collections import Counter
    from amplifier_module_provider_openai import _chat_images

    data_url = f"data:image/png;base64,{DATA}"
    encodings = Counter()

    def counted_dumps(value, **kwargs):
        if isinstance(value, str) and value in {DATA, data_url}:
            encodings[value, kwargs.get("ensure_ascii", True)] += 1
        return json.dumps(value, **kwargs)

    monkeypatch.setattr(_chat_images, "json", SimpleNamespace(dumps=counted_dumps, loads=json.loads))
    p, wire, events = provider()
    try:
        request = ChatRequest(messages=[Message(role="user", content=[
            image("base64"), *(TextBlock(text=f"ordinary field {i}") for i in range(40))
        ])])
        await p.complete(request)
        assert {name for name, payload in events if "raw" in payload} >= {"llm:request", "llm:response"}
        assert data_url in json.dumps(wire[-1][1])
        assert DATA not in json.dumps(events)
        assert encodings == {(value, ascii_only): 1 for value in (DATA, data_url)
                             for ascii_only in (True, False)}
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("escape_slashes", [False, True])
async def test_preescaped_image_url_in_sdk_error_stays_out_of_logs_and_traceback(monkeypatch, caplog, escape_slashes):
    private_url = 'https://images.example.test/private-thumb?token="秘密\\fixture"'
    monkeypatch.setattr(sys.modules[__name__], "URL", private_url)
    escaped_url = json.dumps(private_url)[1:-1]
    if escape_slashes:
        escaped_url = escaped_url.replace("/", r"\/")
    error_body = {"error": {"type": "invalid_request_error", "message": f"bad input {escaped_url}"}}
    p, _, events = provider(replies=[httpx.Response(400, json=error_body)])
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await p.complete(ChatRequest(messages=[Message(role="user", content=[image()])]))
        assert caught.value.status_code == 400 and not caught.value.retryable
        assert json.loads(str(caught.value))["message"] == "bad input [image omitted from diagnostics]"
        assert "private-thumb" not in json.dumps(events) + caplog.text + "".join(traceback.format_exception(caught.value))
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_non_echoing_sdk_failure_does_not_expose_image_in_attached_http_request():
    p, _, _ = provider(replies=[httpx.Response(400, json={
        "error": {"type": "invalid_request_error", "message": "Invalid request."},
    })])
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await p.complete(ChatRequest(messages=[Message(role="user", content=[image("base64")])]))
        assert caught.value.status_code == 400
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_update", [False, True])
async def test_image_compaction_failure_preserves_status_without_media_or_sdk_request(with_update):
    from amplifier_core.llm_errors import LLMError
    from amplifier_module_provider_openai.configuration_updates import configuration_update_message

    model = "gpt-6-astra" if with_update else "gpt-6.2-sol"
    messages = [Message(role="user", content=[image("base64")])]
    if with_update:
        messages.extend([
            Message(role="assistant", content="Received."),
            Message(**configuration_update_message(model, "high", request_effort="low")),
            Message(role="user", content="Continue."),
        ])
    p, wire, _ = provider(replies=[httpx.Response(503, json={"error": {"message": DATA}})])
    request = ChatRequest(model=model, messages=messages, reasoning_effort="low")
    try:
        with pytest.raises(LLMError) as caught:
            await p.compact_context(request)
        assert [path for path, _ in wire] == [
            "/v1/responses/input_tokens", "/v1/responses" if with_update else "/v1/responses/compact",
        ]
        assert DATA in json.dumps(wire[-1][1])
        assert caught.value.status_code == 503 and not caught.value.retryable
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        assert not hasattr(caught.value, "request") and not hasattr(caught.value, "response")
        assert DATA not in str(caught.value)
    finally:
        await p.client.close()
