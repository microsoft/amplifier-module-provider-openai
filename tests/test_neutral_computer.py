"""Neutral v1 contracts at the real Core/SDK boundaries, without live input."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest, ChatResponse, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.neutral_computer import (
    CALLS,
    IMAGE_DROPPED,
    IMAGES,
    KEY,
    _response_model_matches,
    json_agrees,
    neutral_screenshot,
    replay_calls,
    response_calls,
    translate,
    validate_intent,
    validate_schema,
)

PNG = Path(__file__).parent / "fixtures/computer_use/valid-red-pixel.png"


def image(tint=0):
    import base64
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    # A backend fake's full screenshot, matching the real declared pixel space.
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1280, 720, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\x00" + bytes([tint]) * (1280 * 3)) * 720)) + chunk(b"IEND", b"")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}


def spec(alias="desk", operations=None, image_space=True):
    return ToolSpec(name=alias, description="Canonical desktop", parameters={
        "type": "object", "required": ["action"],
        "properties": {
            "action": {"type": "string"},
            "coordinate": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            "text": {"type": "string"},
            "repeat": {"type": "integer", "minimum": 1, "maximum": 100},
            "scroll_direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "scroll_amount": {"type": "integer"},
        }, "additionalProperties": False,
    }, type="function", computer_use={"version": 1,
        "operations": operations or ["screenshot", "mouse_move", "left_click", "right_click", "middle_click", "double_click", "type", "key"],
        "image_space": {"width": 1280, "height": 720, "unit": "pixel"} if image_space else None,
        "unavailable_operations": {},
    })


def request(tool=None, messages=None, **kwargs):
    return ChatRequest(model="gpt-6.1-sol", tools=[tool or spec()],
                       messages=messages or [Message(role="user", content="Observe")], **kwargs)


def provider(**config):
    return OpenAIProvider(api_key="offline-fixture", config={"use_streaming": False, "max_retries": 0, **config})


def block(actions=None, **overrides):
    return {"type": "computer_call", "id": "cu_fixture", "call_id": "call_fixture",
            "status": "completed", "actions": [{"type": "screenshot"}] if actions is None else actions, **overrides}


def envelope(output, model="gpt-6.1-sol", status="completed"):
    return {"id": "resp_fixture", "object": "response", "created_at": 1,
            "model": model, "status": status, "output": output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [{"type": "computer"}],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


def binding(tool=None):
    return {**validate_intent(tool or spec()), "model": "gpt-6.1-sol"}


def context(tool=None):
    tool = tool or spec()
    return {"binding": binding(tool), "legacy": False, "neutral": True,
            "declarations": {tool.name: validate_intent(tool)}, "ordinary_names": []}


def test_real_core_roundtrip_and_finalized_request_purity():
    req = request()
    original = req.model_dump()
    driver = provider(extra_request_params={"model": "gpt-5-mini"})
    params = driver._budget_params(req)
    assert params["model"] == "gpt-5-mini"
    assert params["tools"][0]["type"] == "function"
    assert req.model_dump() == original
    assert not hasattr(driver, "_neutral_computer_context")
    assert "computer_use" not in json.dumps(params)
    assert CALLS not in json.dumps(params)
    roundtrip = ChatRequest.model_validate_json(req.model_dump_json())
    assert roundtrip.tools[0].computer_use == req.tools[0].computer_use


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.6-sol", "gpt-6-astra", "gpt-6.1-sol"])
def test_neutral_compiles_to_bare_native(model):
    driver = provider()
    params, state, _ = driver._assemble_initial_responses_params(request(), model=model)
    assert params["tools"] == [{"type": "computer"}]
    assert state["neutral_computer_context"]["binding"]["alias"] == "desk"
    assert state["neutral_computer_context"]["binding"]["model"] == model
    assert "computer_use" not in json.dumps(params)


@pytest.mark.parametrize("reason", ["model", "endpoint", "image", "operations", "attachments"])
def test_supported_ordinary_fallback_and_native_required_refusal(reason):
    tool = spec(image_space=reason != "image", operations=["screenshot", "scroll"] if reason == "operations" else None)
    driver = provider(base_url="https://proxy.invalid/v1" if reason == "endpoint" else None)
    messages = [Message(role="user", content=[image(), image()])] if reason == "attachments" else None
    req = request(tool, messages=messages)
    kwargs = {"model": "gpt-4o"} if reason == "model" else {}
    wire = driver._budget_params(req, **kwargs)
    assert wire["tools"][0]["type"] == "function"
    assert wire["tools"][0]["name"] == "desk"
    assert "computer_use" not in json.dumps(wire)
    with pytest.raises(InvalidRequestError) as caught:
        driver._budget_params(req, tool_choice={"type": "computer"}, **kwargs)
    assert caught.value.code == "computer_use_native_unavailable"


@pytest.mark.parametrize("intent", [None, {}, {"version": True}, {"version": 2},
    {"version": 1, "operations": []}, {"version": 1, "operations": ["invented"]},
    {"version": 1, "operations": ["screenshot", "screenshot"]},
    {"version": 1, "operations": ["screenshot"], "image_space": {"width": True, "height": 2, "unit": "pixel"}},
    {"version": 1, "operations": ["screenshot"], "unavailable_operations": {"screenshot": "disabled"}},
    {"version": 1, "operations": ["screenshot"], "native_required": True},
])
def test_malformed_intent_fails_before_wire(intent):
    tool = spec()
    tool.computer_use = intent
    with pytest.raises(InvalidRequestError):
        provider()._budget_params(request(tool))


def test_ambiguity_and_overrides_fail_without_mutation():
    req = request()
    req.tools.append(spec("other"))
    before = req.model_dump()
    with pytest.raises(InvalidRequestError, match="multiple aliases"):
        provider()._budget_params(req)
    assert before == req.model_dump()
    for tools in ([{"type": "computer"}], [{"type": "function", "name": "desk", "parameters": {}}]):
        with pytest.raises(InvalidRequestError):
            provider(extra_request_params={"tools": tools})._budget_params(request())
    with pytest.raises(InvalidRequestError):
        provider(extra_request_params={"tools": [], "tool_choice": {"type": "computer"}})._budget_params(request())


def test_effective_client_endpoint_and_dict_transport():
    driver = provider()
    driver._client = SimpleNamespace(base_url="https://proxy.invalid/v1")
    assert driver._budget_params(request())["tools"][0]["type"] == "function"
    tool = spec().model_dump()
    wire = driver._budget_params(ChatRequest(messages=[Message(role="user", content="fixture")]), tools=[tool])
    assert wire["tools"][0]["type"] == "function"
    driver._client.base_url = "https://api.openai.com/v1/"
    assert driver._budget_params(request())["tools"] == [{"type": "computer"}]


@pytest.mark.parametrize("alias", ["computer", "screenshot", "desk"])
def test_ordinary_name_and_vendor_shaped_history_collision_control(alias):
    tool = ToolSpec(name=alias, description="Ordinary", parameters={"type": "object"})
    req = request(tool, messages=[Message(role="assistant", content=[{
        "type": "tool_call", "id": "ordinary", "name": alias,
        "input": {"actions": [{"type": "screenshot"}]},
    }]), Message(role="tool", tool_call_id="ordinary", content="ordinary text")])
    wire = provider()._budget_params(req)
    assert wire["tools"][0]["type"] == "function"
    assert [i["type"] for i in wire["input"]] == ["function_call", "function_call_output"]


@pytest.mark.parametrize("vendor,canonical", [
    ({"type": "screenshot"}, {"action": "screenshot"}),
    ({"type": "move", "x": 426, "y": 87, "keys": None}, {"action": "mouse_move", "coordinate": [426, 87]}),
    ({"type": "click", "button": "left", "x": 4, "y": 7}, {"action": "left_click", "coordinate": [4, 7]}),
    ({"type": "click", "button": "right", "x": 4, "y": 7}, {"action": "right_click", "coordinate": [4, 7]}),
    ({"type": "click", "button": "wheel", "x": 4, "y": 7}, {"action": "middle_click", "coordinate": [4, 7]}),
    ({"type": "double_click", "x": 4, "y": 7}, {"action": "double_click", "coordinate": [4, 7]}),
    ({"type": "type", "text": "exact\ntext"}, {"action": "type", "text": "exact\ntext"}),
    ({"type": "keypress", "keys": ["CTRL", "s"]}, {"action": "key", "text": "ctrl+s", "repeat": 1}),
])
def test_all_supported_members_translate_exactly(vendor, canonical):
    assert translate([vendor], binding()) == {"version": 1, "actions": [canonical], "capture_after": True}


@pytest.mark.parametrize("action", [
    {"type": "scroll", "scroll_y": 120, "scroll_x": 0, "x": 1, "y": 2},
    {"type": "drag", "path": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
    {"type": "wait"}, {"type": "move", "x": 1, "y": 2, "keys": ["SHIFT"]},
    {"type": "move", "x": True, "y": 2}, {"type": "move", "x": 1280, "y": 2},
    {"type": "move", "x": 1.5, "y": 2}, {"type": "move", "x": -1, "y": 2},
    {"type": "click", "button": "back", "x": 1, "y": 2},
    {"type": "click", "x": 1, "y": 2}, {"type": "type", "text": 7},
    {"type": "type", "text": ""}, {"type": "screenshot", "region": [0, 0, 1, 1]},
    {"type": "keypress", "keys": ["CTRL"]}, {"type": "keypress", "keys": ["a", "b"]},
    {"type": "keypress", "keys": ["ctrl+s"]}, {"type": "keypress", "keys": ["ARROWUP"]},
    {"type": "keypress", "keys": ["CTRL", "CONTROL", "s"]},
    {"type": "keypress", "keys": ["DELETE"]},
])
def test_no_guessed_semantics_in_native_actions(action):
    with pytest.raises(InvalidRequestError) as caught:
        translate([{"type": "screenshot"}, action], binding())
    assert caught.value.retryable is False


@pytest.mark.parametrize("overrides", [
    {"pending_safety_checks": [{"id": "safety", "code": "malicious_instructions"}]},
    {"status": "in_progress"}, {"call_id": ""}, {"action": {"type": "screenshot"}},
    {"actions": []}, {"opaque": "unknown"},
])
def test_pending_safety_and_malformed_batches_refused(overrides):
    with pytest.raises(InvalidRequestError):
        response_calls([block(**overrides)], context(), "gpt-6.1-sol", "completed")


def test_disabled_undeclared_familyless_and_wrong_model_refused():
    ctx = context(spec(operations=["screenshot"]))
    bad = block([{"type": "click", "x": 1, "y": 2, "button": "left"}])
    for output in ([bad], [block(), block()], [{"actions": [{"type": "screenshot"}]}],
                   [{"type": "function_call", "name": "computer", "call_id": "f", "arguments": "{}"}]):
        with pytest.raises(InvalidRequestError):
            response_calls(output, ctx, "gpt-6.1-sol", "completed")
    with pytest.raises(InvalidRequestError):
        response_calls([block()], ctx, "gpt-5.4", "completed")
    with pytest.raises(InvalidRequestError):
        response_calls([block()], ctx, "gpt-6.1-sol", "incomplete")


@pytest.mark.asyncio
async def test_sdk_object_response_core_serialization_alias_and_replay():
    from openai import AsyncOpenAI
    from openai.types.responses import Response
    driver = provider()
    wire, state, _ = driver._assemble_initial_responses_params(request())
    assert wire["tools"] == [{"type": "computer"}]
    vendor = block([{"type": "move", "x": 426, "y": 87}, {"type": "keypress", "keys": ["CTRL", "s"]}], pending_safety_checks=[])
    # Obtain the SDK's actual parsed response, not a stricter Pydantic fixture.
    # SDK 2.9 permits new wire fields through its normal forward-compatible parser.
    async def handle(http_request):
        assert json.loads(http_request.content)["tools"] == [{"type": "computer"}]
        return httpx.Response(200, json=envelope([vendor]))
    client = AsyncOpenAI(
        api_key="fixture",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    try:
        parsed = await client.responses.create(
            model="gpt-6.1-sol", input=[], tools=[{"type": "computer"}],
        )
        assert isinstance(parsed, Response)
        response = driver._convert_to_chat_response(parsed, computer_context=state["neutral_computer_context"])
    finally:
        await client.close()
    assert response.tool_calls[0].name == "desk"
    assert response.tool_calls[0].arguments["capture_after"] is True
    assert response.tool_calls[0].arguments["actions"][0]["coordinate"] == [426, 87]
    assert not driver._native_call_types
    saved = Message(role="assistant", content=response.content, metadata=response.metadata)
    saved = Message.model_validate_json(saved.model_dump_json())
    assert replay_calls([saved.model_dump()])["call_fixture"]["alias"] == "desk"
    req = request(messages=[saved, Message(role="tool", tool_call_id="call_fixture", content=[{"type": "text", "text": "final observation"}, image()])])
    original = req.model_dump()
    second = driver._budget_params(req)
    native = [i for i in second["input"] if i.get("type", "").startswith("computer_")]
    assert [i["type"] for i in native] == ["computer_call", "computer_call_output"]
    assert native[0]["actions"] == vendor["actions"]
    assert native[1]["output"]["image_url"].startswith("data:image/png;base64,")
    assert "final observation" in json.dumps(second["input"])
    assert CALLS not in json.dumps(second)
    assert KEY not in json.dumps(second)
    assert req.model_dump() == original
    # A fallback retains canonical arguments and every image, never vendor args.
    fallback = driver._budget_params(req, model="gpt-4o")
    call = next(i for i in fallback["input"] if i.get("type") == "function_call")
    assert json.loads(call["arguments"])["actions"][0]["action"] == "mouse_move"


@pytest.mark.parametrize("content", ["error text", {"success": False, "error": "halt"}, [],
    [image(), image()], [{"type": "text", "text": '{"success":false,"error":"halt"}'}, image()],
])
def test_final_image_selection_never_hides_errors(content):
    with pytest.raises(InvalidRequestError):
        neutral_screenshot(content)


def test_provenance_tampering_and_failed_history_new_user_projection():
    from amplifier_module_provider_openai import _RawResponseObject
    driver = provider()
    response = driver._convert_to_chat_response(_RawResponseObject(envelope([block()])), computer_context=context())
    saved = Message(role="assistant", content=response.content, metadata=response.metadata)
    messages = [saved, Message(role="tool", tool_call_id="call_fixture", content='{"success":false,"error":"halted"}')]
    with pytest.raises(InvalidRequestError):
        driver._budget_params(request(messages=messages))
    messages.append(Message(role="user", content="Discuss the failure, do not retry"))
    req = request(messages=messages)
    original = req.model_dump()
    wire = driver._budget_params(req)
    assert not any(i.get("type") == "computer_call" for i in wire["input"])
    assert "halted" in json.dumps(wire["input"])
    assert "originalSha256" in json.dumps(wire["input"])
    assert req.model_dump() == original
    bad = copy.deepcopy(saved.model_dump())
    bad["content"][0]["input"]["capture_after"] = False
    with pytest.raises(InvalidRequestError):
        replay_calls([bad])


def stored_response(response, storage="loop"):
    """Public response fields in the normal loop's two persisted representations."""
    response = ChatResponse.model_validate_json(response.model_dump_json())
    content = [item.model_dump() for item in response.content]
    calls = [
        {"id": tc.id, "tool": tc.name, "arguments": tc.arguments}
        for tc in (response.tool_calls or [])
    ]
    if storage == "structured":
        calls = []
    elif storage in {"legacy", "name_arguments"}:
        content = []
        if storage == "name_arguments":
            calls = [
                {"id": tc.id, "name": tc.name, "arguments": json.dumps(tc.arguments)}
                for tc in (response.tool_calls or [])
            ]
    elif storage == "both_fields":
        for call in calls:
            call["name"] = call["tool"]
            call["input"] = call["arguments"]
            call["arguments"] = json.dumps(call["arguments"])
            call["tool_call_id"] = call["call_id"] = call["id"]
    saved = Message(
        role="assistant", content=content, tool_calls=calls, metadata=response.metadata,
    )
    return Message.model_validate_json(saved.model_dump_json())


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("storage", ["structured", "legacy", "name_arguments", "loop", "both_fields"])
async def test_actual_sdk_count_create_stream_two_turns_with_no_execution(streaming, storage):
    wires = []
    async def handle(req):
        body = json.loads(req.content)
        wires.append((req.url.path, body))
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        old = any(item.get("type") == "computer_call" for item in body.get("input", []))
        payload = envelope([block(call_id="call_next" if old else "call_fixture")])
        if body.get("stream"):
            sse = "".join("data: " + json.dumps({"type": kind, "sequence_number": seq, "response": payload}) + "\n\n"
                          for seq, kind in enumerate(["response.created", "response.completed"]))
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=payload)
    sdk = openai.AsyncOpenAI(api_key="offline-fixture", max_retries=0,
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    driver = OpenAIProvider(client=sdk, config={"use_streaming": streaming, "max_retries": 0})
    try:
        tool = spec("my_desktop", operations=["screenshot", "left_click", "key"])
        first = request(tool)
        first_before = first.model_dump()
        budget = await driver.request_budget(first, context_estimate=1)
        assert budget["measurement"]["kind"] == "provider_count"
        response = await driver.complete(first)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "my_desktop"
        saved = stored_response(response, storage)
        second = request(tool, messages=[
            saved, Message(role="tool", tool_call_id="call_fixture", content=[image()]),
        ])
        second_before = second.model_dump()
        replay = replay_calls([saved.model_dump()])
        assert list(replay) == ["call_fixture"]
        assert replay["call_fixture"] == response.metadata[CALLS]["call_fixture"]
        await driver.request_budget(second, context_estimate=1)
        await driver.complete(second)
        assert first.model_dump() == first_before
        assert second.model_dump() == second_before
        assert len(wires) == 6  # two budget counts, two dispatch counts, two generations
        for _, body in wires:
            assert body["tools"] == [{"type": "computer"}]
            assert "computer_use" not in json.dumps(body)
            assert CALLS not in json.dumps(body)
        counts = [body for path, body in wires if path.endswith("input_tokens")]
        creates = [body for path, body in wires if path.endswith("responses")]
        assert counts[-1]["input"] == creates[-1]["input"]
        for body in [counts[-2], counts[-1], creates[-1]]:
            calls = [i for i in body["input"] if i.get("type") == "computer_call"]
            outputs = [i for i in body["input"] if i.get("type") == "computer_call_output"]
            assert len(calls) == len(outputs) == 1
            assert calls[0]["call_id"] == outputs[0]["call_id"] == "call_fixture"
            assert calls[0]["actions"] == block()["actions"]
            assert outputs[0]["output"]["image_url"] == neutral_screenshot([image()])[0]
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", [
    "name_tool", "tool_name", "input_arguments", "arguments_input",
    "malformed_json", "duplicate_alias", "call_identity",
    "duplicate_arguments", "duplicate_provenance", "canonical", "vendor_actions",
    "image_binding", "cross_turn",
    "bool_capture", "bool_version", "null_message", "null_record", "null_block",
    "second_identity", "orphan", "same_storage_duplicate", "nan", "non_json",
    "block_only_alias", "block_only_arguments",
    "misplaced_message", "misplaced_block", "duplicate_json_field",
])
async def test_sdk_loop_replay_consistency_refuses_before_http_without_mutation(corruption):
    wires = []

    async def handle(req):
        wires.append(req.url.path)
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        return httpx.Response(200, json=envelope([block()]))

    sdk = openai.AsyncOpenAI(
        api_key="offline", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    driver = OpenAIProvider(client=sdk, config={"use_streaming": False, "max_retries": 0})
    try:
        response = await driver.complete(request())
        saved = stored_response(response).model_dump()
        legacy = saved["tool_calls"][0]
        structured = saved["content"][0]
        if corruption == "name_tool":
            legacy["name"] = legacy["tool"]
            legacy["tool"] = "wrong"
        elif corruption == "tool_name":
            legacy["name"] = "wrong"
        elif corruption == "input_arguments":
            structured["arguments"] = json.dumps({**structured["input"], "capture_after": False})
        elif corruption == "arguments_input":
            structured["arguments"] = json.dumps(structured["input"])
            structured["input"]["capture_after"] = False
        elif corruption == "malformed_json":
            legacy["arguments"] = "{"
        elif corruption == "duplicate_alias":
            structured["name"] = "wrong"
        elif corruption == "call_identity":
            structured["id"] = "tampered"
        elif corruption == "duplicate_arguments":
            structured["input"]["capture_after"] = False
        elif corruption == "duplicate_provenance":
            structured["metadata"][KEY]["alias"] = "wrong"
        elif corruption == "canonical":
            for record in (saved["metadata"][CALLS]["call_fixture"], structured["metadata"][KEY]):
                record["canonical"]["capture_after"] = False
        elif corruption == "vendor_actions":
            for record in (saved["metadata"][CALLS]["call_fixture"], structured["metadata"][KEY]):
                record["vendor_actions"] = [{"type": "move", "x": 1, "y": 2}]
        elif corruption == "image_binding":
            for record in (saved["metadata"][CALLS]["call_fixture"], structured["metadata"][KEY]):
                record["binding"]["image_space"]["width"] = 1
        elif corruption == "bool_capture":
            legacy["arguments"] = {**legacy["arguments"], "capture_after": 1}
        elif corruption == "bool_version":
            legacy["arguments"] = {**legacy["arguments"], "version": True}
        elif corruption == "null_message":
            saved["metadata"][CALLS] = None
        elif corruption == "null_record":
            saved["metadata"][CALLS]["call_fixture"] = None
        elif corruption == "null_block":
            structured["metadata"][KEY] = None
        elif corruption == "second_identity":
            legacy["tool_call_id"] = "wrong"
        elif corruption == "orphan":
            saved["metadata"][CALLS]["orphan"] = copy.deepcopy(saved["metadata"][CALLS]["call_fixture"])
        elif corruption == "same_storage_duplicate":
            saved["tool_calls"].append(copy.deepcopy(legacy))
        elif corruption == "nan":
            legacy["arguments"] = {**legacy["arguments"], "version": float("nan")}
        elif corruption == "non_json":
            legacy["arguments"] = {**legacy["arguments"], "actions": ("not", "json")}
        elif corruption in {"block_only_alias", "block_only_arguments"}:
            saved["metadata"].pop(CALLS)
            if corruption == "block_only_alias":
                legacy["tool"] = "wrong"
            else:
                legacy["arguments"] = {**legacy["arguments"], "capture_after": False}
        elif corruption == "misplaced_message":
            saved["metadata"][KEY] = copy.deepcopy(structured["metadata"][KEY])
        elif corruption == "misplaced_block":
            structured["metadata"][CALLS] = {}
        elif corruption == "duplicate_json_field":
            legacy["arguments"] = json.dumps(legacy["arguments"]).replace('"version": 1', '"version": true, "version": 1')
        saved = Message.model_validate(saved)
        result = Message(role="tool", tool_call_id="call_fixture", content=[image()])
        messages = [saved, result]
        if corruption == "cross_turn":
            messages.extend([saved.model_copy(deep=True), result.model_copy(deep=True)])
        req = request(messages=messages)
        before = req.model_dump()
        dispatched = list(wires)
        # Both the public budgeting and completion boundaries must refuse replay.
        with pytest.raises(InvalidRequestError) as caught:
            await driver.request_budget(req, context_estimate=1)
        assert caught.value.retryable is False
        with pytest.raises(InvalidRequestError):
            await driver.complete(req)
        assert wires == dispatched
        assert req.model_dump() == before
    finally:
        await driver.close()


@pytest.mark.parametrize("alias", ["computer", "screenshot", "unknown"])
def test_loop_storage_without_provenance_stays_ordinary(alias):
    arguments = {"actions": [{"type": "screenshot"}]}
    saved = Message(
        role="assistant",
        content=[{"type": "tool_call", "id": "ordinary", "name": alias, "input": arguments}],
        tool_calls=[{"id": "ordinary", "tool": alias, "arguments": arguments}],
    )
    before = saved.model_dump()
    assert replay_calls([before]) == {}
    tool = ToolSpec(name=alias, description="Ordinary", parameters={"type": "object"})
    req = request(tool, messages=[
        saved, Message(role="tool", tool_call_id="ordinary", content="ordinary text"),
    ])
    wire = provider()._budget_params(req)
    assert [item["type"] for item in wire["input"]] == ["function_call", "function_call_output"]
    assert saved.model_dump() == before


@pytest.mark.asyncio
async def test_actual_sdk_pending_safety_never_returns_executable_call():
    wires = []
    async def handle(req):
        wires.append(req.url.path)
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        return httpx.Response(200, json=envelope([block(pending_safety_checks=[{"id": "s", "code": "malicious_instructions", "message": "confirm"}])]))
    sdk = openai.AsyncOpenAI(api_key="offline", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    driver = OpenAIProvider(client=sdk, config={"use_streaming": False, "max_retries": 0})
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await driver.complete(request())
        assert caught.value.code == "computer_use_safety_pending"
        assert len(wires) == 2
        assert not driver._native_call_types
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_cancellation_does_not_restart_sdk_generation():
    entered = asyncio.Event()
    generations = 0
    async def handle(req):
        nonlocal generations
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        generations += 1
        entered.set()
        await asyncio.Event().wait()
    sdk = openai.AsyncOpenAI(api_key="offline", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    driver = OpenAIProvider(client=sdk, config={"use_streaming": False, "max_retries": 0})
    try:
        task = asyncio.create_task(driver.complete(request()))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert generations == 1
        assert not driver._native_call_types
    finally:
        await driver.close()


def saved_call():
    from amplifier_module_provider_openai import _RawResponseObject

    response = provider()._convert_to_chat_response(
        _RawResponseObject(envelope([block()])), computer_context=context()
    )
    return Message(role="assistant", content=response.content, metadata=response.metadata)


def test_repeated_ids_stay_reserved_after_failed_history_projection():
    messages = [
        saved_call(),
        Message(role="tool", tool_call_id="call_fixture", content='{"success":false,"error":"halted"}'),
        Message(role="user", content="Discuss only"),
    ]
    _, state, _ = provider()._assemble_initial_responses_params(request(messages=messages))
    with pytest.raises(InvalidRequestError, match="duplicate"):
        response_calls([block()], state["neutral_computer_context"], "gpt-6.1-sol", "completed")


def test_duplicate_history_and_conflicting_provenance_fail():
    saved = saved_call()
    result = Message(role="tool", tool_call_id="call_fixture", content=[image()])
    for messages in ([saved, result, saved, result], [saved, result, result]):
        with pytest.raises(InvalidRequestError, match="duplicate"):
            provider()._budget_params(request(messages=messages))
    bad = copy.deepcopy(saved.model_dump())
    bad["metadata"][CALLS]["call_fixture"]["alias"] = "wrong"
    with pytest.raises(InvalidRequestError, match="conflicting"):
        replay_calls([bad])


def test_equivalent_core_storage_representations_are_one_wire_call():
    saved = saved_call().model_dump()
    saved["tool_calls"] = [{
        "id": saved["content"][0]["id"], "name": "desk",
        "arguments": saved["content"][0]["input"],
    }]
    req = request(messages=[
        Message.model_validate(saved),
        Message(role="tool", tool_call_id="call_fixture", content=[image()]),
    ])
    wire = provider()._budget_params(req)
    assert sum(item.get("type") == "computer_call" for item in wire["input"]) == 1
    assert sum(item.get("type") == "computer_call_output" for item in wire["input"]) == 1


def test_old_screenshot_uses_its_saved_dimensions_not_new_binding():
    current = spec()
    current.computer_use["image_space"]["width"] = 1920
    req = request(current, messages=[
        saved_call(), Message(role="tool", tool_call_id="call_fixture", content=[image()]),
    ])
    wire = provider()._budget_params(req)
    assert sum(item.get("type") == "computer_call_output" for item in wire["input"]) == 1


@pytest.mark.parametrize("mode", ["kwargs", "extras"])
def test_ordinary_collision_through_complete_roster_overrides(mode):
    tool = ToolSpec(name="computer", description="Ordinary", parameters={"type": "object"})
    messages = [
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "old", "name": "computer",
            "input": {"actions": [{"type": "screenshot"}]},
        }]),
        Message(role="tool", tool_call_id="old", content="ordinary textual output"),
    ]
    req = ChatRequest(messages=messages)
    driver = provider(**({"extra_request_params": {"tools": [tool.model_dump()]}} if mode == "extras" else {}))
    driver._native_call_ids.add("old")
    driver._native_call_types["old"] = "computer"
    wire = driver._budget_params(req, **({"tools": [tool]} if mode == "kwargs" else {}))
    assert [item["type"] for item in wire["input"]] == ["function_call", "function_call_output"]


def test_choices_are_compiled_or_refused_not_silently_relaxed():
    choice = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "desk"}]}
    wire = provider()._budget_params(request(tool_choice=choice))
    assert wire["tool_choice"] == {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "computer"}]}
    assert choice["tools"][0]["type"] == "function"
    with pytest.raises(InvalidRequestError):
        provider(tool_search={"mode": "namespaced"})._budget_params(
            request(tool_choice={"type": "computer"}), model="gpt-4o"
        )
    nested = {"type": "namespace", "name": "ns", "description": "fixture", "tools": [spec().model_dump()]}
    with pytest.raises(InvalidRequestError, match="nested"):
        provider(extra_request_params={"tools": [nested]})._budget_params(request())


@pytest.mark.parametrize("alias", ["apply_patch", "desk"])
def test_wrong_native_executable_family_cannot_escape_neutral_validation(alias):
    output = [{
        "type": "apply_patch_call", "call_id": "wrong",
        "operation": {"type": "delete_file", "path": "untouched"},
    }]
    with pytest.raises(InvalidRequestError):
        response_calls(output, context(spec(alias)), "gpt-6.1-sol", "completed")


@pytest.mark.parametrize("arguments", [
    {"action": "left_click", "coordinate": [1, 2]},
    {"action": "screenshot", "unexpected": 99},
    {"action": "key", "text": "ctrl+s", "repeat": True},
])
def test_fallback_validates_disabled_operation_schema_and_repeat(arguments):
    tool = spec(operations=["screenshot", "key"])
    tool.parameters["additionalProperties"] = {"type": "string"}
    _, state, _ = provider()._assemble_initial_responses_params(request(tool), model="gpt-4o")
    with pytest.raises(InvalidRequestError):
        response_calls([{
            "type": "function_call", "call_id": "ordinary",
            "name": "desk", "arguments": json.dumps(arguments),
        }], state["neutral_computer_context"], "gpt-4o", "completed")


@pytest.mark.parametrize("keyword", ["enum", "const"])
@pytest.mark.parametrize("left,right,agrees", [
    (True, 1, False), (False, 0, False), (True, 1.0, False),
    (False, 0.0, False), (True, True, True), (False, False, True),
    (1, 1.0, True), (0, 0.0, True), (None, None, True),
    ([True], [1], False), ({"value": False}, {"value": 0}, False),
    ({"nested": [{"value": True}]}, {"nested": [{"value": 1.0}]}, False),
    ([{"nested": [False]}], [{"nested": [0]}], False),
    ({"nested": [True, {"value": False}]}, {"nested": [True, {"value": False}]}, True),
    ({"nested": [1, {"value": 0.0}]}, {"nested": [1.0, {"value": 0}]}, True),
    ([1, 2], [1], False), ({"value": 1}, {"other": 1}, False),
])
def test_schema_json_comparison_recursive_semantics(keyword, left, right, agrees):
    # Arbitrary nested JSON is not structurally reachable through flat actions.
    # Check both comparison directions without relaxing the action contract.
    frozen = json.dumps([left, right])
    for value, constraint in ((left, right), (right, left)):
        assert json_agrees(value, constraint) is agrees
        schema = {keyword: [constraint] if keyword == "enum" else constraint}
        if agrees:
            validate_schema(value, schema)
        else:
            with pytest.raises(InvalidRequestError) as caught:
                validate_schema(value, schema)
            assert caught.value.code == "computer_use_response_invalid"
            assert caught.value.retryable is False
    assert json.dumps([left, right]) == frozen


@pytest.mark.parametrize("kind,value,valid", [
    ("integer", True, False), ("integer", 1.0, False), ("integer", 1, True),
    ("number", True, False), ("number", 1.0, True),
    ("boolean", 1, False), ("boolean", True, True),
])
def test_schema_json_comparison_does_not_relax_parameter_types(kind, value, valid):
    schema = {"type": kind, "enum": [True, 1.0], "const": value}
    if valid:
        validate_schema(value, schema)
    else:
        with pytest.raises(InvalidRequestError):
            validate_schema(value, schema)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True], ids=["ordinary", "native-batch"])
@pytest.mark.parametrize("keyword", ["enum", "const"])
@pytest.mark.parametrize("target", ["repeat", "action-object", "coordinate-list"])
@pytest.mark.parametrize("constraint", [True, 1, 1.0], ids=["boolean-mismatch", "integer", "float-equivalent"])
async def test_schema_json_comparison_public_sdk_response(native, keyword, target, constraint):
    tool = spec(operations=["screenshot", "key", "mouse_move"], image_space=native)
    canonical = {"action": "key", "text": "enter", "repeat": 1}
    vendor = {"type": "keypress", "keys": ["ENTER"]}
    if target == "coordinate-list":
        canonical = {"action": "mouse_move", "coordinate": [1, 2]}
        vendor = {"type": "move", "x": 1, "y": 2}
        schema = tool.parameters["properties"]["coordinate"]
        constant = [constraint, 2]
    elif target == "action-object":
        schema = tool.parameters
        constant = {**canonical, "repeat": constraint}
    else:
        schema = tool.parameters["properties"]["repeat"]
        constant = constraint
    schema[keyword] = [constant] if keyword == "enum" else constant
    output = [block([vendor])] if native else [{
        "type": "function_call", "id": "fc_schema", "call_id": "call_fixture",
        "name": tool.name, "arguments": json.dumps(canonical), "status": "completed",
    }]
    driver, wires = sdk_driver(lambda body: output)
    req = request(tool)
    options = {"auto_continue": False}
    frozen = req.model_dump_json()
    frozen_options = copy.deepcopy(options)
    returned = None
    try:
        budget = await driver.request_budget(req, context_estimate=20)
        assert budget is not None
        assert CALLS not in json.dumps(budget) and IMAGES not in json.dumps(budget)
        assert len(wires) == 1 and wires[0][0].endswith("/input_tokens")
        assert req.model_dump_json() == frozen
        if type(constraint) is bool:
            with pytest.raises(InvalidRequestError) as caught:
                returned = await driver.complete(req, request_options=options)
            assert caught.value.code == "computer_use_response_invalid"
            assert caught.value.retryable is False
            assert returned is None  # No invalid dispatchable call is returned.
        else:
            returned = await driver.complete(req, request_options=options)
            assert len(returned.tool_calls) == 1
            call = returned.tool_calls[0]
            assert call.name == tool.name and call.id == "call_fixture"
            expected = {"version": 1, "actions": [canonical], "capture_after": True} if native else canonical
            assert json_agrees(call.arguments, expected)
        # Schema validation follows generation: count + generation did happen.
        # An invalid reply must not be mistaken for a pre-HTTP refusal.
        assert [path for path, _ in wires] == [
            "/v1/responses/input_tokens", "/v1/responses/input_tokens", "/v1/responses",
        ]
        expected_type = "computer" if native else "function"
        assert all(body["tools"][0]["type"] == expected_type for _, body in wires)
        assert all(CALLS not in json.dumps(body) and IMAGES not in json.dumps(body) for _, body in wires)
    finally:
        await driver.close()
    assert req.model_dump_json() == frozen  # Includes caller tools and history.
    assert options == frozen_options


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", ["enum", "const"])
@pytest.mark.parametrize("value", [True, 1, 1.0], ids=["boolean", "integer", "float"])
async def test_schema_json_comparison_public_boolean_control(keyword, value):
    tool = spec(image_space=False)
    tool.parameters["properties"]["confirmed"] = {
        "type": "boolean", keyword: [True] if keyword == "enum" else True,
    }
    canonical = {"action": "screenshot", "confirmed": value}
    driver, wires = sdk_driver(lambda body: [{
        "type": "function_call", "id": "fc_boolean", "call_id": "boolean",
        "name": tool.name, "arguments": json.dumps(canonical), "status": "completed",
    }])
    req = request(tool)
    frozen = req.model_dump_json()
    try:
        if type(value) is bool:
            response = await driver.complete(req)
            assert len(response.tool_calls) == 1
            assert json_agrees(response.tool_calls[0].arguments, canonical)
        else:
            with pytest.raises(InvalidRequestError):
                await driver.complete(req)
        assert [path for path, _ in wires] == ["/v1/responses/input_tokens", "/v1/responses"]
    finally:
        await driver.close()
    assert req.model_dump_json() == frozen


def test_crc_valid_png_without_pixel_rows_cannot_be_a_screenshot():
    import base64
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1280, 720, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b"")
    with pytest.raises(InvalidRequestError, match="pixel rows"):
        neutral_screenshot("data:image/png;base64," + base64.b64encode(data).decode())


def test_current_batch_requires_full_png_pixels_without_coercing_jpeg():
    import base64

    pixel = "data:image/png;base64," + base64.b64encode(PNG.read_bytes()).decode()
    with pytest.raises(InvalidRequestError, match="full image space"):
        neutral_screenshot(pixel, image_space={"width": 1280, "height": 720})
    jpeg = (Path(__file__).parent / "assets/macbeth-witches-trio.jpg").read_bytes()
    from amplifier_module_provider_openai._computer_result import validate_image

    width, height = validate_image(jpeg, "image/jpeg")
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    with pytest.raises(InvalidRequestError, match="verified PNG"):
        neutral_screenshot(url, image_space={"width": width, "height": height})
    # Ordinary/legacy image handling still retains JPEG container support.
    assert validate_image(jpeg, "image/jpeg") == (width, height)


def test_dimension_failure_projects_only_after_a_real_new_user_turn():
    import base64

    bad_image = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
        "data": base64.b64encode(PNG.read_bytes()).decode()}}
    messages = [saved_call(), Message(role="tool", tool_call_id="call_fixture", content=[bad_image])]
    with pytest.raises(InvalidRequestError, match="full image space"):
        provider()._budget_params(request(messages=messages))
    messages.append(Message(role="user", content="Discuss the failed screenshot"))
    wire, state, _ = provider()._assemble_initial_responses_params(request(messages=messages))
    assert not any(item.get("type") == "computer_call" for item in wire["input"])
    assert "originalSha256" in json.dumps(wire["input"])
    assert "call_fixture" in state["neutral_computer_context"]["historical_ids"]


@pytest.mark.asyncio
async def test_concurrent_generations_keep_request_local_aliases():
    entered = 0
    ready = asyncio.Event()

    async def handle(req):
        nonlocal entered
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        entered += 1
        if entered == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), 5)
        return httpx.Response(200, json=envelope([block(call_id=f"concurrent_{entered}")]))

    sdk = openai.AsyncOpenAI(api_key="offline", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    driver = OpenAIProvider(client=sdk, config={"use_streaming": False, "max_retries": 0})
    try:
        alpha, beta = await asyncio.gather(driver.complete(request(spec("alpha"))), driver.complete(request(spec("beta"))))
        assert alpha.tool_calls[0].name == "alpha"
        assert beta.tool_calls[0].name == "beta"
        assert not hasattr(driver, "_neutral_computer_context")
    finally:
        await driver.close()


def test_sdk_extra_body_final_model_and_tools_are_consumed_before_projection():
    driver = provider(extra_request_params={"extra_body": {"model": "gpt-4o", CALLS: {"must": "strip"}}})
    wire, state, _ = driver._assemble_initial_responses_params(request())
    assert wire["model"] == "gpt-4o"
    assert wire["tools"][0]["type"] == "function"
    assert "extra_body" not in wire
    assert CALLS not in wire
    assert state["neutral_computer_context"]["binding"] is None
    tool = spec().model_dump()
    wire = provider(extra_request_params={"extra_body": {"tools": [tool]}})._budget_params(request())
    assert wire["tools"] == [{"type": "computer"}]
    assert "computer_use" not in json.dumps(wire)
    driver = provider(extra_request_params={"extra_body": {"max_output_tokens": 8192}})
    assert driver._budget_params(request(max_output_tokens=64))["max_output_tokens"] == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_continue", [True, False])
async def test_text_incomplete_continuation_keeps_neutral_binding(auto_continue):
    generations = []

    async def handle(req):
        body = json.loads(req.content)
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 10})
        generations.append(body)
        if len(generations) == 1:
            payload = envelope([{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                                 "content": [{"type": "output_text", "text": "Working", "annotations": []}]}], status="incomplete")
            payload["incomplete_details"] = {"reason": "max_output_tokens"}
        else:
            payload = envelope([block()])
        return httpx.Response(200, json=payload)

    sdk = openai.AsyncOpenAI(api_key="offline", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    driver = OpenAIProvider(client=sdk, config={"use_streaming": False, "max_retries": 0, "auto_continue": auto_continue})
    try:
        response = await driver.complete(request())
        if auto_continue:
            assert len(generations) == 2
            assert response.tool_calls[0].name == "desk"
            assert response.metadata["openai:continuation_count"] == 1
            assert all(body["tools"] == [{"type": "computer"}] for body in generations)
            assert "computer_use" not in json.dumps(generations)
        else:
            assert len(generations) == 1
            assert response.finish_reason == "length"
            assert not response.tool_calls
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_incomplete_native_batch_is_not_continued_or_executed():
    generations = 0

    async def handle(req):
        nonlocal generations
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        generations += 1
        payload = envelope([block()], status="incomplete")
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
        return httpx.Response(200, json=payload)

    sdk = openai.AsyncOpenAI(api_key="offline", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    driver = OpenAIProvider(client=sdk, config={"use_streaming": False, "max_retries": 0})
    try:
        with pytest.raises(InvalidRequestError):
            await driver.complete(request())
        assert generations == 1
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_native_adapter_full_and_delta_views_keep_neutral_alias(monkeypatch):
    from unittest.mock import AsyncMock

    from amplifier_module_provider_openai import _RawResponseObject
    from amplifier_module_provider_openai.native import NativeResponsesProvider

    owner = SimpleNamespace(
        context=SimpleNamespace(get_messages=AsyncMock(return_value=[])),
        runtime=SimpleNamespace(emit=AsyncMock()),
        native_job=lambda identity: None,
        config={},
    )
    driver = NativeResponsesProvider.wrap(
        provider(default_model="gpt-6-astra"), owner_getter=lambda: owner
    )
    calls = []

    async def transport(params, **kwargs):
        calls.append(copy.deepcopy(params))
        return _RawResponseObject(envelope([block(call_id=f"native_{len(calls)}")], model="gpt-6-astra"))

    monkeypatch.setattr(driver, "_native_response", transport)
    try:
        first = await driver.complete(request(), model="gpt-6-astra")
        second = request(messages=[
            Message(role="assistant", content=first.content, metadata=first.metadata),
            Message(role="tool", tool_call_id="native_1", content=[image()]),
        ])
        final = await driver.complete(second, model="gpt-6-astra")
        assert final.tool_calls[0].name == "desk"
        assert list(final.metadata[IMAGES]) == ["native_1"]
        assert len(calls) == 2
        assert calls[1]["tools"] == [{"type": "computer"}]
        assert sum(i.get("type") == "computer_call_output" for i in calls[1]["input"]) == 1
        assert driver._native_computer_context == {}
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_native_neutral_alias_never_starts_an_async_job(monkeypatch):
    from unittest.mock import AsyncMock

    from amplifier_module_provider_openai.native import NativeResponsesProvider

    sent = []
    owner = SimpleNamespace(config={"background_delegate": True},
        start_native_job=AsyncMock(), runtime=SimpleNamespace(emit=AsyncMock()))
    driver = NativeResponsesProvider.wrap(provider(), owner_getter=lambda: owner)
    driver.owner = owner
    declaration = spec("delegate", operations=["screenshot", "scroll"])
    wire, state, _ = driver._assemble_initial_responses_params(request(declaration), model="gpt-6-astra")
    driver._native_computer_context = state["neutral_computer_context"]
    events = iter([
        {"type": "response.created", "response": {"id": "native"}},
        {"type": "response.output_item.done", "item": {
            "type": "function_call", "name": "delegate", "call_id": "async_bad",
            "arguments": '{"action":"left_click","coordinate":[1,2]}', "async": True,
        }},
    ])

    async def send(value):
        sent.append(json.loads(value))

    async def recv():
        return json.dumps(next(events))

    driver.socket = SimpleNamespace(send=send, recv=recv, close=AsyncMock())
    monkeypatch.setattr(driver, "_connect", AsyncMock())
    try:
        with pytest.raises(InvalidRequestError, match="asynchronous delegation"):
            await driver._native_response(wire)
        owner.start_native_job.assert_not_awaited()
        assert "async" not in sent[0]["tools"][0]
    finally:
        await driver.close()


def terminal_output():
    return [{"type": "message", "id": "m_terminal", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Observed.", "annotations": []}]}]


def sdk_driver(output_factory, *, streaming=False, response_factory=None):
    """Real SDK transport with an inspectable sequence, never a private replay seed."""
    wires = []

    async def handle(req):
        body = json.loads(req.content)
        wires.append((req.url.path, body))
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        payload = response_factory(body) if response_factory else envelope(output_factory(body))
        if body.get("stream"):
            sse = "".join("data: " + json.dumps({"type": kind, "sequence_number": seq, "response": payload}) + "\n\n"
                          for seq, kind in enumerate(["response.created", "response.completed"]))
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=payload)

    sdk = openai.AsyncOpenAI(api_key="offline", max_retries=0,
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    return OpenAIProvider(client=sdk, config={"use_streaming": streaming, "max_retries": 0}), wires


@pytest.mark.parametrize("requested,returned", [
    ("", ""), (None, None), (55, 55), ([], []),
    ("gpt-5.5", None), ("gpt-5.5", False), ("gpt-5.5", {}),
])
def test_response_model_matches_requires_nonempty_strings(requested, returned):
    assert _response_model_matches(requested, returned) is False


def test_response_model_alias_resolution_requires_native_capability(monkeypatch):
    from amplifier_module_provider_openai import neutral_computer
    monkeypatch.setattr(neutral_computer, "get_capabilities",
                        lambda model: SimpleNamespace(supports_native_computer_use=False))
    assert _response_model_matches("gpt-5.5", "gpt-5.5-2026-04-23") is False
    assert _response_model_matches("gpt-5.5", "gpt-5.5") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("requested,returned", [
    ("gpt-5.5", "gpt-5.5-2026-04-23"),
    ("gpt-5.5", "gpt-5.5-2024-02-29"),
    ("gpt-5.5", "gpt-5.5"),
    ("gpt-5.5-2026-04-23", "gpt-5.5-2026-04-23"),
])
async def test_actual_sdk_model_alias_snapshot_reply(streaming, reasoning, requested, returned):
    output = [block()]
    assert "pending_safety_checks" not in output[0]
    if reasoning:
        output.insert(0, {"type": "reasoning", "id": "rs_synthetic", "summary": [],
                          "encrypted_content": "synthetic-offline-opaque"})
    payload = envelope(output, model=returned)
    payload["incomplete_details"] = None
    driver, wires = sdk_driver(None, streaming=streaming, response_factory=lambda body: payload)
    req = request().model_copy(update={"model": requested})
    before = req.model_dump()
    try:
        response = await driver.complete(req)
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert (call.id, call.name, call.arguments) == (
            "call_fixture", "desk",
            {"version": 1, "actions": [{"action": "screenshot"}], "capture_after": True},
        )
        saved = stored_response(response)
        record = replay_calls([saved.model_dump()])["call_fixture"]
        assert record["binding"]["model"] == requested
        assert record["vendor_actions"] == [{"type": "screenshot"}]
        assert len(wires) == 2  # mandatory count, then one generation
        assert all(body["model"] == requested for _, body in wires)
        assert all(body["tools"] == [{"type": "computer"}] for _, body in wires)
    finally:
        await driver.close()
    assert req.model_dump() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("requested,model_fields", [
    ("gpt-5.5-2026-04-23", {"model": "gpt-5.5-2026-04-24"}),
    ("gpt-5.5-2026-04-23", {"model": "gpt-5.5"}),
    ("gpt-5.5", {"model": "gpt-5.5-pro-2026-04-23"}),
    ("gpt-5.5", {"model": "gpt-5.5-mini-2026-04-23"}),
    ("gpt-5.5", {"model": "gpt-5.5-chat-2026-04-23"}),
    ("gpt-5.5", {"model": "gpt-5.4-2026-04-23"}),
    ("gpt-5.5", {"model": "gpt-6.1-sol"}),
    ("gpt-5.5", {"model": "gpt-5.50-2026-04-23"}),
    ("gpt-5.5", {"model": "xgpt-5.5-2026-04-23"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-02-29"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-04-31"}),
    ("gpt-5.5", {"model": "gpt-5.5-0000-01-01"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-00-23"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-04-00"}),
    ("gpt-5.5", {"model": "gpt-5.5-20260423"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026/04/23"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-4-23"}),
    ("gpt-5.5", {"model": "gpt-5.5-２０２６-０４-２３"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-04-23\n"}),
    ("gpt-5.5", {"model": "gpt-5.5-2026-04-23-pro"}),
    ("gpt-5.5", {"model": ""}),
    ("gpt-5.5", {}),
    ("gpt-5.5", {"model": None}),
    ("gpt-5.5", {"model": 55}),
    ("gpt-5.4", {"model": "gpt-5.4-2026-04-23"}),
    ("gpt-4o", {"model": "gpt-5.5-2026-04-23"}),
])
async def test_actual_sdk_model_alias_snapshot_invalid_reply(streaming, requested, model_fields):
    payload = envelope([block()])
    payload.pop("model")
    payload.update(model_fields)
    driver, wires = sdk_driver(None, streaming=streaming, response_factory=lambda body: payload)
    req = request().model_copy(update={"model": requested})
    before = req.model_dump()
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await driver.complete(req)
        assert caught.value.retryable is False
        assert caught.value.code == "computer_use_response_invalid"
        assert len(wires) == 2
    finally:
        await driver.close()
    assert req.model_dump() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status,overrides", [
    ("incomplete", {}),
    ("in_progress", {}),
    ("failed", {}),
    ("cancelled", {}),
    ("completed", {"status": "in_progress"}),
    ("completed", {"actions": []}),
    ("completed", {"action": {"type": "screenshot"}}),
    ("completed", {"pending_safety_checks": [
        {"id": "s", "code": "malicious_instructions", "message": "confirm"},
    ]}),
])
async def test_actual_sdk_model_alias_snapshot_keeps_status_batch_safety_guards(streaming, status, overrides):
    payload = envelope([block(**overrides)], model="gpt-5.5-2026-04-23", status=status)
    if status == "incomplete":
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    driver, wires = sdk_driver(None, streaming=streaming, response_factory=lambda body: payload)
    req = request().model_copy(update={"model": "gpt-5.5"})
    before = req.model_dump()
    try:
        with pytest.raises(InvalidRequestError) as caught:
            await driver.complete(req)
        assert caught.value.retryable is False
        assert len(wires) == 2  # no continuation/retry of an invalid native batch
    finally:
        await driver.close()
    assert req.model_dump() == before


def recency_view(history, omission="list"):
    """Source evidence: mirror only the supported recency3 transformation.

    No actual Session.execute acceptance is claimed by this provider-local test.
    """
    view = [Message.model_validate_json(m.model_dump_json()) for m in history]
    results = [m for m in view if m.role == "tool"]
    for result in results[:-3]:
        original = result.model_dump()["content"]
        if isinstance(original, list) and any(b.get("type") == "image" for b in original):
            references = [b["text"] for b in original if b.get("type") == "text"]
            text = references[0] + IMAGE_DROPPED
            result.content = text if omission == "string" else [{"type": "text", "text": text}]
    return view


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("omission", ["list", "string"])
@pytest.mark.parametrize("adapter", [False, True])
async def test_source_four_then_eight_owned_captures_public_roundtrip_and_restart(streaming, omission, adapter):
    import base64
    import hashlib

    from amplifier_module_provider_openai.native import NativeResponsesProvider

    captures = 0
    terminal_at = 4
    expected = {}
    history = [Message(role="user", content="Observe four synthetic frames")]

    def outputs(body):
        nonlocal captures
        if captures == terminal_at:
            return terminal_output()
        captures += 1
        return [block(call_id=f"capture_{captures}")]

    driver, wires = sdk_driver(outputs, streaming=streaming)
    if adapter:
        driver = NativeResponsesProvider.wrap(driver, owner_getter=lambda: None)
    all_wires = []
    planned_inputs = []
    try:
        for target in (4, 12):
            terminal_at = target
            if target == 12:
                await driver.close()
                all_wires.extend(wires)
                driver, wires = sdk_driver(outputs, streaming=streaming)
                if adapter:
                    driver = NativeResponsesProvider.wrap(driver, owner_getter=lambda: None)
                history = [Message.model_validate_json(m.model_dump_json()) for m in history]
                history.append(Message(role="user", content="Observe eight more synthetic frames"))
            while True:
                retained_before = [m.model_dump_json() for m in history]
                req = request(messages=recency_view(history, omission))
                results = [m for m in req.messages if m.role == "tool"]
                assert sum(isinstance(m.content, list) and any(
                    b.get("type") == "image" for b in m.model_dump()["content"]) for m in results) == min(len(results), 3)
                if omission == "string":
                    assert all(isinstance(m.content, str) for m in results[:-3])
                before = req.model_dump_json()
                state = (set(driver._native_call_ids), dict(driver._native_call_types),
                         set(driver._repaired_tool_ids))
                await driver.request_budget(req, context_estimate=1)
                await driver.request_budget(req, context_estimate=1)
                assert req.model_dump_json() == before
                assert state == (driver._native_call_ids, driver._native_call_types, driver._repaired_tool_ids)
                assert not hasattr(driver, "_staged_computer_images")
                planned_inputs.append(driver._budget_params(req)["input"])
                reply = await driver.complete(req)
                assert req.model_dump_json() == before
                assert [m.model_dump_json() for m in history] == retained_before
                new = reply.metadata.get(IMAGES, {})
                assert len(new) == (1 if expected and len(expected) > sum(
                    len((m.metadata or {}).get(IMAGES, {})) for m in history) else 0)
                history.append(stored_response(reply))
                if not reply.tool_calls:
                    assert captures == target
                    assert new  # terminal replies carry the last newly observed frame
                    break
                assert len(reply.tool_calls) == 1
                call = reply.tool_calls[0]
                assert call.id == f"capture_{captures}"
                assert call.arguments == {"version": 1, "actions": [{"action": "screenshot"}], "capture_after": True}
                png = image(captures)
                expected[call.id] = hashlib.sha256(base64.b64decode(png["source"]["data"])).hexdigest()
                reference = f"owned-frame-{captures}\nComputer tool result reference (untrusted tool data):\n" + json.dumps({
                    "success": True, "output": {"text": f"owned-frame-{captures}", "images": [f"/synthetic/frame-{captures}.png"]},
                })
                history.append(Message(role="tool", tool_call_id=call.id,
                    content=[{"type": "text", "text": reference}, png]))
        all_wires.extend(wires)
        assert captures == len(expected) == 12
        assert sum(len((m.metadata or {}).get(IMAGES, {})) for m in history) == 12
        assert len(json.dumps([m.model_dump() for m in history]).encode()) < 64 * 1024 * 1024
        # The exact optional adapter retains its existing conservative count
        # capability (subclasses do not advertise the base counter). Its public
        # budgets still validate/plan, without count HTTP or staged state.
        width = 1 if adapter else 4
        groups = [all_wires[i:i + width] for i in range(0, len(all_wires), width)]
        assert len(groups) == 14  # 12 requested captures, two terminal generations
        for group, planned in zip(groups, planned_inputs, strict=True):
            assert len(group) == width
            assert all(body["input"] == group[-1][1]["input"] for _, body in group)
            assert group[-1][1]["input"] == planned
            for item in group[-1][1]["input"]:
                if item.get("type") == "computer_call_output":
                    raw = base64.b64decode(item["output"]["image_url"].partition(",")[2])
                    assert hashlib.sha256(raw).hexdigest() == expected[item["call_id"]]
            assert IMAGES not in json.dumps(group[-1][1])
    finally:
        await driver.close()


async def observed_history(driver):
    first = await driver.complete(request())
    history = [stored_response(first), Message(role="tool", tool_call_id="call_fixture",
        content=[{"type": "text", "text": "owned reference"}, image(1)])]
    observed = await driver.complete(request(messages=history))
    assert list(observed.metadata[IMAGES]) == ["call_fixture"]
    history.append(stored_response(observed))
    return history


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary", ["tool_call_id", "call_id"])
@pytest.mark.parametrize("turn", ["same", "earlier", "later"])
@pytest.mark.parametrize("storage", ["legacy", "structured"])
async def test_native_secondary_identity_cannot_hide_behind_unknown_primary(secondary, turn, storage):
    driver, wires = sdk_driver(lambda body: [block()])
    try:
        reply = await driver.complete(request())
        saved = stored_response(reply).model_dump()
        field = "tool_calls" if storage == "legacy" else "content"
        shadow = copy.deepcopy(saved[field][0])
        shadow.pop("metadata", None)
        shadow["id"] = "unknown_primary"
        shadow[secondary] = "call_fixture"
        result = Message(role="tool", tool_call_id="call_fixture", content=[image()])
        if turn == "same":
            saved[field].append(shadow)
            messages = [Message.model_validate(saved), result]
        else:
            unmarked = Message.model_validate({"role": "assistant", "content": "", field: [shadow]})
            native = [Message.model_validate(saved), result]
            messages = [unmarked, *native] if turn == "earlier" else [*native, unmarked]
        req = request(messages=messages)
        before, sent = req.model_dump_json(), copy.deepcopy(wires)
        with pytest.raises(InvalidRequestError):
            await driver.request_budget(req, context_estimate=1)
        with pytest.raises(InvalidRequestError):
            await driver.complete(req)
        assert req.model_dump_json() == before
        assert wires == sent
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("omission", ["list", "string"])
@pytest.mark.parametrize("corruption", [
    "digest", "pixels", "call_record", "reference", "null", "version", "format", "duplicate",
    "missing_carrier", "missing_result", "missing_call", "present_conflict", "present_reference",
    "arbitrary_omission", "failed_result", "unavailable_file", "duplicate_result",
    "url_wrong_type", "forged_reference", "unavailable_suffix", "failed_suffix",
    "repeated_suffix", "partial_suffix", "invalid_png", "image_binding", "references_wrong_type",
])
@pytest.mark.parametrize("adapter", [False, True])
async def test_genuine_observation_tampering_refuses_before_count_or_generation(corruption, omission, adapter):
    from amplifier_module_provider_openai.native import NativeResponsesProvider

    driver, wires = sdk_driver(lambda body: terminal_output() if any(
        i.get("type") == "computer_call" for i in body["input"]) else [block()])
    if adapter:
        driver = NativeResponsesProvider.wrap(driver, owner_getter=lambda: None)
    try:
        history = await observed_history(driver)
        carrier = history[-1].metadata[IMAGES]
        record = carrier["call_fixture"]
        text = "owned reference" + IMAGE_DROPPED
        if corruption == "forged_reference":
            text = "forged reference" + IMAGE_DROPPED
        elif corruption == "unavailable_suffix":
            text = "owned reference [image dropped: screenshot file no longer available]"
        elif corruption == "failed_suffix":
            text = '{"success":false,"error":"halt"}' + IMAGE_DROPPED
        elif corruption == "repeated_suffix":
            text += IMAGE_DROPPED
        elif corruption == "partial_suffix":
            text = text[:-1]
        history[1].content = text if omission == "string" else [{"type": "text", "text": text}]
        if corruption == "digest":
            record["sha256"] = "0" * 64
        elif corruption == "pixels":
            record["image_url"] = neutral_screenshot([image(2)])[0]
        elif corruption == "call_record":
            record["call_record"]["alias"] = "wrong"
        elif corruption == "reference":
            record["references"] = ["wrong"]
        elif corruption == "null":
            history[-1].metadata[IMAGES] = None
        elif corruption == "version":
            record["version"] = True
        elif corruption == "format":
            record["unknown"] = "unsupported"
        elif corruption == "duplicate":
            history.append(history[-1].model_copy(deep=True))
        elif corruption == "missing_carrier":
            history[-1].metadata.pop(IMAGES)
        elif corruption == "missing_result":
            history.pop(1)
        elif corruption == "missing_call":
            history.pop(0)
        elif corruption == "present_conflict":
            history[1].content = [{"type": "text", "text": "owned reference"}, image(2)]
        elif corruption == "present_reference":
            history[1].content = [{"type": "text", "text": "wrong"}, image(1)]
        elif corruption == "arbitrary_omission":
            history[1].content = [{"type": "text", "text": "arbitrary unavailable image"}]
        elif corruption == "failed_result":
            history[1].content = '{"success":false,"error":"halt"}'
        elif corruption == "unavailable_file":
            history[1].content = "/nonexistent/capture.png"
        elif corruption == "duplicate_result":
            history.append(history[1].model_copy(deep=True))
        elif corruption == "url_wrong_type":
            record["image_url"] = [image(1)]
        elif corruption == "invalid_png":
            import base64
            import hashlib
            raw = b"not PNG pixels"
            record["image_url"] = "data:image/png;base64," + base64.b64encode(raw).decode()
            record["sha256"] = hashlib.sha256(raw).hexdigest()
        elif corruption == "image_binding":
            record["call_record"]["binding"]["image_space"]["width"] = 1
        elif corruption == "references_wrong_type":
            record["references"] = "owned reference"
        req = request(messages=history)
        before = req.model_dump_json()
        sent = copy.deepcopy(wires)
        with pytest.raises(InvalidRequestError):
            await driver.request_budget(req, context_estimate=1)
        with pytest.raises(InvalidRequestError):
            await driver.complete(req)
        assert req.model_dump_json() == before
        assert wires == sent
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [False, True])
async def test_ordinary_string_omission_never_hydrates_a_native_image(adapter):
    from amplifier_module_provider_openai.native import NativeResponsesProvider

    driver, wires = sdk_driver(lambda body: terminal_output())
    if adapter:
        driver = NativeResponsesProvider.wrap(driver, owner_getter=lambda: None)
    try:
        req = request(ToolSpec(name="desk", description="Ordinary", parameters={"type": "object"}), messages=[
            Message(role="assistant", content="", tool_calls=[{"id": "ordinary", "tool": "desk", "arguments": {}}]),
            Message(role="tool", tool_call_id="ordinary", content="owned reference" + IMAGE_DROPPED),
        ])
        before = req.model_dump_json()
        await driver.request_budget(req, context_estimate=1)
        reply = await driver.complete(req)
        assert IMAGES not in (reply.metadata or {})
        assert req.model_dump_json() == before
        for _, body in wires:
            assert not any(i.get("type") in {"computer_call", "computer_call_output"} for i in body["input"])
            assert IMAGE_DROPPED in json.dumps(body["input"])
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_observations_are_request_local_for_disjoint_histories_reusing_ids():
    driver, wires = sdk_driver(lambda body: terminal_output() if any(
        i.get("type") == "computer_call" for i in body["input"]) else [block()])
    try:
        alpha = await driver.complete(request())
        beta = await driver.complete(request())
        histories = [
            [stored_response(reply), Message(role="tool", tool_call_id="call_fixture",
                content=[{"type": "text", "text": "owned reference"}, image(tint)])]
            for reply, tint in ((alpha, 1), (beta, 2))
        ]
        replies = await asyncio.gather(*(driver.complete(request(messages=h)) for h in histories))
        assert replies[0].metadata[IMAGES]["call_fixture"]["sha256"] != replies[1].metadata[IMAGES]["call_fixture"]["sha256"]
        for history, reply in zip(histories, replies):
            history.append(stored_response(reply))
            history[1].content = [{"type": "text", "text": "owned reference" + IMAGE_DROPPED}]
            await driver.complete(request(messages=history))
        creates = [body for path, body in wires if not path.endswith("input_tokens")]
        assert creates[-2]["input"] != creates[-1]["input"]
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_observation_aggregate_boundary_counts_persisted_plus_new(monkeypatch):
    import amplifier_module_provider_openai.neutral_computer as neutral
    assert neutral.MAX_OBSERVATION_BYTES == 64 * 1024 * 1024
    driver, wires = sdk_driver(lambda body: terminal_output() if any(
        i.get("type") == "computer_call" for i in body["input"]) else [block()])
    try:
        history = await observed_history(driver)
        records = history[-1].metadata[IMAGES]
        size = len(json.dumps(records, ensure_ascii=True, allow_nan=False).encode())
        monkeypatch.setattr(neutral, "MAX_OBSERVATION_BYTES", size)
        await driver.request_budget(request(messages=history), context_estimate=1)
        new_driver, new_wires = sdk_driver(lambda body: [block(call_id="call_new")])
        try:
            genuine = await new_driver.complete(request())
        finally:
            await new_driver.close()
        with_new = [*history, stored_response(genuine),
                    Message(role="tool", tool_call_id="call_new",
                            content=[{"type": "text", "text": "new reference"}, image(2)])]
        sent = copy.deepcopy(wires)
        with pytest.raises(InvalidRequestError, match="64MiB"):
            await driver.request_budget(request(messages=with_new), context_estimate=1)
        assert wires == sent
        assert len(new_wires) == 2
        monkeypatch.setattr(neutral, "MAX_OBSERVATION_BYTES", size - 1)
        sent = copy.deepcopy(wires)
        req = request(messages=history)
        before = req.model_dump_json()
        with pytest.raises(InvalidRequestError, match="64MiB") as caught:
            await driver.request_budget(req, context_estimate=1)
        assert caught.value.code == "computer_images_overflow"
        with pytest.raises(InvalidRequestError):
            await driver.complete(req)
        assert before == req.model_dump_json()
        assert wires == sent
    finally:
        await driver.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [False, True])
async def test_actual_64mib_aggregate_overflow_refuses_new_observation_before_http(adapter):
    from amplifier_module_provider_openai.native import NativeResponsesProvider

    driver, wires = sdk_driver(lambda body: [block()])
    if adapter:
        driver = NativeResponsesProvider.wrap(driver, owner_getter=lambda: None)
    try:
        first = await driver.complete(request())
        req = request(messages=[stored_response(first),
            Message(role="tool", tool_call_id="call_fixture", content=[
                {"type": "text", "text": "x" * (64 * 1024 * 1024)}, image(1)])])
        sent = copy.deepcopy(wires)
        with pytest.raises(InvalidRequestError) as caught:
            await driver.request_budget(req, context_estimate=1)
        assert caught.value.code == "computer_images_overflow"
        with pytest.raises(InvalidRequestError):
            await driver.complete(req)
        assert len(req.messages[1].content[0].text) == 64 * 1024 * 1024
        assert wires == sent
        assert not hasattr(driver, "_staged_computer_images")
    finally:
        await driver.close()


def test_ordinary_legacy_call_is_not_deduplicated_against_a_text_id():
    driver = provider()
    messages = [{"role": "assistant", "content": [
        {"type": "text", "text": "ordinary evidence", "id": "ordinary"}],
        "tool_calls": [{"id": "ordinary", "tool": "lookup", "arguments": {"value": 1}}]},
        {"role": "tool", "tool_call_id": "ordinary", "content": "found"}]
    wire = driver._convert_messages(messages)
    assert sum(i.get("type") == "function_call" for i in wire) == 1
    assert sum(i.get("type") == "function_call_output" for i in wire) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("omission", ["list", "string"])
async def test_owned_native_observations_validate_full_history_before_delta(omission):
    """Public native parsing/planning, not monkeypatched conversion or replay maps."""
    import base64
    import hashlib
    from unittest.mock import AsyncMock

    from amplifier_module_provider_openai.native import NativeResponsesProvider

    class CountedNativeFixture(NativeResponsesProvider):
        # Explicit test-only opt-in: known official-route SDK MockTransport.
        # Production adapter/subclass count capability is not changed.
        def _uses_standard_openai_endpoint(self):
            return str(self.client.base_url) == "https://api.openai.com/v1/"

    history = [Message(role="user", content="Observe four synthetic frames")]
    captures, terminal_at = 0, 4
    expected, published, sent = {}, {}, []

    async def get_messages():
        return [m.model_dump() for m in history]

    owner = SimpleNamespace(context=SimpleNamespace(get_messages=get_messages),
        runtime=SimpleNamespace(emit=AsyncMock()), native_job=lambda identity: None,
        config={}, start_native_job=AsyncMock())
    original, counts = sdk_driver(lambda body: terminal_output())
    driver = CountedNativeFixture.wrap(original, owner_getter=lambda: owner)
    events = asyncio.Queue()
    closed = []

    async def send(value):
        nonlocal captures
        body = json.loads(value)
        sent.append(body)
        identity = f"response_{len(sent)}"
        if captures == terminal_at:
            output = terminal_output()
        else:
            captures += 1
            output = [block(call_id=f"capture_{captures}")]
        payload = envelope(output, model="gpt-6-astra")
        payload["id"] = identity
        events.put_nowait({"type": "response.created", "response": {"id": identity}})
        events.put_nowait({"type": "response.completed", "response": payload})

    async def recv():
        return json.dumps(await events.get())

    async def close():
        closed.append(True)

    driver.socket = SimpleNamespace(send=send, recv=recv, close=close)
    try:
        for target in (4, 12):
            terminal_at = target
            if target == 12:
                history = [Message.model_validate_json(m.model_dump_json()) for m in history]
                history.append(Message(role="user", content="Observe eight more synthetic frames"))
            while True:
                req = request(messages=recency_view(history, omission))
                before = req.model_dump_json()
                state = (driver.epoch, driver.previous_response_id, copy.deepcopy(driver.seen),
                    copy.deepcopy(driver.last_context), set(driver._native_call_ids),
                    dict(driver._native_call_types))
                planned = driver._budget_params(req, model="gpt-6-astra")["input"]
                await driver.request_budget(req, context_estimate=1, model="gpt-6-astra")
                assert state == (driver.epoch, driver.previous_response_id, driver.seen,
                    driver.last_context, driver._native_call_ids, driver._native_call_types)
                assert not hasattr(driver, "_staged_computer_images")
                reply = await driver.complete(req, model="gpt-6-astra")
                assert req.model_dump_json() == before
                # Native dispatch selects a delta after the unchanged full-plan
                # count. That full measurement is conservative, not a claim
                # that the counter remeasured the smaller socket delta.
                assert counts[-1][1]["input"] == planned
                assert counts[-2][1]["input"] == planned
                new = (reply.metadata or {}).get(IMAGES, {})
                assert new.keys().isdisjoint(published)
                assert len(new) == (1 if len(expected) > len(published) else 0)
                published.update(new)
                for item in sent[-1]["input"]:
                    if item.get("type") == "computer_call_output":
                        raw = base64.b64decode(item["output"]["image_url"].partition(",")[2])
                        assert hashlib.sha256(raw).hexdigest() == expected[item["call_id"]]
                if len(sent) > 1:
                    assert sent[-1]["previous_response_id"] == f"response_{len(sent) - 1}"
                    assert not any(i.get("type") == "computer_call" for i in sent[-1]["input"])
                    assert sum(i.get("type") == "computer_call_output" for i in sent[-1]["input"]) == len(new)
                history.append(stored_response(reply))
                if not reply.tool_calls:
                    assert captures == target and new
                    break
                call = reply.tool_calls[0]
                assert call.id == f"capture_{captures}" and call.name == "desk"
                png = image(captures)
                expected[call.id] = hashlib.sha256(base64.b64decode(png["source"]["data"])).hexdigest()
                history.append(Message(role="tool", tool_call_id=call.id,
                    content=[{"type": "text", "text": f"owned reference {captures}"}, png]))
        assert len(expected) == len(set(expected.values())) == len(published) == 12
        assert len(sent) == 14
        # Persisted terminal carrier + new user: no new observations to publish.
        req = request(messages=recency_view(history, omission) + [Message(role="user", content="Discuss these frames")])
        before = req.model_dump_json()
        reply = await driver.complete(req, model="gpt-6-astra")
        assert IMAGES not in (reply.metadata or {})
        assert req.model_dump_json() == before
        assert not any(i.get("type") in {"computer_call", "computer_call_output"} for i in sent[-1]["input"])
        # Corrupt an old, already-delivered pair that delta selection would omit.
        # Complete-history validation must reject it before any count or send.
        for corruption in ("digest", "reference", "missing_carrier", "orphan"):
            bad = ChatRequest.model_validate_json(req.model_dump_json())
            carrier = next(m for m in bad.messages if "capture_1" in (m.metadata or {}).get(IMAGES, {}))
            if corruption == "digest":
                carrier.metadata[IMAGES]["capture_1"]["sha256"] = "0" * 64
            elif corruption == "reference":
                result = next(m for m in bad.messages if m.tool_call_id == "capture_1")
                result.content = "wrong reference" + IMAGE_DROPPED
            elif corruption == "missing_carrier":
                carrier.metadata.pop(IMAGES)
            else:
                bad.messages = [m for m in bad.messages if "capture_1" not in (m.metadata or {}).get(CALLS, {})]
            frozen = bad.model_dump_json()
            dispatched = (len(counts), len(sent))
            with pytest.raises(InvalidRequestError):
                await driver.request_budget(bad, context_estimate=1, model="gpt-6-astra")
            with pytest.raises(InvalidRequestError):
                await driver.complete(bad, model="gpt-6-astra")
            assert (len(counts), len(sent)) == dispatched
            assert bad.model_dump_json() == frozen
        owner.start_native_job.assert_not_awaited()
    finally:
        await driver.close_live()
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    '{"success":false,"error":"safety halt"}',
    "/unavailable/capture.png",
    [{"type": "text", "text": "missing pixels"}],
])
async def test_native_adapter_later_failure_discussion_never_becomes_pixels(result):
    from amplifier_module_provider_openai.native import NativeResponsesProvider

    generations = 0

    def outputs(body):
        nonlocal generations
        generations += 1
        return [block()] if generations == 1 else terminal_output()

    driver, wires = sdk_driver(outputs)
    driver = NativeResponsesProvider.wrap(driver, owner_getter=lambda: None)
    try:
        first = await driver.complete(request())
        history = [stored_response(first), Message(role="tool", tool_call_id="call_fixture", content=result)]
        req = request(messages=history)
        before, sent_before = req.model_dump_json(), len(wires)
        with pytest.raises(InvalidRequestError):
            await driver.request_budget(req, context_estimate=1)
        with pytest.raises(InvalidRequestError):
            await driver.complete(req)
        assert len(wires) == sent_before and req.model_dump_json() == before
        req.messages.append(Message(role="user", content="Discuss the failure; do not retry"))
        before = req.model_dump_json()
        await driver.request_budget(req, context_estimate=1)
        reply = await driver.complete(req)
        assert req.model_dump_json() == before and IMAGES not in (reply.metadata or {})
        for _, body in wires[sent_before:]:
            assert not any(i.get("type") in {"computer_call", "computer_call_output"} for i in body["input"])
            assert "Historical computer-tool failure" in json.dumps(body["input"])
    finally:
        await driver.close()


def legacy_lineage_state(driver):
    """Live state, excluding the one explicitly permitted observation."""
    names = (
        "previous_response_id", "epoch", "seen", "pending_parent",
        "pending_steer", "inflight", "async_calls", "delivered_results",
        "native_calls", "request_uncertain", "_computer_function_lineage",
        "_native_call_ids", "_native_call_types", "_repaired_tool_ids",
        "_checkpoint", "_checkpoint_status", "_pending_additional_tools_item",
        "_tool_search_roster", "_tool_search_extra", "_neutral_computer_context",
    )
    return copy.deepcopy({name: getattr(driver, name, None) for name in names})


def legacy_lineage_fixture():
    from unittest.mock import AsyncMock

    from amplifier_module_provider_openai.native import NativeResponsesProvider

    history = [
        Message(role="user", content="Inspect"),
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "failed-computer", "name": "computer",
            "input": {"actions": [{"type": "screenshot"}]},
        }]),
        Message(role="tool", tool_call_id="failed-computer",
                content=json.dumps({"success": False, "error": {"code": "safety_halt"}})),
    ]
    sent = []

    async def get_messages():
        return [m.model_dump() for m in history]

    owner = SimpleNamespace(context=SimpleNamespace(get_messages=get_messages),
        runtime=SimpleNamespace(emit=AsyncMock()), native_job=lambda identity: None,
        config={}, start_native_job=AsyncMock())
    original, counts = sdk_driver(lambda body: terminal_output())

    class CountedLegacyFixture(NativeResponsesProvider):
        # Only the known official-route MockTransport opts in, not production.
        def _uses_standard_openai_endpoint(self):
            return str(self.client.base_url) == "https://api.openai.com/v1/"

    driver = CountedLegacyFixture.wrap(original, owner_getter=lambda: owner)
    events = asyncio.Queue()

    async def send(value):
        body = json.loads(value)
        sent.append(body)
        identity = f"legacy_response_{len(sent)}"
        payload = envelope(terminal_output(), model="gpt-6-astra")
        payload["id"] = identity
        events.put_nowait({"type": "response.created", "response": {"id": identity}})
        events.put_nowait({"type": "response.completed", "response": payload})

    async def recv():
        return json.dumps(await events.get())

    driver.socket = SimpleNamespace(send=send, recv=recv, close=AsyncMock())
    driver.previous_response_id = "prior"
    tool = ToolSpec(name="computer", description="Legacy computer",
                    type="computer", parameters={"type": "object"})
    req = ChatRequest(model="gpt-6-astra", tools=[tool], messages=history)
    return driver, owner, req, counts, sent


@pytest.mark.asyncio
@pytest.mark.parametrize("public_complete", [False, True])
async def test_legacy_lineage_failure_observation_is_complete_only_and_recovers_once(public_complete):
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    driver, owner, req, counts, sent = legacy_lineage_fixture()
    original = req.model_dump_json()
    state = legacy_lineage_state(driver)
    epoch = driver.epoch

    async def convert(packet):
        if public_complete:
            reply = await driver.complete(packet, model="gpt-6-astra")
            assert IMAGES not in (reply.metadata or {})
            return sent[-1]["input"]
        driver.owner = owner
        token = NATIVE_REQUEST.set(driver)
        try:
            return driver._convert_messages([m.model_dump() for m in packet.messages])
        finally:
            NATIVE_REQUEST.reset(token)
            driver.owner = None

    try:
        # Both budget entry points refuse, without observing the failed turn.
        with pytest.raises(InvalidRequestError):
            driver._budget_params(req, model="gpt-6-astra")
        with pytest.raises(InvalidRequestError):
            await driver.request_budget(req, context_estimate=1, model="gpt-6-astra")
        assert legacy_lineage_state(driver) == state and not driver.last_context
        assert not hasattr(driver, "_staged_computer_images")
        with pytest.raises(InvalidRequestError) as caught:
            await convert(req)
        assert caught.value.code == "computer_result_not_image"
        assert caught.value.result_kind == "error" and caught.value.retryable is False
        assert legacy_lineage_state(driver) == state
        observed = copy.deepcopy(driver.last_context)
        assert observed and "failed-computer" in json.dumps(list(observed))
        assert not hasattr(driver, "_staged_computer_images")
        assert req.model_dump_json() == original and counts == sent == []

        # A reminder cannot recover, and a failed count cannot update observation.
        reminder = req.model_copy(deep=True)
        reminder.messages.append(Message(role="user", content="Explain",
                                          metadata={"ephemeral": True}))
        with pytest.raises(InvalidRequestError):
            await driver.request_budget(reminder, context_estimate=1, model="gpt-6-astra")
        with pytest.raises(InvalidRequestError):
            await convert(reminder)
        assert driver.last_context == observed and legacy_lineage_state(driver) == state
        assert counts == sent == []
        owner.runtime.emit.assert_not_awaited()

        later = req.model_copy(deep=True)
        later.messages.append(Message(role="user", content="Keep the stop. Explain without tools."))
        frozen = later.model_dump_json()
        await driver.request_budget(later, context_estimate=1, model="gpt-6-astra")
        assert driver.last_context == observed and legacy_lineage_state(driver) == state
        assert not hasattr(driver, "_staged_computer_images")
        wire = await convert(later)
        assert driver.epoch != epoch
        assert driver.previous_response_id == ("legacy_response_1" if public_complete else None)
        if public_complete:
            assert sent[-1].get("previous_response_id") is None
        assert "safety_halt" in json.dumps(wire)
        assert not any(i.get("type") in {"computer_call", "computer_call_output"} for i in wire)
        new_epoch, new_context = driver.epoch, copy.deepcopy(driver.last_context)
        if not public_complete:
            driver.previous_response_id = "new-lineage"
        lineage = driver.previous_response_id
        await driver.request_budget(later, context_estimate=1, model="gpt-6-astra")
        assert driver.previous_response_id == lineage and driver.epoch == new_epoch
        again = await convert(later)
        assert driver.epoch == new_epoch and driver.last_context == new_context
        assert not any(i.get("type") in {"computer_call", "computer_call_output"} for i in again)
        if public_complete:
            assert sent[-1]["previous_response_id"] == lineage
        else:
            assert driver.previous_response_id == lineage
        assert later.model_dump_json() == frozen and req.model_dump_json() == original
        assert all(path.endswith("input_tokens") for path, _ in counts)
        assert all(not any(i.get("type") in {"computer_call", "computer_call_output"}
                           for i in body["input"]) for _, body in counts)
        owner.start_native_job.assert_not_awaited()
        if public_complete:
            assert [call.args[0] for call in owner.runtime.emit.await_args_list] == [
                "native.submitting", "native.request", "native.response.created", "native.completed",
            ] * 2
        else:
            owner.runtime.emit.assert_not_awaited()
    finally:
        await driver.close_live()


@pytest.mark.asyncio
@pytest.mark.parametrize("public_complete", [False, True])
async def test_legacy_lineage_pending_steering_refuses_recovery_without_state_commit(public_complete):
    from amplifier_core.llm_errors import LLMError

    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    driver, owner, req, counts, sent = legacy_lineage_fixture()
    req.messages[1:1] = [
        Message(role="assistant", content=[{
            "type": "tool_call", "id": "success-computer", "name": "computer",
            "input": {"actions": [{"type": "screenshot"}]},
        }]),
        Message(role="tool", tool_call_id="success-computer", content=[image(2)]),
    ]

    async def convert():
        if public_complete:
            return await driver.complete(req, model="gpt-6-astra")
        driver.owner = owner
        token = NATIVE_REQUEST.set(driver)
        try:
            return driver._convert_messages([m.model_dump() for m in req.messages])
        finally:
            NATIVE_REQUEST.reset(token)
            driver.owner = None

    try:
        with pytest.raises(InvalidRequestError):
            await convert()
        driver.pending_parent = True
        req.messages.append(Message(role="user", content="Keep the stop. Explain without tools."))
        frozen = req.model_dump_json()
        state, observed = legacy_lineage_state(driver), copy.deepcopy(driver.last_context)
        with pytest.raises(LLMError, match="Pending steering") as caught:
            await convert()
        assert caught.value.retryable is False
        assert legacy_lineage_state(driver) == state and driver.last_context == observed
        assert not hasattr(driver, "_staged_computer_images")
        assert req.model_dump_json() == frozen and counts == sent == []
        owner.runtime.emit.assert_not_awaited()
    finally:
        driver.pending_parent = False
        driver.owner = None
        await driver.close_live()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["png", "orphan", "carrier", "unknown_record"])
async def test_legacy_lineage_invalid_history_cannot_commit_observation(corruption):
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    driver, owner, req, counts, sent = legacy_lineage_fixture()
    if corruption == "png":
        req.messages[-1].content = [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "bm90IFBORw==",
        }}]
    elif corruption == "orphan":
        req.messages.append(Message(role="assistant", content="", metadata={CALLS: {"orphan": {}}}))
    elif corruption == "carrier":
        req.messages.append(Message(role="assistant", content="", metadata={IMAGES: {"orphan": {}}}))
    else:
        req.messages[1].metadata = {CALLS: {"failed-computer": {"unknown": True}}}
    frozen = req.model_dump_json()
    state = legacy_lineage_state(driver)
    driver.owner = owner
    token = NATIVE_REQUEST.set(driver)
    try:
        with pytest.raises(InvalidRequestError):
            driver._convert_messages([m.model_dump() for m in req.messages])
        assert not driver.last_context and legacy_lineage_state(driver) == state
        assert not hasattr(driver, "_staged_computer_images")
        assert req.model_dump_json() == frozen and counts == sent == []
    finally:
        NATIVE_REQUEST.reset(token)
        driver.owner = None
        await driver.close_live()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", [
    "missing_origin", "duplicate_origin", "duplicate_result", "result_before_call",
    "secondary_id", "conflicting_arguments", "missing_primary", "structured_missing_primary",
    "boolean_numeric_arguments", "structured_missing_name", "structured_missing_input",
])
async def test_legacy_lineage_retained_type_is_not_authority_for_invalid_pairing(corruption):
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    driver, owner, req, counts, sent = legacy_lineage_fixture()
    driver.owner = owner
    token = NATIVE_REQUEST.set(driver)
    try:
        # Establish a legacy type through genuine conversion, not a seeded map.
        valid = [m.model_dump() for m in req.messages]
        valid[-1]["content"] = [image(1)]
        driver._convert_messages(valid)
        state, observed = legacy_lineage_state(driver), copy.deepcopy(driver.last_context)
        messages = [m.model_dump() for m in req.messages]
        if corruption == "missing_origin":
            messages.pop(1)
        elif corruption == "duplicate_origin":
            messages[1]["content"].append(copy.deepcopy(messages[1]["content"][0]))
        elif corruption == "duplicate_result":
            messages.append(copy.deepcopy(messages[-1]))
        elif corruption == "result_before_call":
            messages[1], messages[2] = messages[2], messages[1]
        elif corruption == "secondary_id":
            messages[1]["content"][0]["call_id"] = "conflicting"
        elif corruption == "conflicting_arguments":
            messages[1]["content"][0]["arguments"] = {"actions": [{"type": "click", "x": 1, "y": 1}]}
        elif corruption == "missing_primary":
            messages[1]["content"] = []
            messages[1]["tool_calls"] = [{
                "call_id": "failed-computer", "tool": "computer",
                "arguments": {"actions": [{"type": "screenshot"}]},
            }]
        elif corruption == "structured_missing_primary":
            messages[1]["content"][0].pop("id")
            messages[1]["content"][0]["tool_call_id"] = "failed-computer"
        elif corruption == "structured_missing_name":
            call = messages[1]["content"][0]
            call["tool"] = call.pop("name")
        elif corruption == "structured_missing_input":
            call = messages[1]["content"][0]
            call["arguments"] = call.pop("input")
        else:
            call = messages[1]["content"][0]
            call["input"] = {"actions": [{"type": "screenshot", "x": 1}]}
            call["arguments"] = {"actions": [{"type": "screenshot", "x": True}]}
        frozen = copy.deepcopy(messages)
        with pytest.raises(InvalidRequestError):
            driver._convert_messages(messages)
        assert driver.last_context == observed and legacy_lineage_state(driver) == state
        assert messages == frozen and counts == sent == []
        assert driver._staged_computer_images == {}  # prior successful conversion only
    finally:
        NATIVE_REQUEST.reset(token)
        driver.owner = None
        await driver.close_live()


@pytest.mark.asyncio
@pytest.mark.parametrize("public_complete", [False, True])
@pytest.mark.parametrize("invalid_first", [False, True])
@pytest.mark.parametrize("neutral", [False, True])
async def test_legacy_lineage_error_cannot_mask_another_invalid_png(public_complete, invalid_first, neutral):
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    driver, owner, req, counts, sent = legacy_lineage_fixture()
    messages = [m.model_dump() for m in req.messages]
    if neutral:
        producer, _ = sdk_driver(lambda body: [block(call_id="bad-png")])
        try:
            genuine = await producer.complete(request())
            messages.insert(2, stored_response(genuine).model_dump())
        finally:
            await producer.close()
    else:
        messages[1]["content"].append({
            "type": "tool_call", "id": "bad-png", "name": "computer",
            "input": {"actions": [{"type": "screenshot"}]},
        })
    bad_result = {"role": "tool", "tool_call_id": "bad-png", "content": [{
        "type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "bm90IFBORw==",
        },
    }]}
    if invalid_first:
        failed = messages.pop(next(i for i, m in enumerate(messages) if m.get("tool_call_id") == "failed-computer"))
        messages.extend([bad_result, failed])
    else:
        messages.append(bad_result)
    req = req.model_copy(update={"messages": [Message.model_validate(m) for m in messages]})
    frozen = req.model_dump_json()
    state = legacy_lineage_state(driver)
    try:
        if public_complete:
            with pytest.raises(InvalidRequestError):
                await driver.complete(req, model="gpt-6-astra")
        else:
            driver.owner = owner
            token = NATIVE_REQUEST.set(driver)
            try:
                with pytest.raises(InvalidRequestError):
                    driver._convert_messages([m.model_dump() for m in req.messages])
            finally:
                NATIVE_REQUEST.reset(token)
                driver.owner = None
        assert legacy_lineage_state(driver) == state and not driver.last_context
        assert not hasattr(driver, "_staged_computer_images")
        assert req.model_dump_json() == frozen and counts == sent == []
        owner.runtime.emit.assert_not_awaited()
    finally:
        await driver.close_live()
