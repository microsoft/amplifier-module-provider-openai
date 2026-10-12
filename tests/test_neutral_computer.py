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
    KEY,
    neutral_screenshot,
    replay_calls,
    response_calls,
    translate,
    validate_intent,
)

PNG = Path(__file__).parent / "fixtures/computer_use/valid-red-pixel.png"


def image():
    import base64
    import struct
    import zlib

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    # A backend fake's full screenshot, matching the real declared pixel space.
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1280, 720, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\x00" + b"\x00" * (1280 * 3)) * 720)) + chunk(b"IEND", b"")
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
        for tc in response.tool_calls
    ]
    if storage == "structured":
        calls = []
    elif storage in {"legacy", "name_arguments"}:
        content = []
        if storage == "name_arguments":
            calls = [
                {"id": tc.id, "name": tc.name, "arguments": json.dumps(tc.arguments)}
                for tc in response.tool_calls
            ]
    elif storage == "both_fields":
        for call in calls:
            call["name"] = call["tool"]
            call["input"] = call["arguments"]
            call["arguments"] = json.dumps(call["arguments"])
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
