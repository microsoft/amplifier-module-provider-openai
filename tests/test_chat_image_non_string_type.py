"""A "type" that is not a string is data, never an image mark.

Request and response data are free-form JSON below the canonical blocks. A tool
parameter named "type" holds a dict there, and a JSON-schema type can be a list.
A call identity or name in such data is a key only when it is hashable.
Each case here raised ``TypeError: unhashable type`` on c2fe22df. Offline: the
real SDK on a mock transport, and the native protocol doubles.
"""

import copy
import json
import traceback

import httpx
import pytest
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import (
    ChatRequest,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolSpec,
)

from amplifier_module_provider_openai import (
    OpenAIProvider,
    _convert_function_tool_output_content,
)
from amplifier_module_provider_openai._chat_images import (
    OMITTED,
    diagnostic_copy,
    has_images,
    instruction_text,
    media_values,
    project_messages,
)
from tests.test_chat_image_loss import DATA, URL, body, image, provider
from tests.test_native_transport import finished, make

# The shapes that stopped every chat: grep, glob and apply_patch each have a
# parameter NAMED "type"; a nullable parameter has a LIST as its schema type.
NAMED_TYPE = {"type": "object", "properties": {
    "type": {"type": "string"}, "pattern": {"type": "string"},
}}
LIST_TYPE = {"type": "object", "properties": {"id": {"type": ["string", "null"]}}}
TUPLE_TYPE = {"type": "object", "properties": {"id": {"type": ("string", ["null"])}}}
SCHEMAS = {
    "parameter-named-type": NAMED_TYPE, "list-as-schema-type": LIST_TYPE,
    "tuple-as-schema-type": TUPLE_TYPE,
}
NOT_A_STRING = [{"nested": 1}, ["string", "null"]]
# json.loads never makes a tuple. A caller in the same process can pass one,
# and model_dump() keeps it in a free-form field. A tuple that holds a list
# passes isinstance(value, Hashable), and hash() of it raises.
NOT_A_KEY = [*NOT_A_STRING, ("string", ["null"])]


def wire_tool(parameters):
    return {"type": "function", "name": "search", "description": "Search.",
            "parameters": parameters, "strict": False}


def reply(stream, tools):
    """The API returns the request's tool list in every response object."""
    done = {**body(), "tools": tools}
    if not stream:
        return httpx.Response(200, json=done)
    created = {"type": "response.created", "response": {**done, "output": []}, "sequence_number": 0}
    completed = {"type": "response.completed", "response": done, "sequence_number": 1}
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=(
        f"event: response.created\ndata: {json.dumps(created)}\n\n"
        f"event: response.completed\ndata: {json.dumps(completed)}\n\n"
    ))


def raw(events, name):
    return [data["raw"] for event, data in events if event == name]


RAW_CALL = {"role": "assistant", "content": [
    {"type": "tool_call", "id": "one", "name": "inspect", "input": {}},
]}
SENT_USER = {"role": "user", "content": [{"type": "input_text", "text": "next"}]}


def serializer():
    return OpenAIProvider(api_key="fixture", config={"default_model": "gpt-6.1-sol", "max_retries": 0})


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("schema", list(SCHEMAS))
async def test_chat_returns_when_a_tool_schema_holds_a_type_that_is_not_a_string(stream, schema):
    parameters = SCHEMAS[schema]
    p, wire, events = provider(stream=stream, replies=[reply(stream, [wire_tool(parameters)])])
    request = ChatRequest(
        messages=[Message(role="user", content="Say only: ready")],
        tools=[ToolSpec(name="search", description="Search.", parameters=copy.deepcopy(parameters))],
    )
    before = request.model_dump()
    try:
        response = await p.complete(request)
        assert response.content[0].text == "OK"
        assert [path for path, _ in wire] == ["/v1/responses/input_tokens", "/v1/responses"]
        sent = wire[1][1]["tools"]
        assert [tool["name"] for tool in sent] == ["search"]
        on_wire = json.loads(json.dumps(parameters))
        assert sent[0]["parameters"]["properties"] == on_wire["properties"]
        # The diagnostic copies keep the schema. They are data, not media.
        assert raw(events, "llm:request")[0]["tools"] == sent
        assert raw(events, "llm:response")[0]["tools"][0]["parameters"] == on_wire
        assert request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_real_image_marks_are_still_sent_and_omitted_beside_such_a_schema(stream, failure, caplog):
    tools = [wire_tool(NAMED_TYPE), {**wire_tool(LIST_TYPE), "name": "lookup"}]
    p, wire, events = provider(
        stream=stream, failure=failure, replies=None if failure else [reply(stream, tools)],
    )
    request = ChatRequest(
        messages=[Message(role="user", content=[image(), image("base64")])],
        tools=[
            ToolSpec(name="search", parameters=copy.deepcopy(NAMED_TYPE)),
            ToolSpec(name="lookup", parameters=copy.deepcopy(LIST_TYPE)),
        ],
    )
    try:
        diagnostic = ""
        if failure:
            with pytest.raises(InvalidRequestError) as error:
                await p.complete(request)
            assert error.value.retryable is False and error.value.status_code == 400
            diagnostic = "".join(traceback.format_exception(error.value))
        else:
            await p.complete(request)
        assert wire[-1][1]["input"][0]["content"] == [
            {"type": "input_image", "image_url": URL},
            {"type": "input_image", "image_url": f"data:image/png;base64,{DATA}"},
        ]
        assert [tool["name"] for tool in wire[-1][1]["tools"]] == ["search", "lookup"]
        diagnostic += json.dumps(events) + caplog.text
        assert URL not in diagnostic and DATA not in diagnostic
        sent = raw(events, "llm:request")[0]
        assert sent["input"][0]["content"] == [{"type": "input_image", "media": OMITTED}] * 2
        assert sent["tools"] == wire[-1][1]["tools"]
    finally:
        await p.client.close()


CALL = Message(role="assistant", content=[ToolCallBlock(id="one", name="inspect", input={})])


def nested(output, **extra):
    return Message(role="tool", tool_call_id="one", content=[
        ToolResultBlock(tool_call_id="one", output=output, **extra),
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("value", NOT_A_STRING)
@pytest.mark.parametrize("place", [
    "call-arguments", "result-output", "result-envelope-content", "message-metadata",
    "tool-calls-view",
])
async def test_free_form_request_data_named_type_is_sent_as_before(place, value):
    messages = {
        "call-arguments": [
            Message(role="assistant", content=[
                ToolCallBlock(id="one", name="inspect", input={"type": value}),
            ]),
            Message(role="tool", tool_call_id="one", content="plain result"),
        ],
        "result-output": [CALL, nested({"type": value, "rows": []})],
        # has_images: a content list of this shape is not a media envelope.
        "result-envelope-content": [CALL, nested({"content": [{"type": value, "name": "id"}]})],
        "message-metadata": [Message(role="user", content="hello", metadata={"type": value})],
        # project_messages reads the Core tool_calls view beside the content.
        "tool-calls-view": [
            Message(role="assistant", content="", tool_calls=[
                {"id": "one", "tool": "inspect", "arguments": {}, "type": value},
            ]),
            Message(role="tool", tool_call_id="one", content="plain result"),
        ],
    }[place]
    p, wire, _ = provider()
    request = ChatRequest(messages=messages)
    before = request.model_dump()
    try:
        await p.complete(request)
        items = wire[-1][1]["input"]
        if place == "message-metadata":
            assert items == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
        else:
            assert [item["type"] for item in items] == ["function_call", "function_call_output"]
            assert items[0]["call_id"] == items[1]["call_id"] == "one"
        if place == "call-arguments":
            assert json.loads(items[0]["arguments"]) == {"type": value}
        if place in {"result-output", "result-envelope-content"}:
            # Legacy opaque ToolResult JSON remains unchanged, not media-classified.
            result = items[1]["output"]
            result = result[0]["text"] if isinstance(result, list) else result
            assert json.loads(result)[0]["output"] == before["messages"][1]["content"][0]["output"]
        assert request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", NOT_A_STRING)
@pytest.mark.parametrize("place", ["envelope-field", "status-field"])
async def test_image_result_with_such_data_beside_it_still_sends_the_image(place, value):
    blocks = [TextBlock(text="before").model_dump(), image("base64").model_dump()]
    message = (
        nested({"schema": {"type": value}, "content": blocks}) if place == "envelope-field"
        else nested(blocks, error={"type": value})
    )
    p, wire, events = provider()
    request = ChatRequest(messages=[CALL, message])
    before = request.model_dump()
    try:
        await p.complete(request)
        rich = wire[-1][1]["input"][1]["output"]
        key = "schema" if place == "envelope-field" else "error"
        assert json.loads(rich[0]["text"]) == {key: {"type": value}}
        assert rich[1]["text"] == "before"
        assert rich[2] == {"type": "input_image", "image_url": f"data:image/png;base64,{DATA}"}
        assert DATA not in json.dumps(events) and request.model_dump() == before
    finally:
        await p.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", NOT_A_STRING)
async def test_image_result_list_with_such_a_block_gets_the_existing_refusal(value):
    # The same answer as for {"type": "other"}: only text and image blocks qualify.
    p, wire, events = provider()
    request = ChatRequest(messages=[CALL, nested([image("base64").model_dump(), {"type": value}])])
    try:
        with pytest.raises(InvalidRequestError, match="Image-bearing tool result") as error:
            await p.complete(request)
        assert error.value.retryable is False
        assert wire == []
        assert DATA not in "".join(traceback.format_exception(error.value)) + json.dumps(events)
    finally:
        await p.client.close()


@pytest.mark.asyncio
async def test_native_request_with_such_a_schema_is_sent_and_its_trace_omits_media():
    p, _ = make()
    traces, contexts = [], []
    p.trace = lambda name, data: traces.append((name, copy.deepcopy(data)))
    p.trace_context = lambda data: contexts.append(copy.deepcopy(data))
    tools = [wire_tool(NAMED_TYPE), {**wire_tool(LIST_TYPE), "name": "lookup"}]
    params = {"model": "gpt-6-astra", "tools": tools, "input": [{
        "role": "user", "content": [{"type": "input_image", "image_url": URL}],
    }]}
    before = copy.deepcopy(params)
    response = finished("schema-fixture")
    response["response"]["tools"] = copy.deepcopy(tools)
    p.socket.feed(
        {"type": "response.created", "response": {"id": "schema-fixture", "tools": copy.deepcopy(tools)}},
        response,
    )
    await p._native_response(params)
    assert p.socket.sent[0]["tools"] == tools and p.socket.sent[0]["input"] == before["input"]
    assert params == before
    assert URL not in json.dumps([traces, contexts])
    assert contexts[0]["tools"] == tools
    assert [data["response"]["tools"] for name, data in traces if name == "native.receive"] == [tools] * 2


@pytest.mark.parametrize("value", NOT_A_STRING)
def test_media_values_alone_finds_only_real_marks(value):
    data_url = f"data:image/png;base64,{DATA}"
    assert media_values({"type": value}) == set()
    assert media_values({"tools": [wire_tool(NAMED_TYPE), wire_tool(LIST_TYPE)]}) == set()
    assert media_values({
        "tools": [wire_tool(NAMED_TYPE), wire_tool(LIST_TYPE)],
        "metadata": {"type": value},
        "input": [
            {"type": value, "image_url": "not-media", "source": {"url": "not-media"}},
            image().model_dump(), image("base64").model_dump(),
            {"type": "input_image", "image_url": data_url},
            {"type": "computer_screenshot", "image_url": URL + "#screen"},
        ],
    }) == {URL, DATA, data_url, URL + "#screen"}


@pytest.mark.parametrize("value", NOT_A_STRING)
def test_diagnostic_copy_alone_omits_only_real_marks(value):
    plain = {
        "tools": [wire_tool(NAMED_TYPE), wire_tool(LIST_TYPE)],
        "metadata": {"type": value, "image_url": "kept"},
    }
    assert diagnostic_copy(plain, {URL}) == plain
    marked = {**plain, "input": [
        image().model_dump(),
        {"type": "input_image", "image_url": URL},
        {"type": "computer_screenshot", "image_url": URL},
        {"type": value, "text": f"echo {URL}"},
    ]}
    assert diagnostic_copy(marked, {URL}) == {**plain, "input": [
        {"type": "image", "media": OMITTED},
        {"type": "input_image", "media": OMITTED},
        {"type": "computer_screenshot", "media": OMITTED},
        {"type": value, "text": f"echo {OMITTED}"},
    ]}


@pytest.mark.parametrize("value", NOT_A_STRING)
def test_has_images_alone_reads_a_type_that_is_not_a_string_as_no_image(value):
    picture = image("base64").model_dump()
    assert has_images({"content": [{"type": value}]}) is False
    assert has_images({"content": [picture, {"type": value}]}) is False
    assert has_images([{"type": value}]) is False
    assert has_images({"type": value}) is False
    assert has_images({"content": [picture]}) is True
    assert has_images([{"type": value}, picture]) is True


@pytest.mark.parametrize("value", NOT_A_STRING)
@pytest.mark.parametrize("view", ["tool-calls-view", "output-items", "apply-patch-arguments"])
def test_project_messages_alone_keeps_calls_whose_type_is_not_a_string(view, value):
    assistant = {
        "tool-calls-view": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "one", "tool": "inspect", "arguments": {}, "type": value},
        ]},
        "output-items": {"role": "assistant", "content": [], "metadata": {
            "openai:output_items": [{"type": value, "call_id": "one"}],
        }},
        "apply-patch-arguments": {"role": "assistant", "content": [
            {"type": "tool_call", "id": "one", "name": "apply_patch", "input": {"type": value}},
        ]},
    }[view]
    messages = [assistant, {"role": "tool", "tool_call_id": "one", "content": "plain result"}]
    before = copy.deepcopy(messages)
    assert project_messages(messages, qualified=True, model="gpt-6.1-sol", strict_direct=True) == before
    assert messages == before


def saved_items(*items):
    return [
        {"role": "assistant", "content": [{"type": "text", "text": "saved"}],
         "metadata": {"openai:output_items": list(items)}},
        {"role": "user", "content": "next"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("item, refusal, text", [
    ({"type": "function_call", "id": {"nested": 1}, "name": "inspect", "arguments": "{}"},
     InvalidRequestError, "Canonical response call has no unique following real result"),
    ({"type": "function_call", "id": ("one", ["two"]), "name": "inspect", "arguments": "{}"},
     InvalidRequestError, "Canonical response call has no unique following real result"),
    ({"id": ["one"], "name": "inspect"}, LLMError, "Response output contains a non-item value"),
])
async def test_saved_call_whose_identity_is_not_hashable_gets_the_serializers_own_refusal(
    item, refusal, text,
):
    # The answer of the commit before c2fe22df, not a TypeError from the image projection.
    p, wire, _ = provider()
    request = ChatRequest(messages=[Message.model_validate(m) for m in saved_items(item)])
    try:
        with pytest.raises(refusal, match=text):
            await p.complete(request)
        assert wire == []
    finally:
        await p.client.close()


@pytest.mark.parametrize("value", NOT_A_KEY)
@pytest.mark.parametrize("shape", [
    "native-call-identity", "function-call-identity", "tool-calls-view-identity", "call-name",
])
def test_project_messages_alone_skips_an_identity_or_name_that_is_not_hashable(shape, value):
    messages = {
        "native-call-identity": saved_items({"type": "computer_call", "call_id": value}),
        "function-call-identity": saved_items({"type": "function_call", "call_id": value}),
        "tool-calls-view-identity": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": value, "tool": "inspect", "arguments": {}},
            ]},
            {"role": "user", "content": "next"},
        ],
        "call-name": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "one", "tool": "inspect", "name": value, "arguments": {}},
            ]},
            {"role": "tool", "tool_call_id": "one", "content": "plain result"},
        ],
    }[shape]
    before = copy.deepcopy(messages)
    # _convert_messages passes its ordinary names as a frozenset.
    assert project_messages(
        messages, qualified=True, model="gpt-6.1-sol", ordinary_names=frozenset({"computer"}),
    ) == before
    assert messages == before


@pytest.mark.parametrize("value", NOT_A_KEY)
def test_serializer_keeps_a_call_whose_name_is_not_hashable(value):
    p = serializer()
    items = p._convert_messages([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "one", "tool": "inspect", "name": value, "arguments": {}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "plain result"},
    ])
    assert [item["type"] for item in items] == ["function_call", "function_call_output"]
    assert items[0]["call_id"] == items[1]["call_id"] == "one"


# Core refuses the shapes of the next three cases. The native path gives the
# serializer saved history as plain dicts that Core did not read, so they are
# given that way here.
@pytest.mark.parametrize("value", NOT_A_KEY)
@pytest.mark.parametrize("content", ["plain", "image", "nested-image-result"])
def test_serializer_skips_a_saved_message_whose_role_is_not_a_string(content, value):
    picture = image("base64").model_dump()
    message = {"role": value, "content": {
        "plain": "ignored", "image": [picture],
        "nested-image-result": [{"type": "tool_result", "tool_call_id": "one", "output": [picture]}],
    }[content]}
    items = serializer()._convert_messages([RAW_CALL, message, {"role": "user", "content": "next"}])
    assert [item.get("type") for item in items] == ["function_call", "function_call_output", None]
    assert items[1]["output"].startswith("[error] Tool execution result missing")
    assert items[2] == SENT_USER
    assert DATA not in json.dumps(items)


@pytest.mark.parametrize("value", NOT_A_KEY)
def test_serializer_keeps_a_saved_text_result_whose_identity_is_not_hashable(value):
    items = serializer()._convert_messages([RAW_CALL, {"role": "user", "content": [
        {"type": "tool_result", "tool_call_id": value, "output": "plain result"},
    ]}])
    assert [item["type"] for item in items] == ["function_call", "function_call_output"]
    assert items[0]["call_id"] == items[1]["call_id"] == "one"


@pytest.mark.parametrize("value", NOT_A_KEY)
@pytest.mark.parametrize("role", ["user", "tool"])
def test_saved_image_result_whose_identity_is_not_hashable_gets_the_existing_refusal(role, value):
    message = {"role": role, "content": [
        {"type": "tool_result", "tool_call_id": value, "output": [image("base64").model_dump()]},
    ]}
    if role == "tool":
        message["tool_call_id"] = "one"
    with pytest.raises(InvalidRequestError, match="Image-bearing tool result") as error:
        serializer()._convert_messages([RAW_CALL, message])
    assert DATA not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("value", NOT_A_STRING)
def test_instruction_text_alone_skips_a_block_whose_type_is_not_a_string(value):
    assert instruction_text([
        {"type": value, "text": "not an instruction block"},
        {"type": "text", "text": "kept"},
    ]) == "kept"
    p = serializer()
    items = p._convert_messages([{"role": "developer", "content": [
        {"type": value}, {"type": "text", "text": "kept"},
    ]}])
    assert items[0]["content"][0]["text"] == "<context_file>\nkept\n</context_file>"


@pytest.mark.parametrize("value", NOT_A_STRING)
def test_tool_content_whose_text_is_not_a_string_stays_a_legacy_list(value):
    content = [{"type": "text", "text": value}]
    assert _convert_function_tool_output_content(content, "gpt-6.1-sol") is None
    p = serializer()
    items = p._convert_messages([
        {"role": "assistant", "content": [
            {"type": "tool_call", "id": "one", "name": "inspect", "input": {}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": content},
    ])
    assert [item["type"] for item in items] == ["function_call", "function_call_output"]
    assert json.loads(items[1]["output"]) == content
