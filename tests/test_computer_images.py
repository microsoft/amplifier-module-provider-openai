"""Preserve multi-image inputs and completed computer history without replay."""

import copy
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core import llm_errors
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.compaction import compacted_message
from amplifier_module_provider_openai.computer_images import prepare_computer_images


def tool():
    return ToolSpec(
        name="computer",
        description="Mounted computer executor",
        type="computer",
        parameters={
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        },
    )


def image(value="upload"):
    return {
        "type": "input_image",
        "image_url": "data:image/png;base64," + value,
        "detail": "original",
    }


def packet(history=True):
    items = []
    if history:
        for index in range(2):
            items += [
                {
                    "type": "computer_call",
                    "call_id": f"c{index}",
                    "actions": [{"type": "screenshot"}],
                    "status": "completed",
                },
                {
                    "type": "computer_call_output",
                    "call_id": f"c{index}",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": f"data:image/png;base64,screen{index}",
                        "detail": "original",
                    },
                },
            ]
    items += [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Compare both attachments"},
                image("first"),
                image("second"),
            ],
        }
    ]
    return {
        "model": "gpt-5.6-terra",
        "input": items,
        "tools": [{"type": "computer"}],
        "tool_choice": "auto",
    }


@pytest.mark.parametrize("history", [False, True])
def test_all_attachments_and_completed_history_survive(history):
    original = packet(history)
    before = copy.deepcopy(original)
    result = prepare_computer_images(original, [tool()])
    assert original == before
    assert result["input"][-1] == original["input"][-1]
    assert result["tools"][0]["parameters"] == tool().parameters
    assert result["tools"][0]["strict"] is False
    assert result["tools"][0]["type"] == "function"
    for index in range(2 if history else 0):
        call, output = result["input"][index * 2 : index * 2 + 2]
        assert call["call_id"] == output["call_id"] == f"c{index}"
        assert json.loads(call["arguments"]) == {"actions": [{"type": "screenshot"}]}
        assert output["output"] == [image(f"screen{index}")]
    assert prepare_computer_images(result, [tool()]) is result


def test_images_across_turns_and_tool_results_count_but_json_text_does_not():
    original = packet(False)
    first, second = original["input"][0]["content"][-2:]
    original["input"] = [
        {"role": "user", "content": [first]},
        {"type": "function_call_output", "call_id": "f1", "output": [second]},
    ]
    assert prepare_computer_images(original, [tool()]) is not original
    original["input"][1]["output"] = json.dumps([second])
    assert prepare_computer_images(original, [tool()]) is original


def test_native_single_image_and_screenshot_only_requests_stay_native():
    original = packet()
    original["input"][-1]["content"].pop()
    assert prepare_computer_images(original, [tool()]) is original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["input"].pop(1),
        lambda p: p["input"][0].update(pending_safety_checks=[{"id": "must-confirm"}]),
        lambda p: p["input"][0].update(status="incomplete"),
        lambda p: p["input"][0].update(id="opaque-response-item"),
        lambda p: p.update(tool_choice={"type": "computer"}),
        lambda p: p.update(previous_response_id="opaque-server-history"),
        lambda p: p["input"].insert(
            0, {"type": "compaction", "encrypted_content": "opaque"}
        ),
    ],
)
def test_unsafe_projection_fails_without_changing_history(mutation):
    original = packet()
    mutation(original)
    before = copy.deepcopy(original)
    with pytest.raises(
        llm_errors.InvalidRequestError, match="saved history are preserved"
    ) as error:
        prepare_computer_images(original, [tool()])
    assert error.value.retryable is False
    assert original == before


def test_bare_native_declaration_does_not_invent_function_schema():
    with pytest.raises(
        llm_errors.InvalidRequestError, match="no mounted computer function schema"
    ):
        prepare_computer_images(packet(False), [{"type": "computer"}])


def provider(**config):
    coordinator = MagicMock()
    coordinator.get_capability.return_value = None
    coordinator.hooks.emit = AsyncMock()
    return OpenAIProvider(api_key="test", config=config, coordinator=coordinator)


def request():
    return ChatRequest(
        model="gpt-5.6-terra",
        messages=[
            Message(
                role="user",
                content=[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "first",
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "second",
                        },
                    },
                ],
            )
        ],
        tools=[tool()],
        max_output_tokens=64,
    )


def test_budget_and_dispatch_assembly_are_identical_and_pure():
    instance = provider()
    req = request()
    before = req.model_dump()
    planned, state, _ = instance._assemble_initial_responses_params(req)
    assert planned["tools"][0]["type"] == "function"
    assert instance._native_call_types == {}
    instance._commit_initial_assembly_state(state)
    live, _, _ = instance._assemble_initial_responses_params(req)
    assert live == planned
    assert req.model_dump() == before


def test_final_extras_with_an_additional_image_use_the_same_projection():
    wire = packet(False)
    instance = provider(extra_request_params={"input": wire["input"]})
    req = request()
    req.messages = [Message(role="user", content="No images until host extras")]
    result, _, _ = instance._assemble_initial_responses_params(req)
    assert result["input"] == wire["input"]
    assert result["tools"][0]["type"] == "function"


def test_native_compact_window_is_not_rewritten():
    instance = provider()
    req = request()
    canonical = packet()["input"][:2] + [
        {"type": "compaction", "encrypted_content": "opaque"}
    ]
    req.messages.insert(0, Message(**compacted_message(req.model, canonical)))
    before = req.model_dump()
    with pytest.raises(llm_errors.InvalidRequestError, match="native compacted window"):
        instance._assemble_initial_responses_params(req)
    assert req.model_dump() == before


def native_history_request():
    req = request()
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    req.messages[:0] = [
        Message(role="user", content="Earlier computer interaction"),
        Message(
            role="assistant",
            content=[
                {
                    "type": "tool_call",
                    "id": "old-computer",
                    "name": "computer",
                    "input": {"actions": [{"type": "screenshot"}]},
                }
            ],
        ),
        Message(
            role="tool",
            name="computer",
            tool_call_id="old-computer",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": png,
                    },
                }
            ],
        ),
    ]
    return req


def test_real_message_assembly_retains_history_and_next_function_result():
    instance = provider()
    req = native_history_request()
    before = req.model_dump()
    first, state, _ = instance._assemble_initial_responses_params(req)
    assert req.model_dump() == before
    assert all(
        item.get("type") not in {"computer_call", "computer_call_output"}
        for item in first["input"]
    )
    instance._commit_initial_assembly_state(state)
    req.messages += [
        Message(
            role="assistant",
            content=[
                {
                    "type": "tool_call",
                    "id": "next-function",
                    "name": "computer",
                    "input": {"action": "screenshot"},
                }
            ],
        ),
        Message(
            role="tool",
            name="computer",
            tool_call_id="next-function",
            content=req.messages[2].content,
        ),
    ]
    second, _, _ = instance._assemble_initial_responses_params(req)
    outputs = [
        item for item in second["input"] if item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in outputs] == ["old-computer", "next-function"]
    assert outputs[-1]["output"][0]["type"] == "input_image"


@pytest.mark.asyncio
async def test_native_wire_switches_lineage_once_and_preserves_canonical_history():
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    from .test_native_transport import make

    instance, owner = make()
    req = native_history_request()
    req.model = "gpt-6-astra"
    messages = [m.model_dump() for m in req.messages]
    before = copy.deepcopy(messages)
    instance._request_messages = messages
    instance.full_messages = messages
    owner.messages = messages
    instance.previous_response_id = "earlier-native-response"
    instance._native_response = AsyncMock(return_value={})
    instance._guard_assembled_params_with_provider_count = AsyncMock()
    token = NATIVE_REQUEST.set(instance)
    try:
        params, state, _ = instance._assemble_initial_responses_params(req)
        instance._commit_initial_assembly_state(state)
        await instance._create_response(params)
        sent = instance._native_response.call_args.args[0]
        assert instance.previous_response_id is None
        assert sent["input"] == params["input"]
        assert instance._computer_function_lineage is True
        first_epoch = instance.epoch
        instance.previous_response_id = "function-response"
        instance._native_request_attempted = False
        await instance._create_response(params)
        assert instance.previous_response_id == "function-response"
        assert instance.epoch == first_epoch
        assert instance._native_response.call_args.args[0]["input"] == []
        assert messages == before
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
async def test_pending_native_steering_is_not_moved_to_new_computer_transport():
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    from .test_native_transport import make

    instance, owner = make()
    req = native_history_request()
    req.model = "gpt-6-astra"
    instance._request_messages = [m.model_dump() for m in req.messages]
    instance.full_messages = instance._request_messages
    owner.messages = instance.full_messages
    instance.previous_response_id = "native-parent"
    instance.pending_parent = True
    instance._native_response = AsyncMock()
    token = NATIVE_REQUEST.set(instance)
    try:
        params, state, _ = instance._assemble_initial_responses_params(req)
        instance._commit_initial_assembly_state(state)
        with pytest.raises(llm_errors.LLMError, match="Pending steering"):
            await instance._create_response(params)
        instance._native_response.assert_not_called()
        assert instance.previous_response_id == "native-parent"
        assert instance.pending_parent is True
    finally:
        NATIVE_REQUEST.reset(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("singular_action", [False, True])
async def test_native_function_batch_result_continues_without_resending_call(
    singular_action,
):
    from amplifier_module_provider_openai.native import NATIVE_REQUEST

    from .test_native_transport import make

    instance, owner = make()
    req = native_history_request()
    req.model = "gpt-6-astra"
    instance._native_response = AsyncMock(return_value={})
    instance._guard_assembled_params_with_provider_count = AsyncMock()
    token = NATIVE_REQUEST.set(instance)
    try:

        async def send():
            messages = [m.model_dump() for m in req.messages]
            instance._request_messages = messages
            instance.full_messages = messages
            owner.messages = messages
            before = copy.deepcopy(messages)
            params, state, _ = instance._assemble_initial_responses_params(req)
            instance._commit_initial_assembly_state(state)
            instance._native_request_attempted = False
            await instance._create_response(params)
            assert messages == before

        await send()
        instance.previous_response_id = "function-response"
        epoch = instance.epoch
        args = {"actions": [{"type": "screenshot"}]}
        if singular_action:
            args["action"] = "screenshot"
        req.messages += [
            Message(
                role="assistant",
                content=[
                    {
                        "type": "tool_call",
                        "id": "next-batch",
                        "name": "computer",
                        "input": args,
                    }
                ],
                metadata={"converge_live_epoch": epoch},
            ),
            Message(
                role="tool",
                name="computer",
                tool_call_id="next-batch",
                content=req.messages[2].content,
            ),
        ]
        await send()
        sent = instance._native_response.call_args.args[0]
        assert instance._native_response.await_count == 2
        assert instance.previous_response_id == "function-response"
        assert instance.epoch == epoch
        assert len(sent["input"]) == 1
        output = sent["input"][0]
        assert output["type"] == "function_call_output"
        assert output["call_id"] == "next-batch"
        assert output["output"][0]["type"] == "input_image"
        assert all(item.get("type") != "computer" for item in sent["tools"])
    finally:
        NATIVE_REQUEST.reset(token)


def test_only_explicit_validated_function_lineage_can_accept_result_only_delta():
    original = packet()
    original["input"] = original["input"][1:2]
    original["tools"] = [
        {"type": "function", "name": "computer", "parameters": tool().parameters}
    ]
    for ids in (frozenset(), frozenset({"different-call"})):
        with pytest.raises(llm_errors.InvalidRequestError, match="pair is incomplete"):
            prepare_computer_images(
                original, [], function_lineage=True, retained_function_call_ids=ids
            )
    projected = prepare_computer_images(
        original,
        [],
        function_lineage=True,
        retained_function_call_ids=frozenset({"c0"}),
    )
    assert projected["input"][0]["type"] == "function_call_output"
    assert original["input"][0]["type"] == "computer_call_output"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-6-astra"])
async def test_sdk_dispatch_and_budget_see_the_same_preserved_images(model):
    from .test_computer_use_integration import TestWireToolChoice

    instance = provider(use_streaming=False, max_retries=0)
    req = native_history_request()
    req.model = model
    planned, _, _ = instance._assemble_initial_responses_params(req)
    sent = await TestWireToolChoice._capture(instance, req)
    assert sent["input"] == planned["input"]
    assert sent["tools"] == planned["tools"]
    assert sent["tools"][0]["type"] == "function"
