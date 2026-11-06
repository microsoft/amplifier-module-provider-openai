"""Offline boundary checks, including actual SDK 3.29 HTTP serialization."""

import copy
import json

import httpx
import openai
import pytest
from amplifier_core import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai.configuration_updates import (
    KEY,
    RESTATE_KEY,
    compacted_update_message,
    configuration_update_from_response,
    configuration_update_message,
    message_item,
    reject_standalone,
    request_configuration,
    validate_item,
    validate_request,
)

MODEL = "gpt-6-astra"


def update(effort="high", model=MODEL):
    return Message(**configuration_update_message(model, effort, request_effort="low"))


def request_with_update(model=MODEL):
    return ChatRequest(
        model=model,
        reasoning_effort="low",
        messages=[
            Message(role="user", content="Original"),
            Message(role="assistant", content="Answer"),
            update(model=model),
            Message(role="user", content="Analyze"),
        ],
    )


def provider(config=None, client=None):
    return OpenAIProvider(
        api_key="offline",
        client=client,
        config={"default_model": MODEL, "reasoning_effort": "low", **(config or {})},
    )


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"type": "configuration_update"},
        {"type": "configuration_update", "reasoning": {"effort": None}},
        {"type": "configuration_update", "reasoning": {"effort": 1}},
        {"type": "configuration_update", "reasoning": {"effort": []}},
        {"type": "configuration_update", "reasoning": {"effort": "automatic"}},
        {"type": "configuration_update", "reasoning": {"effort": "high", "mode": "pro"}},
        {"type": "configuration_update", "reasoning": {"effort": "high"}, "api_key": "secret"},
        {"type": "configuration_update", "text": {"format": {"type": "json_object"}}},
        {"type": "configuration_update", "reasoning": {"effort": "high"}, "id": "custom"},
    ],
)
def test_minimal_schema_rejects_arbitrary_settings(item):
    with pytest.raises(ValueError, match="only an explicit"):
        validate_item(item)


def test_item_reader_owns_json_without_policy_dependency():
    item = {"type": "configuration_update", "reasoning": {"effort": "high"}}
    message = {
        "role": "user", "content": "",
        "metadata": {KEY: {"version": 1, "model": MODEL, "request_effort": "low", "item": item}},
    }
    wire = message_item(message)
    assert wire == item
    wire["reasoning"]["effort"] = "low"
    assert item["reasoning"]["effort"] == "high"
    message["content"] = "Do not lose this instruction"
    with pytest.raises(ValueError, match="carrier"):
        message_item(message)


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-invented", "gpt-6-astra-2099-01-01"])
def test_exact_model_eligibility(model):
    with pytest.raises(ValueError, match="exact supported"):
        configuration_update_message(model, "high", request_effort="low")


@pytest.mark.parametrize("model,effort", [(MODEL, "none"), ("gpt-6.1-sol", "none"), ("gpt-6.1-sol", "minimal")])
def test_model_specific_effort_restrictions(model, effort):
    with pytest.raises(ValueError, match="not supported"):
        configuration_update_message(model, effort, request_effort="low")


def test_restart_low_high_low_preserves_positions_and_original_request_prefix():
    p = provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Original")], reasoning_effort="low"
    )
    first = p._budget_params(request)
    request.messages.extend([
        Message(role="assistant", content="First answer"),
        update("high"), Message(role="user", content="Hard follow-up"),
    ])
    high = p._budget_params(request)
    request.messages.extend([
        Message(role="assistant", content="Second answer"),
        update("low"), Message(role="user", content="Routine follow-up"),
    ])
    before = request.model_dump()
    # Actual Core serialization/reload and a new provider, no process-local map.
    restarted = ChatRequest.model_validate_json(request.model_dump_json())
    low = provider()._budget_params(restarted)
    assert high["input"][:len(first["input"])] == first["input"]
    assert low["input"][:len(high["input"])] == high["input"]
    assert high["input"][2] == validate_item({"type": "configuration_update", "reasoning": {"effort": "high"}})
    assert low["input"][5] == {"type": "configuration_update", "reasoning": {"effort": "low"}}
    assert first["reasoning"] == high["reasoning"] == low["reasoning"]
    assert first["reasoning"]["effort"] == "low"
    assert p.reasoning_effort == "low"
    assert request.model_dump() == before


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"reasoning": {"effort": "low", "mode": "pro"}}, "standard"),
        ({"reasoning": {"effort": "low", "agent_count": 2}}, "extensions"),
        ({"truncation": "auto"}, "truncation"),
        ({"context_management": [{"type": "compaction", "compact_threshold": 1000}]}, "automatic compaction"),
        ({"model": "gpt-5.6-sol"}, "different model"),
        ({"input": [{"role": "user", "content": "replacement"}]}, "cannot replace"),
        ({"reasoning": {"effort": "none"}}, "original request-level effort"),
        ({"reasoning": {"effort": "high"}}, "original request-level effort"),
        ({"multi_agent": {"enabled": True, "max_concurrent_subagents": 3}}, "single-agent"),
        ({"multi_agent": None}, "single-agent"),
        ({"multi_agent": {"enabled": 0}}, "single-agent"),
    ],
)
def test_final_extra_body_overrides_cannot_bypass_history_validation(extra, match):
    p = provider({"extra_request_params": {"extra_body": extra}})
    with pytest.raises(ValueError, match=match):
        p._budget_params(request_with_update())


def test_raw_update_schema_gate_and_standalone_refusal():
    params = {"model": MODEL, "input": [{"type": "configuration_update", "reasoning": {"effort": "high"}}]}
    with pytest.raises(ValueError, match="Standalone"):
        reject_standalone(params)
    params["input"][0]["reasoning"]["secret"] = "not a setting"
    with pytest.raises(ValueError, match="only an explicit"):
        validate_request(params)


def test_final_model_override_cannot_retarget_recorded_history():
    with pytest.raises(ValueError, match="different model"):
        provider()._budget_params(request_with_update(), model="gpt-6.1-sol")


@pytest.mark.parametrize("effort", [None, "medium"])
def test_original_omission_and_explicit_medium_are_different_baselines(effort):
    p = provider({"reasoning_effort": effort})
    original = ChatRequest(messages=[Message(role="user", content="Original")])
    params = p._budget_params(original)
    baseline = request_configuration(params)
    assert baseline["request_effort"] == effort
    carrier = configuration_update_from_response({"metadata": {"openai:request_configuration": baseline}}, "high")
    original.messages.extend([Message(role="assistant", content="Answer"), Message(**carrier), Message(role="user", content="Next")])
    reloaded = ChatRequest.model_validate_json(original.model_dump_json())
    assert p._budget_params(reloaded).get("reasoning") == params.get("reasoning")
    different = "medium" if effort is None else None
    with pytest.raises(ValueError, match="original request-level effort"):
        provider({"reasoning_effort": different})._budget_params(reloaded)


def test_config_none_sentinel_is_omitted_not_literal_request_none():
    from amplifier_core.llm_errors import InvalidRequestError

    p = provider({"reasoning_effort": "none"})
    params = p._budget_params(ChatRequest(messages=[Message(role="user", content="Original")]))
    assert request_configuration(params)["request_effort"] is None
    with pytest.raises(ValueError, match="not supported"):
        configuration_update_message(MODEL, "none", request_effort=None)
    request = request_with_update()
    request.reasoning_effort = "none"
    with pytest.raises(InvalidRequestError, match="does not support reasoning.effort"):
        p._budget_params(request)


def test_update_from_response_does_not_infer_baseline_from_api_echo():
    with pytest.raises(ValueError, match="No recorded"):
        configuration_update_from_response({"reasoning": {"effort": "medium"}}, "high")


def test_post_compaction_restates_latest_effort_without_touching_window():
    window = [{"type": "compaction", "id": "cmp1", "encrypted_content": "opaque"}]
    updates = [
        {"type": "configuration_update", "reasoning": {"effort": "high"}},
        {"type": "configuration_update", "reasoning": {"effort": "low"}},
    ]
    message = compacted_update_message(MODEL, window, updates, request_effort="low")
    before = copy.deepcopy(message)
    wire = provider()._budget_params(ChatRequest(messages=[
        Message(**message), Message(role="user", content="Next"),
    ]))["input"]
    assert wire[:1] == window
    assert wire[1] == updates[-1]
    assert message["metadata"]["openai:compaction"]["output"] == window
    assert message == before
    assert message["metadata"][RESTATE_KEY]["item"]["reasoning"]["effort"] == "low"


def test_canonical_metadata_replays_opaque_items_without_duplicates():
    output = [
        {"type": "reasoning", "id": "rs1", "encrypted_content": "opaque", "summary": []},
        {"type": "message", "id": "m1", "role": "assistant", "phase": "final_answer",
         "status": "completed", "content": [{"type": "output_text", "text": "answer", "annotations": []}]},
        {"type": "compaction", "id": "cmp", "encrypted_content": "also-opaque"},
    ]
    messages = [
        Message(role="user", content="Original"),
        Message(role="assistant", content="Legacy duplicate", metadata={"openai:output_items": output}),
        update(), Message(role="user", content="Next"),
    ]
    before = [m.model_dump() for m in messages]
    wire = provider({"reasoning_replay_scope": "all"})._budget_params(ChatRequest(messages=messages))["input"]
    assert wire[1:4] == output
    assert len(wire) == 6
    assert [m.model_dump() for m in messages] == before


@pytest.mark.parametrize("scope,old_count,active_count", [
    ("none", 0, 0), ("turn", 0, 1), ("all", 1, 1),
])
def test_canonical_reasoning_obeys_request_scope_without_mutating_saved_output(
    scope, old_count, active_count
):
    from amplifier_module_provider_openai import _RawResponseObject

    p = provider()
    reasoning = {"type": "reasoning", "id": "rs1", "summary": [],
                 "encrypted_content": "opaque"}
    text = {"type": "message", "id": "msg1", "role": "assistant",
            "phase": "commentary", "status": "completed",
            "content": [{"type": "output_text", "text": "Answer", "annotations": []}]}
    response = p._convert_to_chat_response(_RawResponseObject({
        "id": "resp1", "status": "completed", "model": MODEL,
        "output": [reasoning, text],
    }))
    saved = Message(role="assistant", content=response.content, metadata=response.metadata)
    saved = Message.model_validate_json(saved.model_dump_json())
    old = [
        Message(role="user", content="Original"), saved,
        Message(role="user", content="New turn"),
    ]
    active = [
        Message(role="user", content="Original"), saved,
        Message(role="user", content="Ephemeral", metadata={"ephemeral": True}),
    ]
    before = saved.model_dump()
    for messages, count in [(old, old_count), (active, active_count)]:
        wire = p._convert_messages(
            [message.model_dump() for message in messages],
            reasoning_replay_scope=scope,
        )
        assert len([item for item in wire if item.get("type") == "reasoning"]) == count
        assert next(item for item in wire if item.get("id") == "msg1") == text
    assert saved.model_dump() == before
    assert saved.metadata["openai:output_items"] == [reasoning, text]


@pytest.mark.parametrize("results", [[], [
    Message(role="tool", tool_call_id="call1", content="first"),
    Message(role="tool", tool_call_id="call1", content="duplicate"),
]])
def test_canonical_host_call_requires_unique_real_result(results):
    from amplifier_core.llm_errors import InvalidRequestError

    output = [{"type": "function_call", "call_id": "call1", "id": "fc1",
               "name": "read", "arguments": "{}", "status": "completed"}]
    request = ChatRequest(messages=[
        Message(role="user", content="Original"),
        Message(role="assistant", content="", metadata={"openai:output_items": output}),
        *results,
    ])
    before = request.model_dump()
    with pytest.raises(InvalidRequestError, match="unique following real result"):
        provider()._budget_params(request)
    assert request.model_dump() == before


def test_canonical_async_flag_or_metadata_cannot_authorize_missing_result():
    from amplifier_core.llm_errors import InvalidRequestError

    item = {"type": "function_call", "call_id": "async1", "id": "fc1",
            "name": "read", "arguments": "{}", "status": "completed", "async": True}
    message = Message(role="assistant", content="", metadata={
        "openai:output_items": [item], "native_pending_async": True,
    })
    with pytest.raises(InvalidRequestError, match="unique following real result"):
        provider()._convert_messages([message.model_dump()])


@pytest.mark.parametrize("invalid_results", ["duplicate", "before", "wrong_type"])
def test_pending_async_hook_never_bypasses_duplicate_or_preceding_results(invalid_results):
    from amplifier_core.llm_errors import InvalidRequestError

    class PendingProvider(OpenAIProvider):
        def _canonical_pending_async_call(self, item):
            raise AssertionError("Private owner hook must not be consulted for present results")

    call = Message(role="assistant", content="", metadata={"openai:output_items": [{
        "type": "function_call", "call_id": "async1", "id": "fc1",
        "name": "read", "arguments": "{}", "status": "completed", "async": True,
    }]})
    result = Message(role="tool", tool_call_id="async1", content="real result")
    if invalid_results == "wrong_type":
        result = Message(role="assistant", content="", metadata={"openai:output_items": [{
            "type": "computer_call_output", "call_id": "async1",
            "output": {"type": "computer_screenshot", "image_url": "unused"},
        }]})
        messages = [call, result]
    else:
        messages = [result, call] if invalid_results == "before" else [call, result, result]
    with pytest.raises(InvalidRequestError, match="unique following real result"):
        PendingProvider(api_key="offline")._convert_messages([m.model_dump() for m in messages])


def test_canonical_host_call_with_real_result_replays_once():
    output = [{"type": "function_call", "call_id": "call1", "id": "fc1",
               "name": "read", "arguments": "{}", "status": "completed"}]
    messages = [
        Message(role="assistant", content="", metadata={"openai:output_items": output}),
        Message(role="tool", tool_call_id="call1", content="real result"),
    ]
    wire = provider()._convert_messages([m.model_dump() for m in messages])
    assert wire == [*output, {"type": "function_call_output", "call_id": "call1", "output": "real result"}]


def test_canonical_computer_metadata_cannot_resurrect_failed_projected_call():
    from amplifier_core.llm_errors import InvalidRequestError

    messages = [
        {"role": "user", "content": "Inspect"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking"},
                {"type": "tool_call", "id": "call1", "name": "computer",
                 "input": {"actions": [{"type": "screenshot"}]}},
            ],
            "metadata": {"openai:output_items": [
                {"type": "computer_call", "call_id": "call1", "id": "cc1",
                 "actions": [{"type": "screenshot"}], "status": "completed"},
            ]},
        },
        {"role": "tool", "tool_call_id": "call1",
         "content": '{"success":false,"error":{"code":"safety_halt"}}'},
        {"role": "user", "content": "Keep halted. Explain."},
    ]
    before = copy.deepcopy(messages)
    with pytest.raises(InvalidRequestError, match="no failed computer call was resurrected"):
        provider()._convert_messages(messages)
    assert messages == before


def test_message_level_repair_cannot_fabricate_canonical_call_result():
    from amplifier_core.llm_errors import InvalidRequestError

    output = [{"type": "function_call", "call_id": "call1", "id": "fc1",
               "name": "read", "arguments": "{}", "status": "completed"}]
    request = ChatRequest(messages=[Message(
        role="assistant",
        content=[{"type": "tool_call", "id": "call1", "name": "read", "input": {}}],
        metadata={"openai:output_items": output},
    )])
    before = request.model_dump()
    with pytest.raises(InvalidRequestError, match="no result was synthesized"):
        provider()._plan_tool_repairs(request)
    assert request.model_dump() == before


def test_message_repair_respects_only_exact_private_pending_call_qualification():
    item = {"type": "function_call", "call_id": "async1", "id": "fc1",
            "name": "read", "arguments": "{}", "status": "completed", "async": True}

    class PendingProvider(OpenAIProvider):
        def _canonical_pending_async_call(self, candidate):
            return candidate == item

    request = ChatRequest(messages=[Message(
        role="assistant",
        content=[{"type": "tool_call", "id": "async1", "name": "read", "input": {}}],
        metadata={"openai:output_items": [item]},
    )])
    original = request.model_dump()
    planned, event = PendingProvider(api_key="offline")._plan_tool_repairs(request)
    assert planned.model_dump() == original
    assert event is None
    assert request.model_dump() == original


def test_final_input_override_cannot_move_update_with_identical_sequence():
    p = provider()
    request = request_with_update()
    wire = p._budget_params(request)["input"]
    moved = [wire[2], *wire[:2], *wire[3:]]
    p.extra_request_params = {"extra_body": {"input": moved}}
    with pytest.raises(ValueError, match="reposition"):
        p._budget_params(request)


def response_body(output=None):
    return {
        "id": "resp_offline", "object": "response", "created_at": 1,
        "status": "completed", "model": MODEL,
        "output": output if output is not None else [{
            "type": "message", "id": "msg_offline", "role": "assistant",
            "status": "completed", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "OK", "annotations": []}],
        }],
        "reasoning": {"effort": "low", "mode": "standard"},
        "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
        "parallel_tool_calls": True, "tools": [],
    }


@pytest.mark.asyncio
async def test_real_sdk_unknown_input_item_survives_http_and_provider_complete():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 30})
        return httpx.Response(200, json=response_body())

    async with openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        p = provider({"use_streaming": False}, client=client)
        result = await p.complete(request_with_update(), auto_continue=False)
        assert result.content[0].text == "OK"
        carrier = configuration_update_from_response(result, "low")
        assert carrier["metadata"][KEY]["request_effort"] == "low"
        body = next(body for path, body in calls if path == "/v1/responses")
        assert body["reasoning"]["effort"] == "low"
        assert body["input"][2] == {"type": "configuration_update", "reasoning": {"effort": "high"}}
        assert len([1 for path, _ in calls if path == "/v1/responses"]) == 1


@pytest.mark.asyncio
async def test_real_sdk_low_high_low_across_responses_and_core_restart():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 30})
        calls.append(body)
        response = response_body()
        response["id"] = f"resp_{len(calls)}"
        response["output"][0]["id"] = f"msg_{len(calls)}"
        return httpx.Response(200, json=response)

    async with openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        p = provider({"use_streaming": False}, client=client)
        request = ChatRequest(messages=[Message(role="user", content="Original")])
        first = await p.complete(request, auto_continue=False)
        request.messages.extend([
            Message(role="assistant", content=first.content, metadata=first.metadata),
            Message(**configuration_update_from_response(first, "high")),
            Message(role="user", content="Hard follow-up"),
        ])
        second = await p.complete(request, auto_continue=False)
        request.messages.extend([
            Message(role="assistant", content=second.content, metadata=second.metadata),
            Message(**configuration_update_from_response(second, "low")),
            Message(role="user", content="Routine follow-up"),
        ])
        original = request.model_dump()
        restarted = ChatRequest.model_validate_json(request.model_dump_json())
        fresh_provider = provider({"use_streaming": False}, client=client)
        await fresh_provider.complete(restarted, auto_continue=False)
        assert request.model_dump() == original
    assert len(calls) == 3
    assert calls[1]["input"][:len(calls[0]["input"])] == calls[0]["input"]
    assert calls[2]["input"][:len(calls[1]["input"])] == calls[1]["input"]
    assert [body["reasoning"]["effort"] for body in calls] == ["low"] * 3
    assert [
        item["reasoning"]["effort"]
        for item in calls[2]["input"] if item.get("type") == "configuration_update"
    ] == ["high", "low"]


@pytest.mark.asyncio
async def test_real_sdk_streaming_records_actual_baseline_not_response_echo():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 30})
        response = response_body()
        # Deliberately distinct echo: it must not be used to capture wire effort.
        response["reasoning"]["effort"] = "medium"
        created = {**copy.deepcopy(response), "status": "in_progress", "output": []}
        events = [
            {"type": "response.created", "response": created, "sequence_number": 0},
            {"type": "response.completed", "response": response, "sequence_number": 1},
        ]
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text="".join(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                for event in events
            ),
        )

    async with openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        result = await provider(client=client).complete(request_with_update(), auto_continue=False)
        assert result.content[0].text == "OK"
        assert configuration_update_from_response(result, "low")["metadata"][KEY]["request_effort"] == "low"
        body = next(body for path, body in calls if path == "/v1/responses")
        assert body["stream"] is True
        assert body["input"][2]["type"] == "configuration_update"


@pytest.mark.asyncio
async def test_explicit_compaction_uses_responses_not_standalone_then_restates():
    calls = []
    window = [{"type": "compaction", "id": "cmp1", "encrypted_content": "opaque"}]

    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 30})
        assert request.url.path == "/v1/responses"
        assert body["input"][-1] == {"type": "compaction_trigger"}
        return httpx.Response(200, json=response_body(window))

    async with openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        p = provider(client=client)
        request = request_with_update()
        before = request.model_dump()
        compacted = await p.compact_context(request)
        assert request.model_dump() == before
        body = next(body for path, body in calls if path == "/v1/responses")
        assert body["reasoning"]["effort"] == "low"
        assert body["store"] is False
        wire = p._budget_params(ChatRequest(messages=[
            Message(**compacted["message"]), Message(role="user", content="Next"),
        ]))["input"]
        assert wire == [*window, {"type": "configuration_update", "reasoning": {"effort": "high"}}, {
            "role": "user", "content": [{"type": "input_text", "text": "Next"}],
        }]
        assert not any(path.endswith("/compact") for path, _ in calls)


@pytest.mark.asyncio
async def test_invalid_compaction_combo_rejects_before_any_http():
    calls = []

    def handler(request):
        calls.append(request)
        raise AssertionError("No count, compact, or generation API is permitted")

    async with openai.AsyncOpenAI(
        api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        p = provider({"truncation": "auto"}, client)
        with pytest.raises(ValueError, match="truncation"):
            await p.compact_context(request_with_update())
    assert calls == []


@pytest.mark.asyncio
async def test_explicit_compaction_unknown_submission_has_no_sdk_retry_or_fallback():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 30})
        raise httpx.ReadTimeout("Submission outcome unknown", request=request)

    async with openai.AsyncOpenAI(
        api_key="offline", max_retries=2,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        p = provider(client=client)
        with pytest.raises(openai.APITimeoutError):
            await p.compact_context(request_with_update())
    assert calls.count("/v1/responses") == 1
    assert not any(path.endswith("/compact") for path in calls)