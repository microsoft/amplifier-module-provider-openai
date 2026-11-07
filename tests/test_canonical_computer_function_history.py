"""A saved function call stays ordinary even when the current roster changes."""

import copy
import json

import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_core.message_models import ChatRequest

from amplifier_module_provider_openai import OpenAIProvider


def history(result, *, later_user=True):
    arguments = {
        "version": 1,
        "actions": [{"type": "screenshot"}],
        "capture_after": True,
    }
    call = {
        "type": "function_call",
        "id": "fc_saved",
        "call_id": "call_saved",
        "name": "computer",
        "arguments": json.dumps(arguments),
        "status": "completed",
    }
    messages = [
        {"role": "user", "content": "Inspect the fixture"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking the fixture"},
                {
                    "type": "tool_call",
                    "id": "call_saved",
                    "name": "computer",
                    "input": arguments,
                },
            ],
            "metadata": {"openai:output_items": [call]},
        },
        {"role": "tool", "tool_call_id": "call_saved", "content": result},
    ]
    if later_user:
        messages.append(
            {"role": "user", "content": "Explain the saved result without using tools"}
        )
    return messages, call


@pytest.mark.parametrize("later_user", [False, True])
@pytest.mark.parametrize("stale_native_tracking", [False, True])
@pytest.mark.parametrize(
    "result",
    [
        json.dumps(
            {
                "success": True,
                "output": {"images": ["/saved/shot.png"], "text": "captured"},
            }
        ),
        json.dumps({"success": False, "error": {"code": "controller_expired"}}),
    ],
)
def test_canonical_computer_function_keeps_real_result_without_current_roster(
    later_user,
    stale_native_tracking,
    result,
):
    provider = OpenAIProvider(api_key="offline-test", config={})
    if stale_native_tracking:
        provider._native_call_ids.add("call_saved")
        provider._native_call_types["call_saved"] = "computer"
    messages, call = history(result, later_user=later_user)
    original = copy.deepcopy(messages)
    # No current computer tool: expired/removed configuration must not retype history.
    request = ChatRequest(messages=messages, model="gpt-6.1-sol", tools=[])
    params, _, _ = provider._assemble_initial_responses_params(request)
    wire = params["input"]
    assert call in wire
    assert [x for x in wire if x.get("call_id") == "call_saved"] == [
        call,
        {
            "type": "function_call_output",
            "call_id": "call_saved",
            "output": [
                {
                    "type": "input_text",
                    "text": result,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
            ],
        },
    ]
    assert "Historical computer-tool failure" not in json.dumps(wire)
    assert messages == original
    assert (
        request.model_dump()
        == ChatRequest(messages=original, model="gpt-6.1-sol", tools=[]).model_dump()
    )


def test_canonical_function_still_requires_its_actual_result():
    provider = OpenAIProvider(api_key="offline-test", config={})
    messages, _ = history("saved")
    del messages[2]
    with pytest.raises(InvalidRequestError, match="missing its real result"):
        provider._assemble_initial_responses_params(ChatRequest(messages=messages))
