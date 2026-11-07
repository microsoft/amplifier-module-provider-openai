"""Response replay metadata must not override a caller's supplied role."""

import copy

import pytest
from amplifier_core.llm_errors import InvalidRequestError
from amplifier_module_provider_openai import OpenAIProvider


def saved_output():
    return [
        {
            "type": "reasoning",
            "id": "rs1",
            "encrypted_content": "opaque",
            "summary": [],
        },
        {
            "type": "message",
            "id": "msg1",
            "role": "assistant",
            "phase": "final_answer",
            "content": [
                {
                    "type": "output_text",
                    "text": "Old assistant answer",
                    "annotations": [],
                }
            ],
        },
    ]


@pytest.mark.parametrize("role", ["user", "developer", "system"])
def test_nonassistant_content_is_not_replaced_by_carried_response_metadata(role):
    # Consumers may derive a role/content view while retaining saved metadata.
    # A retained output item cannot turn their supplied input into an assistant
    # answer, reasoning, or executable call.
    output = saved_output() + [
        {
            "type": "function_call",
            "id": "fc1",
            "call_id": "call1",
            "name": "old_tool",
            "arguments": "{}",
            "status": "completed",
        }
    ]
    messages = [
        {
            "role": role,
            "content": "Current supplied content",
            "metadata": {"openai:output_items": output},
        }
    ]
    before = copy.deepcopy(messages)
    wire = OpenAIProvider(api_key="offline")._convert_messages(messages)
    expected = "Current supplied content"
    if role == "developer":
        expected = f"<context_file>\n{expected}\n</context_file>"
    assert wire == (
        []
        if role == "system"
        else [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": expected}],
            }
        ]
    )
    assert messages == before


def test_tool_result_uses_its_real_content_despite_carried_response_metadata():
    call = {
        "type": "function_call",
        "id": "fc1",
        "call_id": "call1",
        "name": "read",
        "arguments": "{}",
        "status": "completed",
    }
    messages = [
        {
            "role": "assistant",
            "content": "",
            "metadata": {"openai:output_items": [call]},
        },
        {
            "role": "tool",
            "tool_call_id": "call1",
            "content": "Actual result",
            "metadata": {"openai:output_items": saved_output()},
        },
    ]
    before = copy.deepcopy(messages)
    wire = OpenAIProvider(api_key="offline")._convert_messages(messages)
    assert wire == [
        call,
        {"type": "function_call_output", "call_id": "call1", "output": "Actual result"},
    ]
    assert messages == before


def test_assistant_still_replays_original_items_and_preserves_saved_metadata():
    output = saved_output()
    messages = [
        {
            "role": "assistant",
            "content": "Legacy text view",
            "metadata": {"openai:output_items": output},
        }
    ]
    before = copy.deepcopy(messages)
    wire = OpenAIProvider(api_key="offline")._convert_messages(
        messages, reasoning_replay_scope="all"
    )
    assert wire == output
    assert messages == before


@pytest.mark.parametrize("output", [None, [42]])
def test_assistant_malformed_output_is_still_rejected(output):
    with pytest.raises(
        InvalidRequestError, match="malformed canonical computer output"
    ):
        OpenAIProvider(api_key="offline")._convert_messages(
            [
                {
                    "role": "assistant",
                    "content": "Not a fallback",
                    "metadata": {"openai:output_items": output},
                },
            ]
        )
