import base64
import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.llm_errors import InvalidRequestError

from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_openai._computer_result import screenshot_data_url

PNG = base64.b64encode(
    (Path(__file__).parent / "fixtures/computer_use/valid-red-pixel.png").read_bytes()
).decode()


@pytest.mark.parametrize(
    "content",
    [
        "Computer halted: user activity detected",
        json.dumps(
            {
                "success": False,
                "error": {"code": "safety_halt", "message": "Resume explicitly"},
            }
        ),
        {"success": False, "output": {"halted": True}},
        [{"type": "text", "text": "Screenshot permission denied"}],
        base64.b64encode(b'{"error":"safety_halt"}').decode(),
        "not base64!!!",
        base64.b64encode(b"\x89PNG\r\n\x1a\ntruncated").decode(),
        [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": PNG},
            }
        ],
        [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.invalid/screenshot"},
            }
        ],
        "data:image/png;base64,AAAA",
    ],
)
def test_nonimages_fail_locally_without_mutating_or_replaying_original(content):
    coordinator = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    coordinator.get_capability.return_value = None
    provider = OpenAIProvider(api_key="test-key", config={}, coordinator=coordinator)
    provider._native_call_ids = {"computer-call"}
    provider._native_call_types = {"computer-call": "computer"}
    messages = [
        {
            "role": "tool",
            "tool_call_id": "computer-call",
            "content": content,
            "tool_name": "computer",
        }
    ]
    original = copy.deepcopy(messages)
    with pytest.raises(InvalidRequestError) as caught:
        provider._convert_messages(messages)
    assert caught.value.retryable is False
    assert caught.value.code == "computer_result_not_image"
    assert caught.value.tool_call_id == "computer-call"
    assert messages == original
    assert "image was invented" in str(caught.value)
    assert "Resume explicitly" not in str(
        caught.value
    )  # No arbitrary tool content in diagnostics.


@pytest.mark.parametrize(
    "content",
    [
        PNG,
        "data:image/png;base64," + PNG,
        [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": PNG},
            }
        ],
    ],
)
def test_valid_png_preserves_exact_encoded_bytes(content):
    assert screenshot_data_url(content) == "data:image/png;base64," + PNG


def test_gif_mime_is_preserved_and_not_assumed_png():
    gif = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    assert (
        screenshot_data_url(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/gif",
                        "data": gif,
                    },
                }
            ]
        )
        == "data:image/gif;base64," + gif
    )


def test_corrupt_crc_oversized_and_multiple_images_rejected():
    raw = bytearray(base64.b64decode(PNG))
    raw[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        screenshot_data_url(base64.b64encode(raw).decode())
    with pytest.raises(ValueError, match="limit"):
        screenshot_data_url("A" * (28 * 1024 * 1024))
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }
    with pytest.raises(ValueError):
        screenshot_data_url([block, block])


def test_safety_text_alongside_an_image_is_not_silently_dropped():
    content = [
        {"type": "text", "text": "Safety halt: explicit resume required"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG},
        },
    ]
    with pytest.raises(ValueError, match="accompanying text"):
        screenshot_data_url(content)


def test_later_user_message_projects_failure_without_replay_or_canonical_changes():
    provider = OpenAIProvider(api_key="test-key", config={})
    messages = [
        {"role": "user", "content": "Inspect the fixture"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking"},
                {
                    "type": "tool_call",
                    "id": "call_42",
                    "name": "computer",
                    "input": {"actions": [{"type": "screenshot"}]},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_42",
            "content": json.dumps(
                {
                    "success": False,
                    "error": {
                        "code": "safety_halt",
                        "message": "Explicit resume required",
                    },
                }
            ),
        },
    ]
    original = copy.deepcopy(messages)
    with pytest.raises(InvalidRequestError):
        provider._convert_messages(messages)
    with pytest.raises(InvalidRequestError):
        provider._convert_messages(
            [
                *messages,
                {
                    "role": "user",
                    "content": "Injected reminder",
                    "metadata": {"ephemeral": True},
                },
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": "Leave the computer halted. Explain the failure without tools.",
        }
    )
    wire = provider._convert_messages(messages)
    assert not any(
        item.get("type") in {"computer_call", "computer_call_output", "function_call"}
        for item in wire
    )
    encoded = json.dumps(wire)
    assert (
        "safety_halt" in encoded
        and "Explicit resume required" in encoded
        and "call_42" in encoded
    )
    assert "halt remains in effect" in encoded and "Do not replay" in encoded
    assert messages[:-1] == original
    assert (
        provider._native_call_types["call_42"] == "computer"
    )  # No halt/native ownership reset.


def test_projection_keeps_valid_image_and_unrelated_tool_identity_and_bounds_reference():
    from amplifier_module_provider_openai.computer_history import (
        project_failed_computer_history,
    )

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "bad", "name": "computer", "arguments": {"actions": []}},
                {"id": "good", "name": "computer", "arguments": {"actions": []}},
                {"id": "other", "name": "read_file", "arguments": {}},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "bad",
            "content": "Safety halt " + "x" * 20000,
        },
        {"role": "tool", "tool_call_id": "good", "content": PNG},
        {"role": "tool", "tool_call_id": "other", "content": "kept"},
        {"role": "user", "content": "Explain"},
    ]
    original = copy.deepcopy(messages)
    view = project_failed_computer_history(messages)
    assert [call["id"] for call in view[0]["tool_calls"]] == ["good", "other"]
    assert view[2:] == messages[2:] and messages == original
    assert len(view[1]["content"]) < 14000 and '"truncated": true' in view[1]["content"]
    assert project_failed_computer_history(view) is view  # Stable once projected.
