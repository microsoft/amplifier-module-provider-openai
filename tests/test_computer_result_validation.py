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
