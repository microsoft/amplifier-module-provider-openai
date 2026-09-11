"""Tests for rich ordinary function-call output content."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from amplifier_core.message_models import (
    ChatRequest,
    ImageBlock,
    Message,
    TextBlock,
    ToolCallBlock,
)

from amplifier_module_provider_openai import OpenAIProvider


class DummyResponse:
    """Minimal completed response accepted by the provider response converter."""

    def __init__(self) -> None:
        self.output = []
        self.usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        self.stop_reason = "stop"


class StreamContext:
    """Async context manager for the internal streaming transport."""

    def __init__(self, response: DummyResponse) -> None:
        self.response = response

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> None:
        raise StopAsyncIteration

    async def get_final_response(self) -> DummyResponse:
        return self.response


def _provider(**config: object) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="test-key",
        config={"default_model": "gpt-5.6-terra", "max_retries": 0, **config},
    )


def _image_data() -> str:
    return base64.b64encode(b"test-image").decode("ascii")


def _tool_message(content: object) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": "call_result",
        "tool_name": "inspect",
        "content": content,
    }


def _function_output(messages: list[dict[str, object]]) -> dict[str, object]:
    return next(
        item for item in messages if item.get("type") == "function_call_output"
    )


def _request_with_image_result() -> ChatRequest:
    return ChatRequest(
        messages=[
            Message(
                role="assistant",
                content=[
                    ToolCallBlock(id="call_result", name="inspect", input={})
                ],
            ),
            Message(
                role="tool",
                tool_call_id="call_result",
                content=[
                    TextBlock(text="before"),
                    ImageBlock(
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _image_data(),
                        }
                    ),
                    TextBlock(text="after"),
                ],
            ),
        ]
    )


def test_converter_preserves_rich_function_output_order_and_strips_internal_fields() -> None:
    provider = _provider()
    content = [
        {"type": "text", "text": "before", "visibility": "internal"},
        {
            "type": "image",
            "visibility": "internal",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _image_data(),
                "visibility": "internal",
            },
        },
        {"type": "text", "text": "after"},
    ]
    messages = provider._convert_messages([_tool_message(content)])

    assert _function_output(messages)["output"] == [
        {"type": "input_text", "text": "before"},
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_image_data()}",
        },
        {"type": "input_text", "text": "after"},
    ]
    assert content[1]["visibility"] == "internal"
    assert content[1]["source"]["visibility"] == "internal"


def test_converter_omits_images_for_known_nonvision_model_without_losing_text() -> None:
    provider = _provider(default_model="o3")
    messages = provider._convert_messages(
        [
            _tool_message(
                [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _image_data(),
                        },
                    },
                    {"type": "text", "text": "after"},
                ]
            )
        ]
    )

    assert _function_output(messages)["output"] == [
        {"type": "input_text", "text": "before"},
        {
            "type": "input_text",
            "text": "[Image omitted: selected model does not support vision.]",
        },
        {"type": "input_text", "text": "after"},
    ]


def test_malformed_declared_image_raises_without_leaking_data() -> None:
    provider = _provider()
    image_data = "not-base64-image-data"

    with pytest.raises(ValueError, match="invalid base64") as exc_info:
        provider._convert_messages(
            [
                _tool_message(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        }
                    ]
                )
            ]
        )

    assert image_data not in str(exc_info.value)

    invalid_source_data = "secret-image-data"
    with pytest.raises(ValueError, match="base64 image source") as exc_info:
        provider._convert_messages(
            [
                _tool_message(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "media_type": "image/png",
                                "data": invalid_source_data,
                            },
                        }
                    ]
                )
            ]
        )

    assert invalid_source_data not in str(exc_info.value)


def test_invalid_rich_content_block_type_raises_type_error() -> None:
    provider = _provider()

    with pytest.raises(TypeError, match="only text or image blocks"):
        provider._convert_messages(
            [
                _tool_message(
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _image_data(),
                            },
                        },
                        {"type": "file"},
                    ]
                )
            ]
        )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain text", "plain text"),
        ({"answer": 42}, '{"answer": 42}'),
        (
            [{"type": "text", "text": "text-only"}],
            '[{"type": "text", "text": "text-only"}]',
        ),
    ],
)
def test_legacy_function_outputs_are_unchanged(content: object, expected: str) -> None:
    provider = _provider()
    messages = provider._convert_messages([_tool_message(content)])

    assert _function_output(messages)["output"] == expected


def test_blocking_complete_uses_effective_model_for_rich_tool_output() -> None:
    provider = _provider(use_streaming=False)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_request_with_image_result(), model="o3"))

    params = provider.client.responses.create.await_args.kwargs
    assert params["model"] == "o3"
    output = _function_output(params["input"])["output"]
    assert output[0] == {"type": "input_text", "text": "before"}
    assert output[1]["text"].startswith("[Image omitted:")
    assert output[2] == {"type": "input_text", "text": "after"}


def test_streaming_complete_uses_effective_model_for_rich_tool_output() -> None:
    provider = _provider(use_streaming=True)
    provider.client.responses.stream = MagicMock(
        return_value=StreamContext(DummyResponse())
    )

    asyncio.run(provider.complete(_request_with_image_result(), model="gpt-5.6-terra"))

    params = provider.client.responses.stream.call_args.kwargs
    assert params["model"] == "gpt-5.6-terra"
    output = _function_output(params["input"])["output"]
    assert output[0] == {"type": "input_text", "text": "before"}
    assert output[1] == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{_image_data()}",
    }
    assert output[2] == {"type": "input_text", "text": "after"}