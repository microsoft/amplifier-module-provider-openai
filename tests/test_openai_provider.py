import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.content_models import ThinkingContent
from amplifier_core.message_models import ReasoningBlock
from amplifier_module_provider_openai import OpenAIChatResponse
from amplifier_module_provider_openai import OpenAIProvider


class DummyResponse:
    """Minimal response stub for provider tests."""

    def __init__(self, output=None):
        self.output = output or []
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self.stop_reason = "stop"


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def test_extended_thinking_enables_reasoning_and_budget_adjustment():
    provider = OpenAIProvider(api_key="test-key", config={"max_tokens": 1024})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [{"role": "user", "content": "Hello"}]

    asyncio.run(provider.complete(messages, extended_thinking=True, thinking_budget_tokens=6000))

    provider.client.responses.create.assert_awaited()
    call_kwargs = provider.client.responses.create.await_args_list[0].kwargs

    assert call_kwargs["reasoning"]["effort"] == "high"
    # Default buffer is 1024 tokens, so expect budget + buffer to override defaults
    assert call_kwargs["max_output_tokens"] == 7024


def test_convert_to_chat_response_produces_reasoning_block():
    class FakeReasoningBlock:
        type = "reasoning"
        content = [{"type": "analysis", "text": "Step-by-step"}]
        summary = [{"type": "conclusion", "text": "Final answer"}]
        visibility = "internal"

    fake_response = DummyResponse(output=[FakeReasoningBlock()])
    fake_response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    provider = OpenAIProvider(api_key="test-key")
    chat_response = provider._convert_to_chat_response(fake_response)
    assert isinstance(chat_response, OpenAIChatResponse)

    reasoning_blocks = [block for block in chat_response.content if isinstance(block, ReasoningBlock)]
    assert reasoning_blocks, "Expected ReasoningBlock in chat response content"

    block = reasoning_blocks[0]
    assert block.content == [{"type": "analysis", "text": "Step-by-step"}]
    assert block.summary == [{"type": "conclusion", "text": "Final answer"}]
    assert block.visibility == "internal"

    assert hasattr(chat_response, "content_blocks")
    content_blocks = chat_response.content_blocks
    assert isinstance(content_blocks, list)
    thinking_blocks = [cb for cb in content_blocks if isinstance(cb, ThinkingContent)]
    assert thinking_blocks, "Expected ThinkingContent event block for streaming hooks"
    assert "Step-by-step" in thinking_blocks[0].text

    assert chat_response.text == "Step-by-step"


def test_reasoning_not_in_provider_response_content():
    class OutputText:
        type = "output_text"

        def __init__(self, text: str):
            self.text = text

    class MessageBlock:
        type = "message"

        def __init__(self, content):
            self.content = content

    class ReasoningBlockObj:
        type = "reasoning"

        def __init__(self, text: str):
            self.text = text

    provider = OpenAIProvider(api_key="test-key")
    message_block = MessageBlock([OutputText("visible answer")])
    reasoning_block = ReasoningBlockObj("internal chain of thought")

    content, tool_calls, content_blocks = provider._parse_response_output([message_block, reasoning_block])

    assert "visible answer" in content
    assert "internal chain of thought" not in content
    assert tool_calls == []
    assert any(isinstance(cb, ThinkingContent) for cb in content_blocks)


def test_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with synthetic results and emit event."""
    provider = OpenAIProvider(api_key="test-key")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "do_something", "arguments": '{"value": 1}'}},
            ],
        },
        {"role": "user", "content": "No tool result present"},
    ]

    asyncio.run(provider.complete(messages))

    # Should succeed (not raise validation error)
    provider.client.responses.create.assert_awaited_once()

    # Should not emit validation error
    assert all(event_name != "provider:validation_error" for event_name, _ in fake_coordinator.hooks.events)

    # Should emit repair event
    repair_events = [e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "openai"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_tool_calls_with_empty_arguments_are_filtered():
    provider = OpenAIProvider(api_key="test-key")
    tool_call_block = {
        "type": "tool_call",
        "id": "call-empty",
        "name": "noop",
        "input": {},
    }

    content, tool_calls, content_blocks = provider._parse_response_output([tool_call_block])

    assert content == ""
    assert tool_calls == []
    assert content_blocks == []
