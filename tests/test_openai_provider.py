import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock
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

    messages = [Message(role="user", content="Hello")]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request, extended_thinking=True, thinking_budget_tokens=6000))

    provider.client.responses.create.assert_awaited()
    call_kwargs = provider.client.responses.create.await_args_list[0].kwargs

    assert call_kwargs["reasoning"]["effort"] == "high"
    # Default buffer is 1024 tokens, so expect budget + buffer to override defaults
    assert call_kwargs["max_output_tokens"] == 7024


def test_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with synthetic results and emit event."""
    provider = OpenAIProvider(api_key="test-key")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_1", name="do_something", input={"value": 1})],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

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
