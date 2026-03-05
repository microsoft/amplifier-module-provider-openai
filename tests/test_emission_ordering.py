"""Tests for emission ordering: llm:response must fire before llm:response:raw."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OrderRecordingHooks:
    """Hooks that record event names in emission order."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = OrderRecordingHooks()


class DummyResponse:
    """Minimal response stub with model_dump support for raw debug emission."""

    def __init__(self, status="completed"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = status
        self.id = "resp_test"

    def model_dump(self):
        return {"id": self.id, "status": self.status}


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {
        "debug": True,
        "raw_debug": True,
        **config_overrides,
    }
    return OpenAIProvider(api_key="test-key", config=config)


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_response_raw_fires_after_response():
    """llm:response must appear before llm:response:raw in event order."""
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    asyncio.run(provider.complete(_simple_request()))

    event_names = [name for name, _ in fake_coordinator.hooks.events]

    assert "llm:response" in event_names, f"llm:response not found in {event_names}"
    assert "llm:response:raw" in event_names, (
        f"llm:response:raw not found in {event_names}"
    )

    response_idx = event_names.index("llm:response")
    raw_idx = event_names.index("llm:response:raw")

    assert response_idx < raw_idx, (
        f"llm:response (index {response_idx}) must fire before "
        f"llm:response:raw (index {raw_idx}). Events: {event_names}"
    )
