"""Tests for emission ordering: llm:response must fire before llm:response:raw."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
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

    def __init__(self, status="completed", *, text="Hi", resp_id="resp_test"):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = status
        self.id = resp_id
        # Present on incomplete responses from the OpenAI API
        if status == "incomplete":
            self.incomplete_details = SimpleNamespace(reason="max_output_tokens")

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


@pytest.fixture()
def provider_with_fake_coordinator():
    """Set up an OpenAIProvider wired to a FakeCoordinator with a DummyResponse."""
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    return provider, fake_coordinator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_response_raw_fires_after_response(provider_with_fake_coordinator):
    """llm:response must appear before llm:response:raw in event order."""
    provider, fake_coordinator = provider_with_fake_coordinator

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


def test_full_emission_cascade_ordering(provider_with_fake_coordinator):
    """The three response emissions must flow: llm:response → llm:response:debug → llm:response:raw."""
    provider, fake_coordinator = provider_with_fake_coordinator

    asyncio.run(provider.complete(_simple_request()))

    event_names = [name for name, _ in fake_coordinator.hooks.events]

    assert "llm:response" in event_names, f"llm:response not found in {event_names}"
    assert "llm:response:debug" in event_names, (
        f"llm:response:debug not found in {event_names}"
    )
    assert "llm:response:raw" in event_names, (
        f"llm:response:raw not found in {event_names}"
    )

    response_idx = event_names.index("llm:response")
    debug_idx = event_names.index("llm:response:debug")
    raw_idx = event_names.index("llm:response:raw")

    assert response_idx < debug_idx < raw_idx, (
        f"Expected ordering response({response_idx}) < debug({debug_idx}) < raw({raw_idx}). "
        f"Events: {event_names}"
    )


def test_continuation_path_emission_ordering():
    """llm:response fires before llm:response:raw even after continuation (continuation_count > 0)."""
    provider = _make_provider()
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # First call returns incomplete, second call returns completed
    incomplete_resp = DummyResponse(
        status="incomplete", text="partial", resp_id="resp_1"
    )
    completed_resp = DummyResponse(status="completed", text="done", resp_id="resp_2")
    provider.client.responses.create = AsyncMock(
        side_effect=[incomplete_resp, completed_resp]
    )

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
        f"llm:response:raw (index {raw_idx}) in continuation path. "
        f"Events: {event_names}"
    )

    # Verify continuation_count is present in the llm:response payload
    response_payload = next(
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:response"
    )
    assert response_payload.get("continuation_count") == 1, (
        f"Expected continuation_count=1 in llm:response payload, "
        f"got {response_payload.get('continuation_count')}"
    )

    # Verify continuation is present in the llm:response:raw payload
    raw_payload = next(
        payload
        for name, payload in fake_coordinator.hooks.events
        if name == "llm:response:raw"
    )
    assert raw_payload.get("continuation") == 1, (
        f"Expected continuation=1 in llm:response:raw payload, "
        f"got {raw_payload.get('continuation')}"
    )
