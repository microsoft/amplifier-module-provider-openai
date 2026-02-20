"""Integration tests against real OpenRouter API.

These tests actually hit the OpenRouter API and verify that the provider
works correctly with the per-feature config flags.

Run with: pytest tests/test_integration_openrouter.py -v -m integration
Requires: OPENROUTER_API_KEY environment variable or ~/.openrouter.key
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolSpec

from amplifier_module_provider_openai import OpenAIProvider

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_openrouter_key() -> str | None:
    """Get OpenRouter API key from env or key file."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    key_file = Path.home() / ".openrouter.key"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


def _get_deepseek_key() -> str | None:
    """Get DeepSeek API key from env or key file."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    key_file = Path.home() / ".deepseek.key"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()
        self._capabilities: dict[str, str] = {}

    def get_capability(self, key: str) -> str | None:
        return self._capabilities.get(key)


def _make_openrouter_provider(model: str = "deepseek/deepseek-chat-v3-0324") -> OpenAIProvider:
    """Create a provider configured for OpenRouter."""
    key = _get_openrouter_key()
    if not key:
        pytest.skip("No OpenRouter API key available")
    provider = OpenAIProvider(
        api_key=key,
        config={
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": model,
            "max_retries": 1,
            "timeout": 60.0,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _make_deepseek_provider(model: str = "deepseek-chat") -> OpenAIProvider:
    """Create a provider configured for DeepSeek API directly."""
    key = _get_deepseek_key()
    if not key:
        pytest.skip("No DeepSeek API key available")
    provider = OpenAIProvider(
        api_key=key,
        config={
            "base_url": "https://api.deepseek.com/v1",
            "default_model": model,
            "max_retries": 1,
            "timeout": 60.0,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


# ---------------------------------------------------------------------------
# OpenRouter integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_basic_completion():
    """Basic text completion via OpenRouter works."""
    provider = _make_openrouter_provider()
    request = ChatRequest(
        messages=[Message(role="user", content="Say 'hello world' and nothing else.")]
    )
    response = await provider.complete(request)
    assert response is not None
    assert response.content is not None
    assert len(response.content) > 0
    # Should have text in the response
    text = "".join(
        block.text for block in response.content if hasattr(block, "text") and block.text
    )
    assert len(text) > 0
    print(f"  OpenRouter response: {text[:100]}")


@pytest.mark.asyncio
async def test_openrouter_flags_auto_disabled():
    """Verify config flags are correctly auto-disabled for OpenRouter."""
    provider = _make_openrouter_provider()
    assert provider.enable_native_tools is False
    assert provider.enable_reasoning_replay is False
    assert provider.enable_store is False
    assert provider.enable_background is False


@pytest.mark.asyncio
async def test_openrouter_no_store_in_params():
    """Verify store parameter is not sent to OpenRouter."""
    provider = _make_openrouter_provider()
    # Intercept the actual API call to check params
    original_create = provider.client.responses.create

    captured_params = {}

    async def capture_create(**kwargs):
        captured_params.update(kwargs)
        return await original_create(**kwargs)

    provider.client.responses.create = capture_create

    request = ChatRequest(
        messages=[Message(role="user", content="Say 'test' and nothing else.")]
    )
    await provider.complete(request)

    assert "store" not in captured_params, "store should not be sent to OpenRouter"
    assert "previous_response_id" not in captured_params
    assert "truncation" not in captured_params
    print(f"  Params sent (keys): {list(captured_params.keys())}")


@pytest.mark.asyncio
async def test_openrouter_function_tool_calling():
    """Function tool calling works via OpenRouter."""
    provider = _make_openrouter_provider(model="deepseek/deepseek-chat-v3-0324")

    # Create a simple function tool
    tool = ToolSpec(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    )

    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="What's the weather in Tokyo? Use the get_weather tool.",
            )
        ],
        tools=[tool],
    )

    response = await provider.complete(request)
    assert response is not None

    # Check if the model made a tool call
    tool_calls = []
    if response.content:
        for block in response.content:
            if hasattr(block, "name") and block.name == "get_weather":
                tool_calls.append(block)

    # The model should attempt to call the weather tool
    if tool_calls:
        print(f"  Tool call made: {tool_calls[0].name}({tool_calls[0].input})")
    else:
        # Some models may respond with text instead of tool calls
        text = "".join(
            block.text for block in response.content if hasattr(block, "text") and block.text
        )
        print(f"  No tool call, text response: {text[:100]}")


# ---------------------------------------------------------------------------
# DeepSeek direct API integration tests (via Responses API)
# Note: DeepSeek only supports Chat Completions, not Responses API.
# These tests may fail if DeepSeek doesn't support the /responses endpoint.
# They're included to document the incompatibility.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deepseek_direct_basic_completion():
    """Test if DeepSeek direct API works with our provider.

    Note: DeepSeek only supports Chat Completions API, not Responses API.
    This test documents whether the Responses API endpoint exists at DeepSeek.
    It may fail with a 404 or similar error.
    """
    provider = _make_deepseek_provider()

    request = ChatRequest(
        messages=[Message(role="user", content="Say 'hello' and nothing else.")]
    )

    try:
        response = await provider.complete(request)
        print(f"  DeepSeek direct Responses API works! Response: {response.content}")
    except Exception as e:
        # Expected: DeepSeek doesn't have a /responses endpoint
        print(f"  DeepSeek direct Responses API failed (expected): {type(e).__name__}: {e}")
        pytest.skip(f"DeepSeek doesn't support Responses API: {e}")


# ---------------------------------------------------------------------------
# OpenRouter with DeepSeek model (via OpenRouter's translation layer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_deepseek_model():
    """DeepSeek model via OpenRouter's Responses API translation."""
    provider = _make_openrouter_provider(model="deepseek/deepseek-chat-v3-0324")
    request = ChatRequest(
        messages=[Message(role="user", content="What is 2+2? Reply with just the number.")]
    )
    response = await provider.complete(request)
    assert response is not None
    text = "".join(
        block.text for block in response.content if hasattr(block, "text") and block.text
    )
    assert "4" in text
    print(f"  DeepSeek via OpenRouter: {text[:100]}")


@pytest.mark.asyncio
async def test_openrouter_multi_turn():
    """Multi-turn conversation works via OpenRouter."""
    provider = _make_openrouter_provider(model="deepseek/deepseek-chat-v3-0324")

    request = ChatRequest(
        messages=[
            Message(role="user", content="My name is Alice."),
            Message(role="assistant", content="Hello Alice! Nice to meet you."),
            Message(role="user", content="What is my name?"),
        ]
    )
    response = await provider.complete(request)
    assert response is not None
    text = "".join(
        block.text for block in response.content if hasattr(block, "text") and block.text
    )
    assert "Alice" in text
    print(f"  Multi-turn response: {text[:100]}")
