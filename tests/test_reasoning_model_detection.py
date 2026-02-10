"""Tests for reasoning model capability detection and orphaned reasoning ID stripping.

Fix 2a: The include guard should fire based on model capability, not just
explicit reasoning config. Models like gpt-5.2-codex reason by default —
they never put "reasoning" in params unless the user sets reasoning_effort,
so encrypted_content was never requested and multi-turn reasoning broke.

Fix 2b: When reasoning IDs exist in metadata but encrypted_content was not
available (no ThinkingBlock content), the orphaned reasoning references must
be stripped to prevent 404 errors from the OpenAI API.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_openai import OpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyResponse:
    """Minimal response stub."""

    def __init__(self):
        self.output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi")],
            )
        ]
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        self.status = "completed"
        self.id = "resp_test"


def _make_provider(**config_overrides) -> OpenAIProvider:
    config = {"max_retries": 0, **config_overrides}
    provider = OpenAIProvider(api_key="test-key", config=config)
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())
    return provider


def _get_call_kwargs(provider: OpenAIProvider) -> dict:
    """Extract the kwargs from the last client.responses.create call."""
    return provider.client.responses.create.await_args_list[-1].kwargs


# ---------------------------------------------------------------------------
# Fix 2a: Include guard fires on model capability
# ---------------------------------------------------------------------------


def test_include_guard_fires_for_implicit_reasoning_model():
    """Model gpt-5.2-codex with no explicit reasoning_effort and store=false
    should still get include=[reasoning.encrypted_content] because it may
    reason by default."""
    provider = _make_provider(default_model="gpt-5.2-codex")  # store defaults to false
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,  # No explicit reasoning
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "Implicit reasoning model (gpt-5.2-codex) with store=false should have "
        f"'include' parameter, but it was missing. kwargs keys: {list(kwargs.keys())}"
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]


def test_include_guard_fires_for_o_series_model():
    """Model o4-mini with no explicit reasoning_effort and store=false
    should still get include=[reasoning.encrypted_content]."""
    provider = _make_provider(default_model="o4-mini")  # store defaults to false
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" in kwargs, (
        "o-series model (o4-mini) with store=false should have 'include' parameter"
    )
    assert kwargs["include"] == ["reasoning.encrypted_content"]


def test_include_guard_does_not_fire_for_non_reasoning_model():
    """Model gpt-4.1-mini with no reasoning should NOT get include parameter."""
    provider = _make_provider(default_model="gpt-4.1-mini")  # store defaults to false
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        reasoning_effort=None,
    )
    asyncio.run(provider.complete(request))

    kwargs = _get_call_kwargs(provider)
    assert "include" not in kwargs, (
        "Non-reasoning model (gpt-4.1-mini) should NOT have 'include' parameter, "
        f"but got include={kwargs.get('include')}"
    )


# ---------------------------------------------------------------------------
# Fix 2b: Orphaned reasoning ID stripping
# ---------------------------------------------------------------------------


def test_orphaned_reasoning_ids_stripped():
    """When reasoning IDs exist in metadata but no reasoning items can be
    reconstructed from content blocks, the orphaned references must be
    removed — they would cause 404 errors on the API.

    This tests the scenario where a ThinkingBlock has a reasoning_id but
    the encrypted_content is missing (None/empty) — the reasoning item
    gets built with just an id, which is an orphan that the API rejects.
    """
    provider = _make_provider()

    # Simulate an assistant message with a ThinkingBlock that has a
    # reasoning_id but NO encrypted_content (it was not requested)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me think...",
                    "content": [None, "rs_orphan_123"],  # [no encrypted_content, reasoning_id]
                },
                {
                    "type": "text",
                    "text": "The answer is 4.",
                },
            ],
            "metadata": {
                "openai:reasoning_items": ["rs_orphan_123"],  # ID present in metadata
            },
        },
        {"role": "user", "content": "Are you sure?"},
    ]

    result = provider._convert_messages(messages)

    # The reasoning item was built with just an id (no encrypted_content).
    # The orphan stripping should have removed it.
    reasoning_items = [item for item in result if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 0, (
        f"Orphaned reasoning items (no encrypted_content) should have been stripped, "
        f"but found: {reasoning_items}"
    )


def test_reasoning_continuity_with_encrypted_content():
    """When encrypted_content IS available in ThinkingBlock content, reasoning
    items should be reconstructed correctly (happy path)."""
    provider = _make_provider()

    # Simulate an assistant message with a proper ThinkingBlock containing
    # encrypted_content and reasoning_id in the content field
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me calculate...",
                    "content": ["encrypted_data_here", "rs_abc123"],  # [encrypted_content, reasoning_id]
                },
                {
                    "type": "text",
                    "text": "The answer is 4.",
                },
            ],
            "metadata": {
                "openai:reasoning_items": ["rs_abc123"],
            },
        },
        {"role": "user", "content": "Are you sure?"},
    ]

    result = provider._convert_messages(messages)

    # Reasoning items SHOULD be present
    reasoning_items = [item for item in result if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1, (
        f"Expected 1 reasoning item from encrypted content, but got {len(reasoning_items)}: {reasoning_items}"
    )
    assert reasoning_items[0]["id"] == "rs_abc123"
    assert reasoning_items[0]["encrypted_content"] == "encrypted_data_here"
    assert reasoning_items[0]["summary"] == [
        {"type": "summary_text", "text": "Let me calculate..."}
    ]
